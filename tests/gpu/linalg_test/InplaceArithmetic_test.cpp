#include <algorithm>
#include <cmath>

#include "gtest/gtest.h"

#include "gpu_test_tools.h"
#include "cytnx.hpp"

// Coverage for the #1003 GPU in-place migration (iAdd/iSub/iMul/iDiv routed through the typed
// cu*_dispatch instead of the legacy cuAri_ii table). The per-op *_test.cpp suites only exercise
// contiguous, dtype-preserving tensor<>tensor in-place; these tests target the branches that the
// migration newly handles:
//   - dtype-widening in-place (result narrowed back to the LHS dtype),
//   - mixed complex/real in-place (ComplexFloat op= Double stays ComplexFloat),
//   - non-contiguous scalar in-place (layout-preserving, the #988 regression),
//   - non-contiguous tensor<>tensor in-place (previously corrupting for iAdd/iSub and a hard error
//     for iMul/iDiv; now produces correct values in a contiguous LHS).
// The reference is the CPU path, run on host copies of the same inputs.

namespace InplaceArithmeticTest {

  using cytnx::Tensor;
  namespace la = cytnx::linalg;

  // op_code: 0 iAdd, 1 iMul, 2 iSub, 3 iDiv (matches the CPU DispatchInplaceArithmeticCPU codes).
  void ApplyInplace(int op_code, Tensor& lhs, const Tensor& rhs) {
    switch (op_code) {
      case 0:
        la::iAdd(lhs, rhs);
        break;
      case 1:
        la::iMul(lhs, rhs);
        break;
      case 2:
        la::iSub(lhs, rhs);
        break;
      case 3:
        la::iDiv(lhs, rhs);
        break;
    }
  }

  const std::vector<int> kOps = {0, 1, 2, 3};

  // GPU in-place result must match the CPU in-place result exactly (dtype + layout + values).
  ::testing::AssertionResult CheckAgainstCpu(int op_code, const Tensor& lhs0, const Tensor& rhs0) {
    Tensor gpu_lhs = lhs0.clone();
    ApplyInplace(op_code, gpu_lhs, rhs0);

    Tensor cpu_lhs = lhs0.to(cytnx::Device.cpu);
    ApplyInplace(op_code, cpu_lhs, rhs0.to(cytnx::Device.cpu));

    Tensor gpu_lhs_cpu = gpu_lhs.to(cytnx::Device.cpu);
    if (gpu_lhs_cpu.dtype() != cpu_lhs.dtype())
      return ::testing::AssertionFailure() << "op " << op_code << ": GPU dtype "
                                           << gpu_lhs_cpu.dtype() << " != CPU " << cpu_lhs.dtype();
    if (!cytnx::TestTools::AreNearlyEqTensor(gpu_lhs_cpu, cpu_lhs, 1e-5))
      return ::testing::AssertionFailure() << "op " << op_code << ": GPU value/layout != CPU";
    return ::testing::AssertionSuccess();
  }

  // Same, but comparing logical values only (contiguous-ized). Used for non-contiguous
  // tensor<>tensor in-place, where the GPU result is intentionally contiguous while CPU stays
  // strided.
  ::testing::AssertionResult CheckValuesAgainstCpu(int op_code, const Tensor& lhs0,
                                                   const Tensor& rhs0) {
    Tensor gpu_lhs = lhs0.clone();
    ApplyInplace(op_code, gpu_lhs, rhs0);

    Tensor cpu_lhs = lhs0.to(cytnx::Device.cpu);
    ApplyInplace(op_code, cpu_lhs, rhs0.to(cytnx::Device.cpu));

    Tensor gpu_c = gpu_lhs.to(cytnx::Device.cpu).contiguous();
    Tensor cpu_c = cpu_lhs.contiguous();
    if (gpu_c.dtype() != cpu_c.dtype())
      return ::testing::AssertionFailure()
             << "op " << op_code << ": GPU dtype " << gpu_c.dtype() << " != CPU " << cpu_c.dtype();
    if (!cytnx::TestTools::AreNearlyEqTensor(gpu_c, cpu_c, 1e-5))
      return ::testing::AssertionFailure() << "op " << op_code << ": GPU values != CPU";
    return ::testing::AssertionSuccess();
  }

  // A nonzero singleton (value 3) of the requested dtype, on the GPU. Deterministic so iDiv never
  // divides by zero.
  Tensor NonzeroScalar(unsigned int dtype) {
    return cytnx::arange(3, 4, 1, dtype).to(cytnx::Device.cuda);  // shape {1}, value 3
  }

  // Relative element-wise comparison. Used for mixed real/complex in-place: the GPU computes in the
  // promoted dtype (per #1003) then narrows to the LHS dtype, while CPU's std::complex op= narrows
  // the RHS to the LHS scalar type first, so the two agree only to float rounding.
  bool NearlyEqRel(const Tensor& a_dev, const Tensor& b_dev, double rel_tol) {
    Tensor a = a_dev.to(cytnx::Device.cpu).contiguous().astype(cytnx::Type.ComplexDouble);
    Tensor b = b_dev.to(cytnx::Device.cpu).contiguous().astype(cytnx::Type.ComplexDouble);
    cytnx::cytnx_uint64 n = a.storage().size();
    for (cytnx::cytnx_uint64 i = 0; i < n; i++) {
      cytnx::cytnx_complex128 x = a.storage().at<cytnx::cytnx_complex128>(i);
      cytnx::cytnx_complex128 y = b.storage().at<cytnx::cytnx_complex128>(i);
      double denom = std::max(std::abs(y), 1e-9);
      if (std::abs(x - y) / denom > rel_tol) return false;
    }
    return true;
  }

  ::testing::AssertionResult CheckAgainstCpuRel(int op_code, const Tensor& lhs0, const Tensor& rhs0,
                                                double rel_tol) {
    Tensor gpu_lhs = lhs0.clone();
    ApplyInplace(op_code, gpu_lhs, rhs0);
    Tensor cpu_lhs = lhs0.to(cytnx::Device.cpu);
    ApplyInplace(op_code, cpu_lhs, rhs0.to(cytnx::Device.cpu));
    if (gpu_lhs.dtype() != cpu_lhs.dtype())
      return ::testing::AssertionFailure() << "op " << op_code << ": dtype mismatch";
    if (!NearlyEqRel(gpu_lhs, cpu_lhs, rel_tol))
      return ::testing::AssertionFailure() << "op " << op_code << ": GPU vs CPU exceeds rel tol";
    return ::testing::AssertionSuccess();
  }

  // Contiguous dtype-widening in-place: e.g. Float LHS op= Double RHS must stay Float (narrowed
  // back). Integer RHS is skipped for iDiv to avoid integer division by a random zero.
  TEST(GpuInplaceArithmetic, dtype_widening_stays_lhs) {
    struct Pair {
      unsigned int l, r;
    };
    const std::vector<Pair> pairs = {
      {cytnx::Type.Float, cytnx::Type.Double},
      {cytnx::Type.Int32, cytnx::Type.Int64},
      {cytnx::Type.Int16, cytnx::Type.Double},
      {cytnx::Type.ComplexFloat, cytnx::Type.ComplexDouble},
    };
    for (int op : kOps) {
      for (auto p : pairs) {
        if (op == 3 && !cytnx::Type.is_float(p.r)) continue;  // avoid integer div-by-zero
        Tensor lhs = Tensor({4, 5}, p.l).to(cytnx::Device.cuda);
        Tensor rhs = Tensor({4, 5}, p.r).to(cytnx::Device.cuda);
        cytnx::TestTools::InitTensorUniform(lhs, /*seed=*/1);
        cytnx::TestTools::InitTensorUniform(rhs, /*seed=*/2);
        SCOPED_TRACE("op " + std::to_string(op) + " ldtype " + std::to_string(p.l) + " rdtype " +
                     std::to_string(p.r));
        Tensor probe = lhs.clone();
        ApplyInplace(op, probe, rhs);
        EXPECT_EQ(probe.dtype(), p.l);  // in-place keeps the LHS dtype
        EXPECT_TRUE(CheckAgainstCpuRel(op, lhs, rhs, 1e-4));
      }
    }
  }

  // Mixed complex/real in-place: ComplexFloat op= Double promotes to ComplexDouble internally but
  // must be stored back as ComplexFloat. Double RHS is nonzero-random so iDiv is safe.
  TEST(GpuInplaceArithmetic, complex_lhs_real_rhs_stays_complex_float) {
    for (int op : kOps) {
      Tensor lhs = Tensor({6}, cytnx::Type.ComplexFloat).to(cytnx::Device.cuda);
      Tensor rhs = Tensor({6}, cytnx::Type.Double).to(cytnx::Device.cuda);
      cytnx::TestTools::InitTensorUniform(lhs, 3);
      cytnx::TestTools::InitTensorUniform(rhs, 4);
      Tensor probe = lhs.clone();
      ApplyInplace(op, probe, rhs);
      EXPECT_EQ(probe.dtype(), cytnx::Type.ComplexFloat);
      EXPECT_TRUE(CheckAgainstCpuRel(op, lhs, rhs, 1e-4));
    }
  }

  // Non-contiguous scalar in-place (the #988 regression): a permuted LHS scaled/offset by a
  // singleton RHS must stay correct and preserve the (non-contiguous) layout.
  TEST(GpuInplaceArithmetic, noncontiguous_scalar) {
    const std::vector<unsigned int> dtypes = {cytnx::Type.Double, cytnx::Type.ComplexDouble,
                                              cytnx::Type.Int64};
    for (int op : kOps) {
      for (auto dt : dtypes) {
        Tensor base = Tensor({3, 4, 5}, dt).to(cytnx::Device.cuda);
        cytnx::TestTools::InitTensorUniform(base, 5);
        Tensor lhs = base.permute({2, 0, 1});  // non-contiguous view
        EXPECT_FALSE(lhs.is_contiguous());
        Tensor scalar = NonzeroScalar(dt);
        SCOPED_TRACE("op " + std::to_string(op) + " dtype " + std::to_string(dt));
        EXPECT_TRUE(CheckAgainstCpu(op, lhs, scalar));
      }
    }
  }

  // Non-contiguous tensor<>tensor in-place: correct logical values (GPU result is contiguous). Only
  // float/complex dtypes, so random RHS is nonzero and iDiv is safe.
  TEST(GpuInplaceArithmetic, noncontiguous_tensor_values) {
    const std::vector<unsigned int> dtypes = {cytnx::Type.Double, cytnx::Type.ComplexDouble,
                                              cytnx::Type.Float};
    for (int op : kOps) {
      for (auto dt : dtypes) {
        Tensor lbase = Tensor({3, 4, 5}, dt).to(cytnx::Device.cuda);
        Tensor rbase = Tensor({3, 4, 5}, dt).to(cytnx::Device.cuda);
        cytnx::TestTools::InitTensorUniform(lbase, 7);
        cytnx::TestTools::InitTensorUniform(rbase, 8);
        Tensor lhs = lbase.permute({1, 2, 0});
        Tensor rhs = rbase.permute({1, 2, 0});
        EXPECT_FALSE(lhs.is_contiguous());
        SCOPED_TRACE("op " + std::to_string(op) + " dtype " + std::to_string(dt));
        EXPECT_TRUE(CheckValuesAgainstCpu(op, lhs, rhs));
      }
    }
  }

  // Independent (hand-computed) values, not derived from another Cytnx path.
  TEST(GpuInplaceArithmetic, independent_values) {
    // Double LHS += Double scalar 2.5  ->  [2.5, 3.5, 4.5, 5.5]
    Tensor a = cytnx::arange(0, 4, 1, cytnx::Type.Double, cytnx::Device.cuda);
    Tensor s = Tensor({1}, cytnx::Type.Double);
    s.at<cytnx::cytnx_double>({0}) = 2.5;
    la::iAdd(a, s.to(cytnx::Device.cuda));
    Tensor a_cpu = a.to(cytnx::Device.cpu);
    for (cytnx::cytnx_uint64 i = 0; i < 4; i++)
      EXPECT_DOUBLE_EQ(a_cpu.at<cytnx::cytnx_double>({i}), double(i) + 2.5);

    // Int64 LHS /= Int64 scalar 2  ->  integer division, stays Int64
    Tensor b = cytnx::arange(0, 8, 1, cytnx::Type.Int64, cytnx::Device.cuda);
    Tensor t = Tensor({1}, cytnx::Type.Int64);
    t.at<cytnx::cytnx_int64>({0}) = 2;
    la::iDiv(b, t.to(cytnx::Device.cuda));
    EXPECT_EQ(b.dtype(), cytnx::Type.Int64);
    Tensor b_cpu = b.to(cytnx::Device.cpu);
    const cytnx::cytnx_int64 expected[8] = {0, 0, 1, 1, 2, 2, 3, 3};
    for (cytnx::cytnx_uint64 i = 0; i < 8; i++)
      EXPECT_EQ(b_cpu.at<cytnx::cytnx_int64>({i}), expected[i]);
  }

}  // namespace InplaceArithmeticTest
