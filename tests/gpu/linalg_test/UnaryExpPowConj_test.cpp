#include "gtest/gtest.h"

#include "gpu_test_tools.h"
#include "cytnx.hpp"

// Coverage for the #1003 step-11 typed GPU dispatch of the unary ops Exp, Pow and Conj (previously
// the legacy lii.cuExp_ii / cuPow_ii / cuConj_inplace_ii lookup tables, which had no GPU tests).
// Each GPU result is checked against the CPU result (dtype + value).

namespace UnaryExpPowConjTest {

  using cytnx::Tensor;
  namespace la = cytnx::linalg;

  ::testing::AssertionResult SameAsCpu(const char* what, const Tensor& gpu, const Tensor& cpu_ref) {
    Tensor g = gpu.to(cytnx::Device.cpu);
    if (g.dtype() != cpu_ref.dtype())
      return ::testing::AssertionFailure()
             << what << ": GPU dtype " << g.dtype() << " != CPU " << cpu_ref.dtype();
    if (!cytnx::gpu_test::AreNearlyEqTensor(g, cpu_ref, 1e-4))
      return ::testing::AssertionFailure() << what << ": GPU value != CPU";
    return ::testing::AssertionSuccess();
  }

  // Small deterministic real-valued input (in [-0.5, 0.5]) cast to `dtype`, on the GPU. Kept small
  // so exp/pow do not overflow; a complex dtype gets zero imaginary part (still exercises the
  // complex kernel path).
  Tensor SmallInput(unsigned int dtype) {
    Tensor base = cytnx::arange(0, 64, 1, cytnx::Type.Double);  // [0..63]
    base = (base - 31.5) / 63.0;  // [-0.5, 0.5]
    return base.astype(dtype).to(cytnx::Device.cuda);
  }

  const std::vector<unsigned int> kFloatish = {cytnx::Type.Double, cytnx::Type.Float,
                                               cytnx::Type.ComplexDouble, cytnx::Type.ComplexFloat};

  TEST(GpuUnary, exp_matches_cpu) {
    for (auto dt : kFloatish) {
      Tensor t = SmallInput(dt);
      SCOPED_TRACE("Exp dtype " + std::to_string(dt));
      EXPECT_TRUE(SameAsCpu("Exp", la::Exp(t), la::Exp(t.to(cytnx::Device.cpu))));

      Tensor gi = t.clone(), ci = t.to(cytnx::Device.cpu);
      la::Exp_(gi);
      la::Exp_(ci);
      EXPECT_TRUE(SameAsCpu("Exp_", gi, ci));
    }
  }

  TEST(GpuUnary, pow_matches_cpu) {
    for (auto dt : kFloatish) {
      Tensor t = SmallInput(dt);
      SCOPED_TRACE("Pow dtype " + std::to_string(dt));
      EXPECT_TRUE(SameAsCpu("Pow", la::Pow(t, 2.0), la::Pow(t.to(cytnx::Device.cpu), 2.0)));

      Tensor gi = t.clone(), ci = t.to(cytnx::Device.cpu);
      la::Pow_(gi, 3.0);
      la::Pow_(ci, 3.0);
      EXPECT_TRUE(SameAsCpu("Pow_", gi, ci));
    }
  }

  TEST(GpuUnary, conj_matches_cpu) {
    for (auto dt : {cytnx::Type.ComplexDouble, cytnx::Type.ComplexFloat}) {
      Tensor t = Tensor({64}, dt).to(cytnx::Device.cuda);
      cytnx::gpu_test::InitTensorUniform(t, /*seed=*/7);  // genuine complex (nonzero imag)
      SCOPED_TRACE("Conj dtype " + std::to_string(dt));
      EXPECT_TRUE(SameAsCpu("Conj", la::Conj(t), la::Conj(t.to(cytnx::Device.cpu))));

      Tensor gi = t.clone(), ci = t.to(cytnx::Device.cpu);
      la::Conj_(gi);
      la::Conj_(ci);
      EXPECT_TRUE(SameAsCpu("Conj_", gi, ci));
    }
  }

}  // namespace UnaryExpPowConjTest
