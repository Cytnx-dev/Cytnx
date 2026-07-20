#include <cmath>
#include <complex>
#include <vector>

#include "gtest/gtest.h"

#include "gpu_test_tools.h"
#include "cytnx.hpp"

// Coverage for the #1003 step-11 typed GPU dispatch of the unary ops Exp, Pow and Conj (previously
// the legacy lii.cuExp_ii / cuPow_ii / cuConj_inplace_ii lookup tables, which had no GPU tests).
// GpuUnary.matches_std_library checks each op against INDEPENDENT std-library expected values (per
// CLAUDE.md, so a shared CPU/GPU semantic bug cannot hide behind a GPU-vs-CPU compare); the
// *_matches_cpu tests additionally cross-check the CPU dispatch across every floating dtype.

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

  // Independent expected values from the C++ standard library (NOT the Cytnx CPU path), per
  // CLAUDE.md. Concrete real + complex inputs built on the CPU; the GPU result must match
  // std::exp / std::pow / std::conj directly.
  TEST(GpuUnary, matches_std_library) {
    using cytnx::cytnx_complex128;
    using cytnx::cytnx_uint64;

    // real Double: Exp and Pow(., 3)
    {
      const std::vector<double> in = {-0.75, -0.25, 0.0, 0.5, 1.25};
      Tensor t = cytnx::zeros({static_cast<cytnx_uint64>(in.size())}, cytnx::Type.Double);
      for (cytnx_uint64 i = 0; i < in.size(); ++i) t.at<double>({i}) = in[i];
      Tensor ge = la::Exp(t.to(cytnx::Device.cuda)).to(cytnx::Device.cpu);
      Tensor gp = la::Pow(t.to(cytnx::Device.cuda), 3.0).to(cytnx::Device.cpu);
      for (cytnx_uint64 i = 0; i < in.size(); ++i) {
        EXPECT_NEAR(ge.at<double>({i}), std::exp(in[i]), 1e-12);
        EXPECT_NEAR(gp.at<double>({i}), std::pow(in[i], 3.0), 1e-12);
      }
    }
    // ComplexDouble: Exp, Pow(., 2), Conj
    {
      const std::vector<std::complex<double>> in = {
        {0.5, -0.3}, {-0.2, 0.7}, {1.0, 0.0}, {0.0, 1.0}};
      Tensor t = cytnx::zeros({static_cast<cytnx_uint64>(in.size())}, cytnx::Type.ComplexDouble);
      for (cytnx_uint64 i = 0; i < in.size(); ++i)
        t.at<cytnx_complex128>({i}) = cytnx_complex128(in[i].real(), in[i].imag());
      Tensor ge = la::Exp(t.to(cytnx::Device.cuda)).to(cytnx::Device.cpu);
      Tensor gp = la::Pow(t.to(cytnx::Device.cuda), 2.0).to(cytnx::Device.cpu);
      Tensor gc = la::Conj(t.to(cytnx::Device.cuda)).to(cytnx::Device.cpu);
      for (cytnx_uint64 i = 0; i < in.size(); ++i) {
        const std::complex<double> e = std::exp(in[i]), p = std::pow(in[i], 2.0),
                                   c = std::conj(in[i]);
        const cytnx_complex128 ae = ge.at<cytnx_complex128>({i});
        const cytnx_complex128 ap = gp.at<cytnx_complex128>({i});
        const cytnx_complex128 ac = gc.at<cytnx_complex128>({i});
        EXPECT_NEAR(ae.real(), e.real(), 1e-12);
        EXPECT_NEAR(ae.imag(), e.imag(), 1e-12);
        EXPECT_NEAR(ap.real(), p.real(), 1e-12);
        EXPECT_NEAR(ap.imag(), p.imag(), 1e-12);
        EXPECT_NEAR(ac.real(), c.real(), 1e-12);
        EXPECT_NEAR(ac.imag(), c.imag(), 1e-12);
      }
    }
  }

  // Abs of a floating -0.0 must clear the sign bit. The fix routes floating through
  // cuda::std::abs; the old `x < 0 ? -x : x` returned -0.0 because -0.0 < 0 is false.
  TEST(GpuUnary, abs_negative_zero) {
    {
      Tensor t = cytnx::zeros({2}, cytnx::Type.Double);
      t.at<double>({0}) = -0.0;
      t.at<double>({1}) = -3.5;
      Tensor g = la::Abs(t.to(cytnx::Device.cuda)).to(cytnx::Device.cpu);
      EXPECT_FALSE(std::signbit(g.at<double>({0}))) << "Abs(-0.0) must be +0.0";
      EXPECT_DOUBLE_EQ(g.at<double>({0}), 0.0);
      EXPECT_DOUBLE_EQ(g.at<double>({1}), 3.5);
    }
    {
      Tensor t = cytnx::zeros({2}, cytnx::Type.Float);
      t.at<float>({0}) = -0.0f;
      t.at<float>({1}) = -3.5f;
      Tensor g = la::Abs(t.to(cytnx::Device.cuda)).to(cytnx::Device.cpu);
      EXPECT_FALSE(std::signbit(g.at<float>({0}))) << "Abs(-0.0f) must be +0.0f";
      EXPECT_FLOAT_EQ(g.at<float>({1}), 3.5f);
    }
  }

}  // namespace UnaryExpPowConjTest
