#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "gpu_test_tools.h"

// GPU Outer correctness. Outer(a, b) for rank-1 a (len m) and rank-1 b (len n)
// gives an m x n tensor with out[i, j] = a[i] * b[j]. Every expected value is
// computed independently from that definition (literals or a per-element
// recompute from the raw inputs), never by comparing against another Cytnx
// path. The mixed ComplexFloat x Double case mirrors the CPU
// DtypePromotion.OuterComplexfloatDouble regression: the result must be
// ComplexDouble, computed and stored through that output type. Outer on the GPU
// now routes through cuKron (reshaped) after #1003 retired the per-dtype
// cuOuter_ii table, so the Int16 diagonal that used to segfault on a missing
// dispatch row (#1099) is also covered.
//
// Inputs are built on the CPU and moved with .to(Device.cuda) (GPU arange
// ignores a non-unit step for real dtypes, #1070).
namespace cytnx {
  namespace gpu_test {
    namespace {

      TEST(GpuOuter, DoubleMatchesHandComputed) {
        Tensor a = zeros({3}, Type.Double);
        a.at<cytnx_double>({0}) = -1.5;
        a.at<cytnx_double>({1}) = 0.0;
        a.at<cytnx_double>({2}) = 2.25;
        Tensor b = zeros({2}, Type.Double);
        b.at<cytnx_double>({0}) = 4.0;
        b.at<cytnx_double>({1}) = -0.5;

        Tensor out = linalg::Outer(a.to(Device.cuda), b.to(Device.cuda)).to(Device.cpu);

        ASSERT_EQ(out.dtype(), Type.Double);
        ASSERT_EQ(out.shape(), (std::vector<cytnx_uint64>{3, 2}));
        const double expected[3][2] = {{-6.0, 0.75}, {0.0, -0.0}, {9.0, -1.125}};
        for (cytnx_uint64 i = 0; i < 3; i++)
          for (cytnx_uint64 j = 0; j < 2; j++)
            EXPECT_DOUBLE_EQ(out.at<cytnx_double>({i, j}), expected[i][j])
              << "at (" << i << "," << j << ")";
      }

      TEST(GpuOuter, ComplexDoubleMatchesHandComputed) {
        Tensor a = zeros({2}, Type.ComplexDouble);
        a.at<cytnx_complex128>({0}) = cytnx_complex128(1, 2);
        a.at<cytnx_complex128>({1}) = cytnx_complex128(-3, 1);
        Tensor b = zeros({2}, Type.ComplexDouble);
        b.at<cytnx_complex128>({0}) = cytnx_complex128(0, -1);
        b.at<cytnx_complex128>({1}) = cytnx_complex128(2, 2);

        Tensor out = linalg::Outer(a.to(Device.cuda), b.to(Device.cuda)).to(Device.cpu);

        ASSERT_EQ(out.dtype(), Type.ComplexDouble);
        ASSERT_EQ(out.shape(), (std::vector<cytnx_uint64>{2, 2}));
        // (1+2i)*(0-1i) = 2-1i ; (1+2i)*(2+2i) = -2+6i
        // (-3+1i)*(0-1i)= 1+3i ; (-3+1i)*(2+2i)= -8-4i
        auto near = [&](cytnx_uint64 i, cytnx_uint64 j, double re, double im) {
          auto v = out.at<cytnx_complex128>({i, j});
          EXPECT_NEAR(v.real(), re, 1e-12) << "real at (" << i << "," << j << ")";
          EXPECT_NEAR(v.imag(), im, 1e-12) << "imag at (" << i << "," << j << ")";
        };
        near(0, 0, 2, -1);
        near(0, 1, -2, 6);
        near(1, 0, 1, 3);
        near(1, 1, -8, -4);
      }

      // The promotion discriminator (mirrors CPU DtypePromotion.OuterComplexfloatDouble):
      // ComplexFloat (x) Double -> ComplexDouble, full double precision.
      TEST(GpuOuter, MixedComplexFloatDoublePromotesToComplexDouble) {
        Tensor a = zeros({2}, Type.ComplexFloat);
        a.at<cytnx_complex64>({0}) = cytnx_complex64(1, 1);
        a.at<cytnx_complex64>({1}) = cytnx_complex64(2, 0);
        Tensor b = zeros({2}, Type.Double);
        b.at<cytnx_double>({0}) = 3;
        b.at<cytnx_double>({1}) = 0.5;

        Tensor out = linalg::Outer(a.to(Device.cuda), b.to(Device.cuda)).to(Device.cpu);

        ASSERT_EQ(out.dtype(), Type.ComplexDouble);
        ASSERT_EQ(out.shape(), (std::vector<cytnx_uint64>{2, 2}));
        auto near = [&](cytnx_uint64 i, cytnx_uint64 j, double re, double im) {
          auto v = out.at<cytnx_complex128>({i, j});
          EXPECT_NEAR(v.real(), re, 1e-6) << "real at (" << i << "," << j << ")";
          EXPECT_NEAR(v.imag(), im, 1e-6) << "imag at (" << i << "," << j << ")";
        };
        near(0, 0, 3, 3);  // (1+1i)*3
        near(0, 1, 0.5, 0.5);  // (1+1i)*0.5
        near(1, 0, 6, 0);  // 2*3
        near(1, 1, 1, 0);  // 2*0.5
      }

      // Regression for #1099: Int16 (x) Int16 Outer used to hit a null dispatch
      // row and segfault; it now routes through cuKron. Hand-computed, negative
      // values included.
      TEST(GpuOuter, Int16DiagonalNoLongerSegfaults) {
        Tensor a = zeros({2}, Type.Int16);
        a.at<cytnx_int16>({0}) = 3;
        a.at<cytnx_int16>({1}) = -2;
        Tensor b = zeros({2}, Type.Int16);
        b.at<cytnx_int16>({0}) = 4;
        b.at<cytnx_int16>({1}) = 5;

        Tensor out = linalg::Outer(a.to(Device.cuda), b.to(Device.cuda)).to(Device.cpu);
        ASSERT_EQ(out.dtype(), Type.Int16);
        EXPECT_EQ(out.at<cytnx_int16>({0, 0}), 12);
        EXPECT_EQ(out.at<cytnx_int16>({0, 1}), 15);
        EXPECT_EQ(out.at<cytnx_int16>({1, 0}), -8);
        EXPECT_EQ(out.at<cytnx_int16>({1, 1}), -10);
      }

      // Broad cross-check: GPU Outer vs CPU Outer over every real/complex dtype.
      // InitTensorUniform draws from [-1000, 1000], so products reach ~1e6; at
      // that magnitude a complex-float multiply diverges from the CPU one at the
      // float32 ULP (~0.06), hence the looser ComplexFloat tolerance (matches
      // Mul_test's convention). A single real multiply is bit-identical.
      TEST(GpuOuter, GpuMatchesCpuAllDtypes) {
        for (auto dtype : dtype_list) {
          if (dtype == Type.Bool) continue;  // Outer has no Bool kernel
          SCOPED_TRACE("dtype " + std::to_string(dtype));
          const cytnx_double tol = (dtype == Type.ComplexFloat) ? 0.1 : 1e-6;
          Tensor a({5}, dtype);
          Tensor b({7}, dtype);
          InitTensorUniform(a, /*seed=*/3);
          InitTensorUniform(b, /*seed=*/4);
          Tensor expected = linalg::Outer(a, b);
          Tensor gpu = linalg::Outer(a.to(Device.cuda), b.to(Device.cuda)).to(Device.cpu);
          EXPECT_EQ(gpu.dtype(), expected.dtype());
          EXPECT_TRUE(AreNearlyEqTensor(gpu, expected, tol));
        }
      }

    }  // namespace
  }  // namespace gpu_test
}  // namespace cytnx
