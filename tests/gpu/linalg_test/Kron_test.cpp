#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "gpu_test_tools.h"

// GPU Kron correctness. Every expected value here is computed independently
// from the Kronecker-product definition
//
//     out[i, j] = A[i / bR, j / bC] * B[i % bR, j % bC]
//
// (A is aR x aC, B is bR x bC, out is (aR*bR) x (aC*bC)), written out as
// literals or recomputed from the raw inputs -- never by comparing against
// another Cytnx path that could share the same bug. The mixed
// ComplexFloat x Double case is the #984/#999 promotion discriminator: the
// output must be ComplexDouble (not the narrower ComplexFloat) and carry full
// double precision. #1003 retired the Type_list_gpu / type_promote_gpu_t
// promotion trait, so Kron's GPU path computes the promoted type with the
// shared host type_promote_from_pointer_t and maps to the CUDA-native kernel
// type via to_cuda_t at the launch boundary -- exactly what that case
// exercises.
//
// Inputs are built on the CPU and copied to the GPU with .to(Device.cuda) --
// GPU arange ignores a non-unit step for real dtypes (#1070), so never build
// GPU inputs with arange directly.
namespace cytnx {
  namespace gpu_test {
    namespace {

      // 2x2 Kron 2x2 -> 4x4, real. A = [[1,2],[3,4]], B = [[0,5],[6,7]].
      // Kron(A,B) laid out by hand:
      //   [ 0  5   0 10 ]
      //   [ 6  7  12 14 ]
      //   [ 0 15   0 20 ]
      //   [18 21  24 28 ]
      TEST(GpuKron, DoubleMatchesHandComputed) {
        Tensor a = zeros({2, 2}, Type.Double);
        a.at<cytnx_double>({0, 0}) = 1;
        a.at<cytnx_double>({0, 1}) = 2;
        a.at<cytnx_double>({1, 0}) = 3;
        a.at<cytnx_double>({1, 1}) = 4;
        Tensor b = zeros({2, 2}, Type.Double);
        b.at<cytnx_double>({0, 0}) = 0;
        b.at<cytnx_double>({0, 1}) = 5;
        b.at<cytnx_double>({1, 0}) = 6;
        b.at<cytnx_double>({1, 1}) = 7;

        Tensor out = linalg::Kron(a.to(Device.cuda), b.to(Device.cuda)).to(Device.cpu);

        ASSERT_EQ(out.dtype(), Type.Double);
        ASSERT_EQ(out.shape(), (std::vector<cytnx_uint64>{4, 4}));
        const double expected[4][4] = {
          {0, 5, 0, 10}, {6, 7, 12, 14}, {0, 15, 0, 20}, {18, 21, 24, 28}};
        for (cytnx_uint64 i = 0; i < 4; i++)
          for (cytnx_uint64 j = 0; j < 4; j++)
            EXPECT_DOUBLE_EQ(out.at<cytnx_double>({i, j}), expected[i][j])
              << "at (" << i << "," << j << ")";
      }

      // Fractional and negative values, recomputed per-element from the raw
      // inputs (independent of Kron's own layout).
      TEST(GpuKron, DoubleFractionalNegative) {
        Tensor a = zeros({3, 2}, Type.Double);
        a.at<cytnx_double>({0, 0}) = -1.5;
        a.at<cytnx_double>({0, 1}) = 0.25;
        a.at<cytnx_double>({1, 0}) = 2.0;
        a.at<cytnx_double>({1, 1}) = -3.5;
        a.at<cytnx_double>({2, 0}) = 0.0;
        a.at<cytnx_double>({2, 1}) = 4.75;
        Tensor b = zeros({2, 3}, Type.Double);
        b.at<cytnx_double>({0, 0}) = 1.0;
        b.at<cytnx_double>({0, 1}) = -2.0;
        b.at<cytnx_double>({0, 2}) = 0.5;
        b.at<cytnx_double>({1, 0}) = -0.25;
        b.at<cytnx_double>({1, 1}) = 3.0;
        b.at<cytnx_double>({1, 2}) = -4.0;

        Tensor out = linalg::Kron(a.to(Device.cuda), b.to(Device.cuda)).to(Device.cpu);

        ASSERT_EQ(out.dtype(), Type.Double);
        ASSERT_EQ(out.shape(), (std::vector<cytnx_uint64>{6, 6}));
        const cytnx_uint64 bR = 2, bC = 3;
        for (cytnx_uint64 i = 0; i < 6; i++)
          for (cytnx_uint64 j = 0; j < 6; j++) {
            const double expected =
              a.at<cytnx_double>({i / bR, j / bC}) * b.at<cytnx_double>({i % bR, j % bC});
            EXPECT_DOUBLE_EQ(out.at<cytnx_double>({i, j}), expected)
              << "at (" << i << "," << j << ")";
          }
      }

      // Complex x complex, hand-computed. A = [[1+1i, 2-1i]], B = [[0+1i],[3+0i]].
      TEST(GpuKron, ComplexDoubleMatchesHandComputed) {
        Tensor a = zeros({1, 2}, Type.ComplexDouble);
        a.at<cytnx_complex128>({0, 0}) = cytnx_complex128(1, 1);
        a.at<cytnx_complex128>({0, 1}) = cytnx_complex128(2, -1);
        Tensor b = zeros({2, 1}, Type.ComplexDouble);
        b.at<cytnx_complex128>({0, 0}) = cytnx_complex128(0, 1);
        b.at<cytnx_complex128>({1, 0}) = cytnx_complex128(3, 0);

        Tensor out = linalg::Kron(a.to(Device.cuda), b.to(Device.cuda)).to(Device.cpu);

        ASSERT_EQ(out.dtype(), Type.ComplexDouble);
        ASSERT_EQ(out.shape(), (std::vector<cytnx_uint64>{2, 2}));
        // out[i,j] = a[0, j] * b[i, 0]
        // (1+1i)*(0+1i) = -1+1i ; (2-1i)*(0+1i) = 1+2i
        // (1+1i)*(3+0i) =  3+3i ; (2-1i)*(3+0i) = 6-3i
        auto near = [&](cytnx_uint64 i, cytnx_uint64 j, double re, double im) {
          auto v = out.at<cytnx_complex128>({i, j});
          EXPECT_NEAR(v.real(), re, 1e-12) << "real at (" << i << "," << j << ")";
          EXPECT_NEAR(v.imag(), im, 1e-12) << "imag at (" << i << "," << j << ")";
        };
        near(0, 0, -1, 1);
        near(0, 1, 1, 2);
        near(1, 0, 3, 3);
        near(1, 1, 6, -3);
      }

      // The #984/#999 promotion discriminator: ComplexFloat (x) Double must
      // produce ComplexDouble output (the higher-precision complex), with the
      // product computed and stored through that output type.
      TEST(GpuKron, MixedComplexFloatDoublePromotesToComplexDouble) {
        Tensor a = zeros({2, 1}, Type.ComplexFloat);
        a.at<cytnx_complex64>({0, 0}) = cytnx_complex64(1.5f, -2.5f);
        a.at<cytnx_complex64>({1, 0}) = cytnx_complex64(0.0f, 4.0f);
        Tensor b = zeros({1, 2}, Type.Double);
        b.at<cytnx_double>({0, 0}) = 2.0;
        b.at<cytnx_double>({0, 1}) = -0.5;

        Tensor out = linalg::Kron(a.to(Device.cuda), b.to(Device.cuda)).to(Device.cpu);

        ASSERT_EQ(out.dtype(), Type.ComplexDouble);
        ASSERT_EQ(out.shape(), (std::vector<cytnx_uint64>{2, 2}));
        // out[i,j] = a[i,0] * b[0,j]
        auto near = [&](cytnx_uint64 i, cytnx_uint64 j, double re, double im) {
          auto v = out.at<cytnx_complex128>({i, j});
          EXPECT_NEAR(v.real(), re, 1e-6) << "real at (" << i << "," << j << ")";
          EXPECT_NEAR(v.imag(), im, 1e-6) << "imag at (" << i << "," << j << ")";
        };
        near(0, 0, 3.0, -5.0);  // (1.5-2.5i)*2
        near(0, 1, -0.75, 1.25);  // (1.5-2.5i)*-0.5
        near(1, 0, 0.0, 8.0);  // (0+4i)*2
        near(1, 1, 0.0, -2.0);  // (0+4i)*-0.5
      }

      // Integer Kron with negative values, hand-computed (independent expected
      // values, not a CPU comparison). Rank-1: element (i*n + j) = a_i * b_j.
      TEST(GpuKron, Int16HandComputed) {
        Tensor a = zeros({2}, Type.Int16);
        a.at<cytnx_int16>({0}) = 3;
        a.at<cytnx_int16>({1}) = -2;
        Tensor b = zeros({2}, Type.Int16);
        b.at<cytnx_int16>({0}) = 4;
        b.at<cytnx_int16>({1}) = 5;

        Tensor out = linalg::Kron(a.to(Device.cuda), b.to(Device.cuda)).to(Device.cpu);
        ASSERT_EQ(out.dtype(), Type.Int16);
        const cytnx_int16 expect[4] = {12, 15, -8, -10};
        for (cytnx_uint64 k = 0; k < 4; ++k) EXPECT_EQ(out.at<cytnx_int16>({k}), expect[k]);
      }

      // Broad cross-check: GPU Kron vs CPU Kron over every real/complex dtype
      // and a few shapes. Secondary to the independent-value tests above.
      TEST(GpuKron, GpuMatchesCpuAllDtypes) {
        const std::vector<std::pair<std::vector<cytnx_uint64>, std::vector<cytnx_uint64>>> shapes =
          {{{2, 3}, {3, 2}}, {{4, 1}, {2, 5}}, {{1, 1}, {3, 3}}};
        for (auto dtype : dtype_list) {
          if (dtype == Type.Bool) continue;  // Kron has no Bool kernel
          // See Outer_test's note: ~1e6-magnitude complex-float products diverge
          // from the CPU at the float32 ULP, so ComplexFloat gets a looser tol.
          const cytnx_double tol = (dtype == Type.ComplexFloat) ? 0.1 : 1e-6;
          for (const auto& s : shapes) {
            SCOPED_TRACE("dtype " + std::to_string(dtype));
            Tensor a(s.first, dtype);
            Tensor b(s.second, dtype);
            InitTensorUniform(a, /*seed=*/1);
            InitTensorUniform(b, /*seed=*/2);
            Tensor expected = linalg::Kron(a, b);
            Tensor gpu = linalg::Kron(a.to(Device.cuda), b.to(Device.cuda)).to(Device.cpu);
            EXPECT_EQ(gpu.dtype(), expected.dtype());
            EXPECT_TRUE(AreNearlyEqTensor(gpu, expected, tol));
          }
        }
      }

    }  // namespace
  }  // namespace gpu_test
}  // namespace cytnx
