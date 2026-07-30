#include <cstddef>

#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "gpu_test_tools.h"

// GPU Kron correctness. Every expected value here is computed independently
// from the Kronecker-product definition
//
//     out[i, j] = A[i / b_rows, j / b_cols] * B[i % b_rows, j % b_cols]
//
// (A is a_rows x a_cols, B is b_rows x b_cols, out is (a_rows*b_rows) x
// (a_cols*b_cols)), written out as
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

      // Builds a sweep operand, bounding Uint16 away from signed-overflow UB.
      //
      // GetRandRange gives the narrow types their full numeric_limits range, so Uint16 draws
      // from [0, 65535] -- unlike Uint32/Uint64, which use [0, 1000]. Two uint16_t operands
      // undergo integral promotion to int before multiplying, and 65535 * 65535 is 4.29e9,
      // past INT_MAX. That is undefined behaviour in the CPU oracle and the GPU kernel alike,
      // before the product is narrowed back to uint16, which makes any comparison between them
      // compiler-dependent rather than a real check. Int16 is safe: 32768 * 32768 fits in int.
      //
      // Bounding here rather than in GetRandRange keeps the change to this PR's tests; the
      // promotion hazard in the kernels themselves is a separate matter.
      Tensor MakeSweepOperand(const std::vector<cytnx_uint64>& shape, unsigned int dtype,
                              unsigned int seed) {
        if (dtype == Type.Uint16) {
          Tensor bounded(shape, Type.Double);
          random::uniform_(bounded, 0, 1000, seed);
          return bounded.astype(Type.Uint16);
        }
        Tensor t(shape, dtype);
        InitTensorUniform(t, seed);
        return t;
      }

      // 2x2 Kron 2x2 -> 4x4, real. A = [[1,2],[3,4]], B = [[0,5],[6,7]].
      // Kron(A,B) laid out by hand:
      //   [ 0  5   0 10 ]
      //   [ 6  7  12 14 ]
      //   [ 0 15   0 20 ]
      //   [18 21  24 28 ]
      TEST(Kron, GpuDoubleMatchesHandComputed) {
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
        for (std::size_t i = 0; i < 4; i++)
          for (std::size_t j = 0; j < 4; j++)
            EXPECT_DOUBLE_EQ(out.at<cytnx_double>({i, j}), expected[i][j])
              << "at (" << i << "," << j << ")";
      }

      // Fractional and negative values, recomputed per-element from the raw
      // inputs (independent of Kron's own layout).
      TEST(Kron, GpuDoubleFractionalNegative) {
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
        const std::size_t b_rows = 2, b_cols = 3;
        for (std::size_t i = 0; i < 6; i++)
          for (std::size_t j = 0; j < 6; j++) {
            const double expected = a.at<cytnx_double>({i / b_rows, j / b_cols}) *
                                    b.at<cytnx_double>({i % b_rows, j % b_cols});
            EXPECT_DOUBLE_EQ(out.at<cytnx_double>({i, j}), expected)
              << "at (" << i << "," << j << ")";
          }
      }

      // Complex x complex, hand-computed. A = [[1+1i, 2-1i]], B = [[0+1i],[3+0i]].
      TEST(Kron, GpuComplexDoubleMatchesHandComputed) {
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
        Tensor expected = zeros({2, 2}, Type.ComplexDouble);
        expected.at<cytnx_complex128>({0, 0}) = cytnx_complex128(-1, 1);
        expected.at<cytnx_complex128>({0, 1}) = cytnx_complex128(1, 2);
        expected.at<cytnx_complex128>({1, 0}) = cytnx_complex128(3, 3);
        expected.at<cytnx_complex128>({1, 1}) = cytnx_complex128(6, -3);
        EXPECT_TRUE(AreNearlyEqTensor(out, expected, 1e-12));
      }

      // The #984/#999 promotion discriminator: ComplexFloat (x) Double must
      // produce ComplexDouble output (the higher-precision complex), with the
      // product computed and stored through that output type.
      //
      // The Double operands are deliberately values float32 cannot represent
      // (0.1, 3.3). An implementation that reports ComplexDouble but multiplies
      // through ComplexFloat would round them first and land 5.9e-9 (x0.1) to
      // 1.9e-7 (x3.3) away from the double product -- 6e3 to 2e5 times the
      // tolerance below, so it fails loudly. With float-exact operands (2.0,
      // -0.5) both paths give bit-identical answers and this test would pass on
      // a narrowed computation.
      //
      // The ComplexFloat side keeps float-exact components (1.5, -2.5, 0, 4) so
      // the only precision question under test is the one above; float -> double
      // widening of the operand is exact.
      TEST(Kron, GpuMixedComplexFloatDoublePromotesToComplexDouble) {
        Tensor a = zeros({2, 1}, Type.ComplexFloat);
        a.at<cytnx_complex64>({0, 0}) = cytnx_complex64(1.5f, -2.5f);
        a.at<cytnx_complex64>({1, 0}) = cytnx_complex64(0.0f, 4.0f);
        Tensor b = zeros({1, 2}, Type.Double);
        b.at<cytnx_double>({0, 0}) = 0.1;
        b.at<cytnx_double>({0, 1}) = 3.3;

        Tensor out = linalg::Kron(a.to(Device.cuda), b.to(Device.cuda)).to(Device.cpu);

        ASSERT_EQ(out.dtype(), Type.ComplexDouble);
        ASSERT_EQ(out.shape(), (std::vector<cytnx_uint64>{2, 2}));
        // out[i,j] = a[i,0] * b[0,j], evaluated here in host double arithmetic
        // rather than written as decimal literals: 1.5 * 3.3 is 4.949999999999999,
        // not 4.95, and the point of the test is the exact double product.
        Tensor expected = zeros({2, 2}, Type.ComplexDouble);
        expected.at<cytnx_complex128>({0, 0}) = cytnx_complex128(1.5 * 0.1, -2.5 * 0.1);
        expected.at<cytnx_complex128>({0, 1}) = cytnx_complex128(1.5 * 3.3, -2.5 * 3.3);
        expected.at<cytnx_complex128>({1, 0}) = cytnx_complex128(0.0, 4.0 * 0.1);
        expected.at<cytnx_complex128>({1, 1}) = cytnx_complex128(0.0, 4.0 * 3.3);
        EXPECT_TRUE(AreNearlyEqTensor(out, expected, 1e-12));
      }

      // Integer Kron with negative values, hand-computed (independent expected
      // values, not a CPU comparison). Rank-1: element (i*n + j) = a_i * b_j.
      TEST(Kron, GpuInt16HandComputed) {
        Tensor a = zeros({2}, Type.Int16);
        a.at<cytnx_int16>({0}) = 3;
        a.at<cytnx_int16>({1}) = -2;
        Tensor b = zeros({2}, Type.Int16);
        b.at<cytnx_int16>({0}) = 4;
        b.at<cytnx_int16>({1}) = 5;

        Tensor out = linalg::Kron(a.to(Device.cuda), b.to(Device.cuda)).to(Device.cpu);
        ASSERT_EQ(out.dtype(), Type.Int16);
        const cytnx_int16 expect[4] = {12, 15, -8, -10};
        for (std::size_t k = 0; k < 4; ++k) EXPECT_EQ(out.at<cytnx_int16>({k}), expect[k]);
      }

      // Kron short-circuits on a rank-0 operand -- `if (lhs.is_scalar() || rhs.is_scalar())
      // return lhs * rhs` (Kron.cpp) -- which is a scalar broadcast, not the Kronecker kernel.
      // Shape {1, 1} does not reach it: that is still rank 2. Both operand orders are covered,
      // and a mixed-dtype pair, because promotion on this path goes through Mul rather than
      // through Kron's own output-type selection.
      TEST(Kron, GpuRankZeroOperandTakesScalarPath) {
        Tensor s = zeros({}, Type.Double);
        s.item<cytnx_double>() = -1.5;
        ASSERT_TRUE(s.is_scalar());
        ASSERT_EQ(s.shape().size(), 0u);

        Tensor v = zeros({3}, Type.Double);
        v.at<cytnx_double>({0}) = 2.0;
        v.at<cytnx_double>({1}) = -4.0;
        v.at<cytnx_double>({2}) = 0.5;
        const double expect[3] = {-3.0, 6.0, -0.75};  // -1.5 * each element

        {
          SCOPED_TRACE("rank-0 on the left");
          Tensor out = linalg::Kron(s.to(Device.cuda), v.to(Device.cuda)).to(Device.cpu);
          ASSERT_EQ(out.dtype(), Type.Double);
          ASSERT_EQ(out.shape(), (std::vector<cytnx_uint64>{3}));
          for (std::size_t k = 0; k < 3; ++k)
            EXPECT_DOUBLE_EQ(out.at<cytnx_double>({k}), expect[k]) << "element " << k;
        }
        {
          SCOPED_TRACE("rank-0 on the right");
          Tensor out = linalg::Kron(v.to(Device.cuda), s.to(Device.cuda)).to(Device.cpu);
          ASSERT_EQ(out.dtype(), Type.Double);
          ASSERT_EQ(out.shape(), (std::vector<cytnx_uint64>{3}));
          for (std::size_t k = 0; k < 3; ++k)
            EXPECT_DOUBLE_EQ(out.at<cytnx_double>({k}), expect[k]) << "element " << k;
        }
        {
          // Same float32-inexact operands as the kernel-path discriminator above, so a narrowed
          // scalar multiply is caught here too rather than only in the Kronecker path.
          SCOPED_TRACE("ComplexFloat rank-0 x Double vector -> ComplexDouble");
          Tensor cs = zeros({}, Type.ComplexFloat);
          cs.item<cytnx_complex64>() = cytnx_complex64(1.5f, -2.5f);
          Tensor d = zeros({2}, Type.Double);
          d.at<cytnx_double>({0}) = 0.1;
          d.at<cytnx_double>({1}) = 3.3;

          Tensor out = linalg::Kron(cs.to(Device.cuda), d.to(Device.cuda)).to(Device.cpu);
          ASSERT_EQ(out.dtype(), Type.ComplexDouble);
          ASSERT_EQ(out.shape(), (std::vector<cytnx_uint64>{2}));
          Tensor expected = zeros({2}, Type.ComplexDouble);
          expected.at<cytnx_complex128>({0}) = cytnx_complex128(1.5 * 0.1, -2.5 * 0.1);
          expected.at<cytnx_complex128>({1}) = cytnx_complex128(1.5 * 3.3, -2.5 * 3.3);
          EXPECT_TRUE(AreNearlyEqTensor(out, expected, 1e-12));
        }
      }

      // Mixed signed/unsigned dtype pairs, hand-computed. The broad sweep below
      // builds both operands from a single dtype, so every off-diagonal
      // promotion row would otherwise be untested. Type_class::type_promote
      // steps an unsigned operand up to the adjacent signed type when the other
      // side is signed and higher precision -- so Uint32 (x) Int16 is Int32, not
      // Uint32, and a negative product has to survive rather than wrap.
      TEST(Kron, GpuMixedSignedUnsignedPromotion) {
        {
          SCOPED_TRACE("Int32 (x) Uint32 -> Int32");
          Tensor a = zeros({2}, Type.Int32);
          a.at<cytnx_int32>({0}) = -3;
          a.at<cytnx_int32>({1}) = 7;
          Tensor b = zeros({2}, Type.Uint32);
          b.at<cytnx_uint32>({0}) = 4;
          b.at<cytnx_uint32>({1}) = 2;

          Tensor out = linalg::Kron(a.to(Device.cuda), b.to(Device.cuda)).to(Device.cpu);
          ASSERT_EQ(out.dtype(), Type.Int32);
          const cytnx_int32 expect[4] = {-12, -6, 28, 14};
          for (std::size_t k = 0; k < 4; ++k)
            EXPECT_EQ(out.at<cytnx_int32>({k}), expect[k]) << "element " << k;
        }
        {
          SCOPED_TRACE("Uint32 (x) Int16 -> Int32");
          Tensor a = zeros({2}, Type.Uint32);
          a.at<cytnx_uint32>({0}) = 5;
          a.at<cytnx_uint32>({1}) = 1;
          Tensor b = zeros({2}, Type.Int16);
          b.at<cytnx_int16>({0}) = -2;
          b.at<cytnx_int16>({1}) = 3;

          Tensor out = linalg::Kron(a.to(Device.cuda), b.to(Device.cuda)).to(Device.cpu);
          ASSERT_EQ(out.dtype(), Type.Int32);
          const cytnx_int32 expect[4] = {-10, 15, -2, 3};
          for (std::size_t k = 0; k < 4; ++k)
            EXPECT_EQ(out.at<cytnx_int32>({k}), expect[k]) << "element " << k;
        }
      }

      // Broad cross-check: GPU Kron vs CPU Kron over every dtype and a few
      // shapes. Deliberately a GPU-vs-CPU consistency check, not an independent
      // oracle -- the hand-computed tests above carry the correctness burden;
      // this one catches GPU-only regressions across the dtype matrix.
      TEST(Kron, GpuMatchesCpuAllDtypes) {
        const std::vector<std::pair<std::vector<cytnx_uint64>, std::vector<cytnx_uint64>>> shapes =
          {{{2, 3}, {3, 2}}, {{4, 1}, {2, 5}}, {{1, 1}, {3, 3}}};
        for (auto dtype : dtype_list) {
          // See Outer_test's note: ~1e6-magnitude complex-float products diverge
          // from the CPU at the float32 ULP, so ComplexFloat gets a looser tol.
          const cytnx_double tol = (dtype == Type.ComplexFloat) ? 0.1 : 1e-6;
          for (const auto& s : shapes) {
            SCOPED_TRACE("dtype " + std::to_string(dtype));
            Tensor a = MakeSweepOperand(s.first, dtype, /*seed=*/1);
            Tensor b = MakeSweepOperand(s.second, dtype, /*seed=*/2);
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
