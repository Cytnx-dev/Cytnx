#include <cstddef>

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

      // See Kron_test's copy: GetRandRange draws Uint16 from the full [0, 65535], and two such
      // values promote to int before multiplying -- 65535 * 65535 overflows INT_MAX, which is
      // undefined behaviour in the GPU kernel and the CPU oracle alike and makes a comparison
      // between them compiler-dependent. Bound Uint16 to the [0, 1000] the wider unsigned dtypes
      // already use. Int16 is safe: 32768 * 32768 fits in int.
      Tensor make_sweep_operand(const std::vector<cytnx_uint64>& shape, unsigned int dtype,
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

      TEST(Outer, GpuDoubleMatchesHandComputed) {
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
        for (std::size_t i = 0; i < 3; i++)
          for (std::size_t j = 0; j < 2; j++)
            EXPECT_DOUBLE_EQ(out.at<cytnx_double>({i, j}), expected[i][j])
              << "at (" << i << "," << j << ")";
      }

      TEST(Outer, GpuComplexDoubleMatchesHandComputed) {
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
        Tensor expected = zeros({2, 2}, Type.ComplexDouble);
        expected.at<cytnx_complex128>({0, 0}) = cytnx_complex128(2, -1);
        expected.at<cytnx_complex128>({0, 1}) = cytnx_complex128(-2, 6);
        expected.at<cytnx_complex128>({1, 0}) = cytnx_complex128(1, 3);
        expected.at<cytnx_complex128>({1, 1}) = cytnx_complex128(-8, -4);
        EXPECT_TRUE(AreNearlyEqTensor(out, expected, 1e-12));
      }

      // The promotion discriminator (mirrors CPU DtypePromotion.OuterComplexfloatDouble):
      // ComplexFloat (x) Double -> ComplexDouble, full double precision.
      //
      // As in Kron_test's counterpart, the Double operands are values float32
      // cannot represent (0.1, 3.3). Computing through ComplexFloat would round
      // them and miss the double product by 1.5e-9 to 9.5e-8 -- far outside the
      // tolerance below. Float-exact operands (3, 0.5) would let a narrowed
      // computation pass unnoticed.
      TEST(Outer, GpuMixedComplexFloatDoublePromotesToComplexDouble) {
        Tensor a = zeros({2}, Type.ComplexFloat);
        a.at<cytnx_complex64>({0}) = cytnx_complex64(1, 1);
        a.at<cytnx_complex64>({1}) = cytnx_complex64(2, 0);
        Tensor b = zeros({2}, Type.Double);
        b.at<cytnx_double>({0}) = 0.1;
        b.at<cytnx_double>({1}) = 3.3;

        Tensor out = linalg::Outer(a.to(Device.cuda), b.to(Device.cuda)).to(Device.cpu);

        ASSERT_EQ(out.dtype(), Type.ComplexDouble);
        ASSERT_EQ(out.shape(), (std::vector<cytnx_uint64>{2, 2}));
        // Evaluated in host double arithmetic, not written as decimal literals.
        Tensor expected = zeros({2, 2}, Type.ComplexDouble);
        expected.at<cytnx_complex128>({0, 0}) = cytnx_complex128(1.0 * 0.1, 1.0 * 0.1);
        expected.at<cytnx_complex128>({0, 1}) = cytnx_complex128(1.0 * 3.3, 1.0 * 3.3);
        expected.at<cytnx_complex128>({1, 0}) = cytnx_complex128(2.0 * 0.1, 0.0);
        expected.at<cytnx_complex128>({1, 1}) = cytnx_complex128(2.0 * 3.3, 0.0);
        EXPECT_TRUE(AreNearlyEqTensor(out, expected, 1e-12));
      }

      // Regression for #1099: Int16 (x) Int16 Outer used to hit a null dispatch
      // row and segfault; it now routes through cuKron. Hand-computed, negative
      // values included.
      TEST(Outer, GpuInt16DiagonalNoLongerSegfaults) {
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

      // Mixed signed/unsigned dtype pair, hand-computed: the sweep below uses a
      // single dtype for both operands. Uint32 (x) Int16 promotes to Int32, so
      // the negative products must survive rather than wrap.
      TEST(Outer, GpuMixedSignedUnsignedPromotion) {
        Tensor a = zeros({2}, Type.Uint32);
        a.at<cytnx_uint32>({0}) = 5;
        a.at<cytnx_uint32>({1}) = 1;
        Tensor b = zeros({2}, Type.Int16);
        b.at<cytnx_int16>({0}) = -2;
        b.at<cytnx_int16>({1}) = 3;

        Tensor out = linalg::Outer(a.to(Device.cuda), b.to(Device.cuda)).to(Device.cpu);

        ASSERT_EQ(out.dtype(), Type.Int32);
        ASSERT_EQ(out.shape(), (std::vector<cytnx_uint64>{2, 2}));
        const cytnx_int32 expect[2][2] = {{-10, 15}, {-2, 3}};
        for (std::size_t i = 0; i < 2; ++i)
          for (std::size_t j = 0; j < 2; ++j)
            EXPECT_EQ(out.at<cytnx_int32>({i, j}), expect[i][j]) << "at (" << i << "," << j << ")";
      }

      // Broad cross-check: GPU Outer vs CPU Outer over every dtype. Deliberately
      // a GPU-vs-CPU consistency check rather than an independent oracle; the
      // hand-computed tests above carry the correctness burden.
      // InitTensorUniform draws from [-1000, 1000], so products reach ~1e6; at
      // that magnitude a complex-float multiply diverges from the CPU one at the
      // float32 ULP (~0.06), hence the looser ComplexFloat tolerance (matches
      // Mul_test's convention). A single real multiply is bit-identical.
      TEST(Outer, GpuMatchesCpuAllDtypes) {
        for (auto dtype : dtype_list) {
          SCOPED_TRACE("dtype " + std::to_string(dtype));
          const double tol = (dtype == Type.ComplexFloat) ? 0.1 : 1e-6;
          Tensor a = make_sweep_operand({5}, dtype, /*seed=*/3);
          Tensor b = make_sweep_operand({7}, dtype, /*seed=*/4);
          Tensor expected = linalg::Outer(a, b);
          Tensor gpu = linalg::Outer(a.to(Device.cuda), b.to(Device.cuda)).to(Device.cpu);
          EXPECT_EQ(gpu.dtype(), expected.dtype());
          EXPECT_TRUE(AreNearlyEqTensor(gpu, expected, tol));
        }
      }

    }  // namespace
  }  // namespace gpu_test
}  // namespace cytnx
