#include <vector>

#include "gtest/gtest.h"

#include "Device.hpp"
#include "Generator.hpp"
#include "Type.hpp"
#include "linalg.hpp"

// Kron_general is the one CPU kernel that multiplies two differently-typed operands elementwise,
// so it is where the #1003 "compute through the operation's output type" rule is observable. The
// kernel now converts both operands to TO -- the promoted dtype the caller already allocated --
// before multiplying, rather than relying on whatever operator C++ finds for the raw operand
// pair. These tests pin the result dtype and hand-computed values; the ComplexDouble x
// ComplexFloat row is the pair std::complex provides no operator for at all.
namespace cytnx {
  namespace test {
    namespace {

      TEST(Kron, MixedComplexPrecisionPromotesToComplexDouble) {
        // l = [1.5 - 2.25i, -0.5 + 1i] (ComplexDouble), r = [0.5 + 0.25i] (ComplexFloat).
        // Every value is exact in binary floating point, so the expectations are independent of
        // the order the kernel converts and multiplies in.
        Tensor l = zeros({2}, Type.ComplexDouble, Device.cpu);
        l.at<cytnx_complex128>({0}) = cytnx_complex128(1.5, -2.25);
        l.at<cytnx_complex128>({1}) = cytnx_complex128(-0.5, 1.0);
        Tensor r = zeros({1}, Type.ComplexFloat, Device.cpu);
        r.at<cytnx_complex64>({0}) = cytnx_complex64(0.5f, 0.25f);

        Tensor out = linalg::Kron(l, r);
        ASSERT_EQ(out.dtype(), Type.ComplexDouble);
        ASSERT_EQ(out.shape(), std::vector<cytnx_uint64>({2}));

        // (1.5 - 2.25i)(0.5 + 0.25i) = (0.75 + 0.5625) + (0.375 - 1.125)i = 1.3125 - 0.75i
        EXPECT_DOUBLE_EQ(out.at<cytnx_complex128>({0}).real(), 1.3125);
        EXPECT_DOUBLE_EQ(out.at<cytnx_complex128>({0}).imag(), -0.75);
        // (-0.5 + 1i)(0.5 + 0.25i) = (-0.25 - 0.25) + (-0.125 + 0.5)i = -0.5 + 0.375i
        EXPECT_DOUBLE_EQ(out.at<cytnx_complex128>({1}).real(), -0.5);
        EXPECT_DOUBLE_EQ(out.at<cytnx_complex128>({1}).imag(), 0.375);
      }

      TEST(Kron, ComplexFloatWithDoublePromotesToComplexDouble) {
        // ComplexFloat x Double crosses the real/complex boundary: type_promote takes the real
        // counterparts first (Float vs Double -> Double) and re-complexifies, so the output is
        // ComplexDouble rather than the input's ComplexFloat (#858, #982).
        Tensor l = zeros({2}, Type.ComplexFloat, Device.cpu);
        l.at<cytnx_complex64>({0}) = cytnx_complex64(1.5f, -0.25f);
        l.at<cytnx_complex64>({1}) = cytnx_complex64(-2.0f, 0.5f);
        Tensor r = zeros({2}, Type.Double, Device.cpu);
        r.at<cytnx_double>({0}) = 0.5;
        r.at<cytnx_double>({1}) = -4.0;

        Tensor out = linalg::Kron(l, r);
        ASSERT_EQ(out.dtype(), Type.ComplexDouble);
        ASSERT_EQ(out.shape(), std::vector<cytnx_uint64>({4}));

        const cytnx_complex128 expect[4] = {
          cytnx_complex128(0.75, -0.125), cytnx_complex128(-6.0, 1.0), cytnx_complex128(-1.0, 0.25),
          cytnx_complex128(8.0, -2.0)};
        for (cytnx_uint64 i = 0; i < 4; ++i) {
          EXPECT_DOUBLE_EQ(out.at<cytnx_complex128>({i}).real(), expect[i].real());
          EXPECT_DOUBLE_EQ(out.at<cytnx_complex128>({i}).imag(), expect[i].imag());
        }
      }

      TEST(Kron, ComplexWithSignedIntegerKeepsSign) {
        // The integer operand is converted to the complex output type, so a negative factor stays
        // negative instead of passing through any unsigned intermediate.
        Tensor l = zeros({2}, Type.ComplexDouble, Device.cpu);
        l.at<cytnx_complex128>({0}) = cytnx_complex128(1.25, -0.5);
        l.at<cytnx_complex128>({1}) = cytnx_complex128(-3.0, 2.0);
        Tensor r = zeros({2}, Type.Int32, Device.cpu);
        r.at<cytnx_int32>({0}) = -2;
        r.at<cytnx_int32>({1}) = 3;

        Tensor out = linalg::Kron(l, r);
        ASSERT_EQ(out.dtype(), Type.ComplexDouble);
        ASSERT_EQ(out.shape(), std::vector<cytnx_uint64>({4}));

        const cytnx_complex128 expect[4] = {
          cytnx_complex128(-2.5, 1.0), cytnx_complex128(3.75, -1.5), cytnx_complex128(6.0, -4.0),
          cytnx_complex128(-9.0, 6.0)};
        for (cytnx_uint64 i = 0; i < 4; ++i) {
          EXPECT_DOUBLE_EQ(out.at<cytnx_complex128>({i}).real(), expect[i].real());
          EXPECT_DOUBLE_EQ(out.at<cytnx_complex128>({i}).imag(), expect[i].imag());
        }
      }

      TEST(Kron, SignedUnsignedMixPromotesToSigned) {
        // Uint32 x Int64 promotes to Int64, and the kernel converts both operands to Int64 before
        // multiplying, so the negative operand survives instead of wrapping through an unsigned
        // intermediate the way C++'s own usual arithmetic conversions would pick.
        Tensor l = zeros({2}, Type.Uint32, Device.cpu);
        l.at<cytnx_uint32>({0}) = 3;
        l.at<cytnx_uint32>({1}) = 7;
        Tensor r = zeros({1}, Type.Int64, Device.cpu);
        r.at<cytnx_int64>({0}) = -5;

        Tensor out = linalg::Kron(l, r);
        ASSERT_EQ(out.dtype(), Type.Int64);
        ASSERT_EQ(out.shape(), std::vector<cytnx_uint64>({2}));
        EXPECT_EQ(out.at<cytnx_int64>({0}), -15);
        EXPECT_EQ(out.at<cytnx_int64>({1}), -35);
      }

      TEST(Kron, RealMatrixShapeAndValues) {
        // The ordinary same-dtype path, as a control: Kron of 2x2 with 2x2 is 4x4 with
        // out[i*2+k, j*2+l] = a[i, j] * b[k, l]. Fractional and negative entries.
        Tensor a = zeros({2, 2}, Type.Double, Device.cpu);
        a.at<cytnx_double>({0, 0}) = 1.5;
        a.at<cytnx_double>({0, 1}) = -2.0;
        a.at<cytnx_double>({1, 0}) = 0.0;
        a.at<cytnx_double>({1, 1}) = 0.25;
        Tensor b = zeros({2, 2}, Type.Double, Device.cpu);
        b.at<cytnx_double>({0, 0}) = 4.0;
        b.at<cytnx_double>({0, 1}) = 0.5;
        b.at<cytnx_double>({1, 0}) = -1.0;
        b.at<cytnx_double>({1, 1}) = 2.0;

        Tensor out = linalg::Kron(a, b);
        ASSERT_EQ(out.dtype(), Type.Double);
        ASSERT_EQ(out.shape(), std::vector<cytnx_uint64>({4, 4}));

        const double expect[4][4] = {{6.0, 0.75, -8.0, -1.0},
                                     {-1.5, 3.0, 2.0, -4.0},
                                     {0.0, 0.0, 1.0, 0.125},
                                     {0.0, 0.0, -0.25, 0.5}};
        for (cytnx_uint64 i = 0; i < 4; ++i)
          for (cytnx_uint64 j = 0; j < 4; ++j)
            EXPECT_DOUBLE_EQ(out.at<cytnx_double>({i, j}), expect[i][j]);
      }

    }  // namespace
  }  // namespace test
}  // namespace cytnx
