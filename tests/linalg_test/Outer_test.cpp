#include <vector>

#include "gtest/gtest.h"

#include "Device.hpp"
#include "Generator.hpp"
#include "linalg.hpp"
#include "Type.hpp"

// Outer(a, b) for rank-1 a (length m) and rank-1 b (length n) is the m x n
// matrix out[i, j] = a_i * b_j. #1003 reimplemented Outer as
// Kron(a, b).reshape({m, n}); these tests pin the observable behaviour with
// independent, hand-computed expected values: output shape, dtype promotion,
// and the elementwise products. They include the Int16 / Uint16 / Bool
// same-dtype pairs whose dispatch rows were missing in the retired per-dtype
// Outer_ii table and segfaulted (#1099) -- Outer casts both operands to the
// promoted dtype first, so only the diagonal Outer_ii[t][t] entries were ever
// reached, and those three were nullptr.

namespace cytnx {
  namespace test {
    namespace {

      TEST(Outer, RealRectangularShapeAndValues) {
        // a = [1, 2, 3], b = [4, 5] -> 3x2 with out[i, j] = a_i * b_j.
        Tensor a = zeros({3}, Type.Double, Device.cpu);
        a.at<cytnx_double>({0}) = 1;
        a.at<cytnx_double>({1}) = 2;
        a.at<cytnx_double>({2}) = 3;
        Tensor b = zeros({2}, Type.Double, Device.cpu);
        b.at<cytnx_double>({0}) = 4;
        b.at<cytnx_double>({1}) = 5;

        Tensor out = linalg::Outer(a, b);
        ASSERT_EQ(out.shape(), std::vector<cytnx_uint64>({3, 2}));
        ASSERT_EQ(out.dtype(), Type.Double);
        const double expect[3][2] = {{4, 5}, {8, 10}, {12, 15}};
        for (cytnx_uint64 i = 0; i < 3; ++i)
          for (cytnx_uint64 j = 0; j < 2; ++j)
            EXPECT_DOUBLE_EQ(out.at<cytnx_double>({i, j}), expect[i][j]);
      }

      TEST(Outer, SingleElementOperandsStayRank1) {
        // A rank-1 length-1 tensor is not a scalar (rank != 0), so Outer keeps
        // the full outer-product shape rather than collapsing.
        Tensor a = zeros({1}, Type.Double, Device.cpu);
        a.at<cytnx_double>({0}) = 2;
        Tensor b = zeros({2}, Type.Double, Device.cpu);
        b.at<cytnx_double>({0}) = 3;
        b.at<cytnx_double>({1}) = 4;

        Tensor out = linalg::Outer(a, b);
        ASSERT_EQ(out.shape(), std::vector<cytnx_uint64>({1, 2}));
        EXPECT_DOUBLE_EQ(out.at<cytnx_double>({0, 0}), 6);
        EXPECT_DOUBLE_EQ(out.at<cytnx_double>({0, 1}), 8);

        Tensor out_t = linalg::Outer(b, a);
        ASSERT_EQ(out_t.shape(), std::vector<cytnx_uint64>({2, 1}));
        EXPECT_DOUBLE_EQ(out_t.at<cytnx_double>({0, 0}), 6);
        EXPECT_DOUBLE_EQ(out_t.at<cytnx_double>({1, 0}), 8);
      }

      TEST(Outer, ComplexFloatDoublePromotesToComplexDouble) {
        // The promoted dtype differs from both inputs; the wider complex output
        // must hold full double precision.
        Tensor a = zeros({2}, Type.ComplexFloat, Device.cpu);
        a.at<cytnx_complex64>({0}) = cytnx_complex64(1, 1);
        a.at<cytnx_complex64>({1}) = cytnx_complex64(2, 0);
        Tensor b = zeros({2}, Type.Double, Device.cpu);
        b.at<cytnx_double>({0}) = 3;
        b.at<cytnx_double>({1}) = 0.5;

        Tensor out = linalg::Outer(a, b);
        ASSERT_EQ(out.dtype(), Type.ComplexDouble);
        const cytnx_complex128 v00 = out.at<cytnx_complex128>({0, 0});  // (1+1i) * 3
        EXPECT_DOUBLE_EQ(v00.real(), 3);
        EXPECT_DOUBLE_EQ(v00.imag(), 3);
        const cytnx_complex128 v11 = out.at<cytnx_complex128>({1, 1});  // (2+0i) * 0.5
        EXPECT_DOUBLE_EQ(v11.real(), 1);
        EXPECT_DOUBLE_EQ(v11.imag(), 0);
      }

      TEST(Outer, Int16DiagonalNoLongerSegfaults) {  // #1099
        Tensor a = zeros({2}, Type.Int16, Device.cpu);
        a.at<cytnx_int16>({0}) = 3;
        a.at<cytnx_int16>({1}) = -2;
        Tensor b = zeros({2}, Type.Int16, Device.cpu);
        b.at<cytnx_int16>({0}) = 4;
        b.at<cytnx_int16>({1}) = 5;

        Tensor out = linalg::Outer(a, b);
        ASSERT_EQ(out.dtype(), Type.Int16);
        EXPECT_EQ(out.at<cytnx_int16>({0, 0}), 12);
        EXPECT_EQ(out.at<cytnx_int16>({0, 1}), 15);
        EXPECT_EQ(out.at<cytnx_int16>({1, 0}), -8);
        EXPECT_EQ(out.at<cytnx_int16>({1, 1}), -10);
      }

      TEST(Outer, Uint16DiagonalNoLongerSegfaults) {  // #1099
        Tensor a = zeros({2}, Type.Uint16, Device.cpu);
        a.at<cytnx_uint16>({0}) = 3;
        a.at<cytnx_uint16>({1}) = 7;
        Tensor b = zeros({1}, Type.Uint16, Device.cpu);
        b.at<cytnx_uint16>({0}) = 6;

        Tensor out = linalg::Outer(a, b);
        ASSERT_EQ(out.shape(), std::vector<cytnx_uint64>({2, 1}));
        ASSERT_EQ(out.dtype(), Type.Uint16);
        EXPECT_EQ(out.at<cytnx_uint16>({0, 0}), 18);
        EXPECT_EQ(out.at<cytnx_uint16>({1, 0}), 42);
      }

      TEST(Outer, BoolDiagonalGivesElementwiseAnd) {  // #1099
        // For Bool operands the product is logical AND.
        Tensor a = zeros({2}, Type.Bool, Device.cpu);
        a.at<cytnx_bool>({0}) = true;
        a.at<cytnx_bool>({1}) = false;
        Tensor b = zeros({2}, Type.Bool, Device.cpu);
        b.at<cytnx_bool>({0}) = true;
        b.at<cytnx_bool>({1}) = true;

        Tensor out = linalg::Outer(a, b);
        ASSERT_EQ(out.dtype(), Type.Bool);
        EXPECT_TRUE(out.at<cytnx_bool>({0, 0}));
        EXPECT_TRUE(out.at<cytnx_bool>({0, 1}));
        EXPECT_FALSE(out.at<cytnx_bool>({1, 0}));
        EXPECT_FALSE(out.at<cytnx_bool>({1, 1}));
      }

    }  // namespace
  }  // namespace test
}  // namespace cytnx
