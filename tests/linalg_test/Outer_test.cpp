#include <vector>

#include "gtest/gtest.h"

#include "Device.hpp"
#include "Generator.hpp"
#include "linalg.hpp"
#include "Type.hpp"

// Regression coverage for linalg::Outer over the small-integer and Bool
// dtypes. The CPU Outer_ii dispatch table was missing every Int16, Uint16, and
// Bool *source* row, so linalg::Outer with such a left operand dereferenced a
// null function pointer and segfaulted. These tests exercise exactly those
// dtypes (out_dtype == Int16/Uint16/Bool) and check independently hand-computed
// values, so they crash on the pre-fix binary and pass once the rows are
// registered. Outer(a, b) gives out[i, j] = a[i] * b[j].
namespace cytnx {
  namespace test {
    namespace {

      // Int16 x Int16 -> Int16, with negative values. Products stay within
      // int16 range so the expected values are exact.
      TEST(OuterDtypeCoverage, Int16) {
        Tensor a = zeros({3}, Type.Int16, Device.cpu);
        a.at<cytnx_int16>({0}) = 2;
        a.at<cytnx_int16>({1}) = -3;
        a.at<cytnx_int16>({2}) = 4;
        Tensor b = zeros({2}, Type.Int16, Device.cpu);
        b.at<cytnx_int16>({0}) = 5;
        b.at<cytnx_int16>({1}) = -6;

        Tensor out = linalg::Outer(a, b);

        ASSERT_EQ(out.dtype(), Type.Int16);
        ASSERT_EQ(out.shape(), (std::vector<cytnx_uint64>{3, 2}));
        const cytnx_int16 expected[3][2] = {{10, -12}, {-15, 18}, {20, -24}};
        for (cytnx_uint64 i = 0; i < 3; i++)
          for (cytnx_uint64 j = 0; j < 2; j++)
            EXPECT_EQ(out.at<cytnx_int16>({i, j}), expected[i][j])
              << "at (" << i << "," << j << ")";
      }

      // Uint16 x Uint16 -> Uint16.
      TEST(OuterDtypeCoverage, Uint16) {
        Tensor a = zeros({2}, Type.Uint16, Device.cpu);
        a.at<cytnx_uint16>({0}) = 2;
        a.at<cytnx_uint16>({1}) = 3;
        Tensor b = zeros({3}, Type.Uint16, Device.cpu);
        b.at<cytnx_uint16>({0}) = 4;
        b.at<cytnx_uint16>({1}) = 5;
        b.at<cytnx_uint16>({2}) = 6;

        Tensor out = linalg::Outer(a, b);

        ASSERT_EQ(out.dtype(), Type.Uint16);
        ASSERT_EQ(out.shape(), (std::vector<cytnx_uint64>{2, 3}));
        const cytnx_uint16 expected[2][3] = {{8, 10, 12}, {12, 15, 18}};
        for (cytnx_uint64 i = 0; i < 2; i++)
          for (cytnx_uint64 j = 0; j < 3; j++)
            EXPECT_EQ(out.at<cytnx_uint16>({i, j}), expected[i][j])
              << "at (" << i << "," << j << ")";
      }

      // Bool x Bool -> Bool. The outer product of two boolean vectors is the
      // logical AND (product) of their entries.
      TEST(OuterDtypeCoverage, Bool) {
        Tensor a = zeros({2}, Type.Bool, Device.cpu);
        a.at<cytnx_bool>({0}) = true;
        a.at<cytnx_bool>({1}) = false;
        Tensor b = zeros({3}, Type.Bool, Device.cpu);
        b.at<cytnx_bool>({0}) = true;
        b.at<cytnx_bool>({1}) = true;
        b.at<cytnx_bool>({2}) = false;

        Tensor out = linalg::Outer(a, b);

        ASSERT_EQ(out.dtype(), Type.Bool);
        ASSERT_EQ(out.shape(), (std::vector<cytnx_uint64>{2, 3}));
        const bool expected[2][3] = {{true, true, false}, {false, false, false}};
        for (cytnx_uint64 i = 0; i < 2; i++)
          for (cytnx_uint64 j = 0; j < 3; j++)
            EXPECT_EQ(out.at<cytnx_bool>({i, j}), expected[i][j]) << "at (" << i << "," << j << ")";
      }

      // Int16 x Bool -> Int16 (Bool promotes to Int16): exercises the Int16
      // source row against a different second dtype.
      TEST(OuterDtypeCoverage, Int16TimesBool) {
        Tensor a = zeros({2}, Type.Int16, Device.cpu);
        a.at<cytnx_int16>({0}) = 7;
        a.at<cytnx_int16>({1}) = -9;
        Tensor b = zeros({2}, Type.Bool, Device.cpu);
        b.at<cytnx_bool>({0}) = true;
        b.at<cytnx_bool>({1}) = false;

        Tensor out = linalg::Outer(a, b);

        ASSERT_EQ(out.dtype(), Type.Int16);
        ASSERT_EQ(out.shape(), (std::vector<cytnx_uint64>{2, 2}));
        const cytnx_int16 expected[2][2] = {{7, 0}, {-9, 0}};
        for (cytnx_uint64 i = 0; i < 2; i++)
          for (cytnx_uint64 j = 0; j < 2; j++)
            EXPECT_EQ(out.at<cytnx_int16>({i, j}), expected[i][j])
              << "at (" << i << "," << j << ")";
      }

    }  // namespace
  }  // namespace test
}  // namespace cytnx
