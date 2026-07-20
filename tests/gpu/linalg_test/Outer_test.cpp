#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "gpu_test_tools.h"

// Outer on the GPU now routes through cuKron (reshaped) after #1003 retired the
// per-dtype cuOuter_ii table. These tests move CPU-built rank-1 inputs to the
// GPU, run Outer there, copy the result back, and check it against independent
// hand-computed values -- including an Int16 diagonal case (one of the dispatch
// rows missing from the retired table, #1099) and a mixed-precision promotion.

namespace cytnx {
  namespace gpu_test {
    namespace {

      TEST(Outer, GpuRealRectangularShapeAndValues) {
        Tensor a = zeros({3}, Type.Double);
        a.at<cytnx_double>({0}) = 1;
        a.at<cytnx_double>({1}) = 2;
        a.at<cytnx_double>({2}) = 3;
        Tensor b = zeros({2}, Type.Double);
        b.at<cytnx_double>({0}) = 4;
        b.at<cytnx_double>({1}) = 5;

        Tensor out = linalg::Outer(a.to(Device.cuda), b.to(Device.cuda)).to(Device.cpu);
        ASSERT_EQ(out.shape(), std::vector<cytnx_uint64>({3, 2}));
        ASSERT_EQ(out.dtype(), Type.Double);
        const double expect[3][2] = {{4, 5}, {8, 10}, {12, 15}};
        for (cytnx_uint64 i = 0; i < 3; ++i)
          for (cytnx_uint64 j = 0; j < 2; ++j)
            EXPECT_DOUBLE_EQ(out.at<cytnx_double>({i, j}), expect[i][j]);
      }

      TEST(Outer, GpuInt16DiagonalNoLongerSegfaults) {  // #1099
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

      TEST(Outer, GpuComplexFloatDoublePromotesToComplexDouble) {
        Tensor a = zeros({2}, Type.ComplexFloat);
        a.at<cytnx_complex64>({0}) = cytnx_complex64(1, 1);
        a.at<cytnx_complex64>({1}) = cytnx_complex64(2, 0);
        Tensor b = zeros({2}, Type.Double);
        b.at<cytnx_double>({0}) = 3;
        b.at<cytnx_double>({1}) = 0.5;

        Tensor out = linalg::Outer(a.to(Device.cuda), b.to(Device.cuda)).to(Device.cpu);
        ASSERT_EQ(out.dtype(), Type.ComplexDouble);
        const cytnx_complex128 v00 = out.at<cytnx_complex128>({0, 0});  // (1+1i) * 3
        EXPECT_DOUBLE_EQ(v00.real(), 3);
        EXPECT_DOUBLE_EQ(v00.imag(), 3);
        const cytnx_complex128 v11 = out.at<cytnx_complex128>({1, 1});  // (2+0i) * 0.5
        EXPECT_DOUBLE_EQ(v11.real(), 1);
        EXPECT_DOUBLE_EQ(v11.imag(), 0);
      }

    }  // namespace
  }  // namespace gpu_test
}  // namespace cytnx
