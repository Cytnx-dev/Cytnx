#include "cytnx.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <vector>

namespace cytnx {
  namespace {

    void ExpectGpuEmpty(const Tensor &tensor, const std::vector<cytnx_uint64> &shape,
                        unsigned int dtype) {
      EXPECT_TRUE(tensor.is_empty());
      EXPECT_EQ(tensor.shape(), shape);
      EXPECT_EQ(tensor.dtype(), dtype);
      EXPECT_EQ(tensor.device(), Device.cuda);
    }

    void ExpectGpuAllEqual(const Tensor &tensor, double expected) {
      Tensor flat = tensor.to(Device.cpu).reshape({static_cast<cytnx_int64>(tensor.size())});
      for (cytnx_uint64 i = 0; i < flat.size(); ++i) {
        EXPECT_DOUBLE_EQ(flat.at<double>({i}), expected);
      }
    }

    TEST(ZeroExtentGpuLinalgTest, ElementwiseAndReductionsDoNotLaunchZeroBlockKernels) {
      const std::vector<cytnx_uint64> shape{2, 0, 3};
      Tensor lhs(shape, Type.Float, Device.cuda);
      Tensor rhs(shape, Type.Double, Device.cuda);

      ExpectGpuEmpty(lhs.astype(Type.Double), shape, Type.Double);
      ExpectGpuEmpty(linalg::Abs(Tensor(shape, Type.ComplexFloat, Device.cuda)), shape, Type.Float);
      ExpectGpuEmpty(linalg::Conj(Tensor(shape, Type.ComplexFloat, Device.cuda)), shape,
                     Type.ComplexFloat);
      ExpectGpuEmpty(linalg::Exp(lhs), shape, Type.Double);
      ExpectGpuEmpty(linalg::Inv(Tensor(shape, Type.Int64, Device.cuda)), shape, Type.Double);
      ExpectGpuEmpty(linalg::Pow(Tensor(shape, Type.Int64, Device.cuda), 2.0), shape, Type.Double);
      ExpectGpuEmpty(linalg::Mod(lhs, rhs), shape, Type.Double);
      ExpectGpuEmpty(lhs == rhs, shape, Type.Bool);
      ExpectGpuEmpty(linalg::Mod(lhs, 2.0), shape, Type.Double);
      ExpectGpuEmpty(linalg::Cpr(lhs, 2.0), shape, Type.Bool);
      EXPECT_THROW(linalg::Mod(Tensor(shape, Type.ComplexDouble, Device.cuda), rhs),
                   std::logic_error);

      EXPECT_DOUBLE_EQ(linalg::Sum(rhs).to(Device.cpu).at<double>({}), 0.0);
      EXPECT_THROW(linalg::Sum(Tensor(shape, Type.Bool, Device.cuda)), std::logic_error);
      EXPECT_DOUBLE_EQ(linalg::Norm(rhs).to(Device.cpu).at<double>({}), 0.0);
      EXPECT_THROW(linalg::Min(rhs), std::logic_error);
      EXPECT_THROW(linalg::Max(rhs), std::logic_error);
    }

    TEST(ZeroExtentGpuLinalgTest, ProductsUseEmptyAndZeroInnerDimensionSemantics) {
      Tensor product = linalg::Matmul(Tensor({2, 0}, Type.Float, Device.cuda),
                                      Tensor({0, 3}, Type.Double, Device.cuda));
      EXPECT_EQ(product.shape(), (std::vector<cytnx_uint64>{2, 3}));
      EXPECT_EQ(product.dtype(), Type.Double);
      ExpectGpuAllEqual(product, 0.0);

      Tensor contracted = linalg::Tensordot(Tensor({2, 0, 3}, Type.Double, Device.cuda),
                                            Tensor({4, 0, 5}, Type.Double, Device.cuda), {1}, {1});
      EXPECT_EQ(contracted.shape(), (std::vector<cytnx_uint64>{2, 3, 4, 5}));
      ExpectGpuAllEqual(contracted, 0.0);

      Tensor c = ones({2, 3}, Type.Double, Device.cuda);
      linalg::Gemm_(Scalar(3.0), Tensor({2, 0}, Type.Double, Device.cuda),
                    Tensor({0, 3}, Type.Double, Device.cuda), Scalar(2.0), c);
      ExpectGpuAllEqual(c, 2.0);

      c.fill(std::numeric_limits<double>::quiet_NaN());
      linalg::Gemm_(Scalar(3.0), Tensor({2, 0}, Type.Double, Device.cuda),
                    Tensor({0, 3}, Type.Double, Device.cuda), Scalar(0.0), c);
      ExpectGpuAllEqual(c, 0.0);

      EXPECT_DOUBLE_EQ(linalg::Det(Tensor({0, 0}, Type.Double, Device.cuda)).at<double>({}), 1.0);
    }

    TEST(ZeroExtentGpuLinalgTest, ThinFactorizationsBypassGpuLibraries) {
      Tensor matrix({3, 0}, Type.Float, Device.cuda);

      std::vector<Tensor> svd = linalg::Svd(matrix);
      ASSERT_EQ(svd.size(), 3);
      ExpectGpuEmpty(svd[0], {0}, Type.Float);
      ExpectGpuEmpty(svd[1], {3, 0}, Type.Float);
      ExpectGpuEmpty(svd[2], {0, 0}, Type.Float);

      std::vector<Tensor> qr = linalg::Qr(matrix, true);
      ASSERT_EQ(qr.size(), 3);
      ExpectGpuEmpty(qr[0], {3, 0}, Type.Float);
      ExpectGpuEmpty(qr[1], {0, 0}, Type.Float);
      ExpectGpuEmpty(qr[2], {0}, Type.Float);

      std::vector<Tensor> eig = linalg::Eigh(Tensor({0, 0}, Type.Double, Device.cuda));
      ASSERT_EQ(eig.size(), 2);
      ExpectGpuEmpty(eig[0], {0}, Type.Double);
      ExpectGpuEmpty(eig[1], {0, 0}, Type.Double);

      std::vector<Tensor> tridiag = linalg::Tridiag(Tensor({0}, Type.Double, Device.cuda),
                                                    Tensor({0}, Type.Double, Device.cuda));
      ASSERT_EQ(tridiag.size(), 2);
      ExpectGpuEmpty(tridiag[0], {0}, Type.Double);
      ExpectGpuEmpty(tridiag[1], {0, 0}, Type.Double);
    }

    TEST(ZeroExtentGpuLinalgTest, IntegerPowReadsThePromotedBuffer) {
      Tensor input = arange(1, 4, 1, Type.Int32, Device.cuda);
      Tensor result = linalg::Pow(input, 2.0).to(Device.cpu);

      ASSERT_EQ(result.dtype(), Type.Double);
      EXPECT_DOUBLE_EQ(result.at<double>({0}), 1.0);
      EXPECT_DOUBLE_EQ(result.at<double>({1}), 4.0);
      EXPECT_DOUBLE_EQ(result.at<double>({2}), 9.0);
    }

  }  // namespace
}  // namespace cytnx
