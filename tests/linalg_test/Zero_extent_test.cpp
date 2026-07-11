#include "cytnx.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <vector>

namespace cytnx {
  namespace {

    void ExpectEmpty(const Tensor &tensor, const std::vector<cytnx_uint64> &shape,
                     unsigned int dtype) {
      EXPECT_TRUE(tensor.is_empty());
      EXPECT_EQ(tensor.shape(), shape);
      EXPECT_EQ(tensor.dtype(), dtype);
    }

    void ExpectAllEqual(const Tensor &tensor, double expected) {
      Tensor flat = tensor.to(Device.cpu).reshape({static_cast<cytnx_int64>(tensor.size())});
      for (cytnx_uint64 i = 0; i < flat.size(); ++i) {
        EXPECT_DOUBLE_EQ(flat.at<double>({i}), expected);
      }
    }

    TEST(ZeroExtentLinalgTest, UnaryOperationsPreserveShapeAndExpectedDtype) {
      const std::vector<cytnx_uint64> shape{2, 0, 3};

      ExpectEmpty(linalg::Abs(Tensor(shape, Type.ComplexFloat)), shape, Type.Float);
      ExpectEmpty(linalg::Conj(Tensor(shape, Type.ComplexFloat)), shape, Type.ComplexFloat);
      ExpectEmpty(linalg::Exp(Tensor(shape, Type.Float)), shape, Type.Double);
      ExpectEmpty(linalg::Expf(Tensor(shape, Type.Double)), shape, Type.Float);
      ExpectEmpty(linalg::Inv(Tensor(shape, Type.Int64)), shape, Type.Double);
      ExpectEmpty(linalg::Pow(Tensor(shape, Type.Int64), 2.0), shape, Type.Double);

      Tensor value(shape, Type.ComplexFloat);
      linalg::Abs_(value);
      ExpectEmpty(value, shape, Type.Float);

      value = Tensor(shape, Type.Float);
      linalg::Conj_(value);
      ExpectEmpty(value, shape, Type.Float);

      value = Tensor(shape, Type.Float);
      linalg::Exp_(value);
      ExpectEmpty(value, shape, Type.Double);

      value = Tensor(shape, Type.Double);
      linalg::Expf_(value);
      ExpectEmpty(value, shape, Type.Float);

      value = Tensor(shape, Type.Int64);
      linalg::Inv_(value);
      ExpectEmpty(value, shape, Type.Double);

      value = Tensor(shape, Type.Int64);
      linalg::Pow_(value, 2.0);
      ExpectEmpty(value, shape, Type.Double);
    }

    TEST(ZeroExtentLinalgTest, BinaryOperationsDoNoWork) {
      const std::vector<cytnx_uint64> shape{2, 0, 3};
      Tensor lhs(shape, Type.Float);
      Tensor rhs(shape, Type.Double);

      ExpectEmpty(lhs + rhs, shape, Type.Double);
      ExpectEmpty(lhs - rhs, shape, Type.Double);
      ExpectEmpty(lhs * rhs, shape, Type.Double);
      ExpectEmpty(lhs / rhs, shape, Type.Double);
      ExpectEmpty(linalg::Mod(lhs, rhs), shape, Type.Double);
      ExpectEmpty(lhs == rhs, shape, Type.Bool);

      Tensor scalar({}, Type.Double);
      scalar.at<double>({}) = 2.0;
      ExpectEmpty(lhs + scalar, shape, Type.Double);
      ExpectEmpty(linalg::Mod(lhs, scalar), shape, Type.Double);
      ExpectEmpty(lhs == scalar, shape, Type.Bool);
      ExpectEmpty(linalg::Mod(lhs, 2.0), shape, Type.Double);
      ExpectEmpty(linalg::Cpr(lhs, 2.0), shape, Type.Bool);
      EXPECT_THROW(linalg::Mod(Tensor(shape, Type.ComplexDouble), rhs), std::logic_error);

      linalg::iAdd(lhs, rhs);
      EXPECT_TRUE(lhs.is_empty());
      linalg::iSub(lhs, rhs);
      linalg::iMul(lhs, rhs);
      linalg::iDiv(lhs, rhs);
      EXPECT_EQ(lhs.shape(), shape);
    }

    TEST(ZeroExtentLinalgTest, ReductionsUseIdentitiesOrRejectUndefinedExtrema) {
      Tensor real({2, 0, 3}, Type.Double);
      Tensor complex({0}, Type.ComplexFloat);

      EXPECT_DOUBLE_EQ(linalg::Sum(real).at<double>({}), 0.0);
      EXPECT_THROW(linalg::Sum(Tensor({0}, Type.Bool)), std::logic_error);
      EXPECT_FLOAT_EQ(linalg::Norm(complex).at<float>({}), 0.0f);
      EXPECT_DOUBLE_EQ(
        linalg::Vectordot(Tensor({0}, Type.Float), Tensor({0}, Type.Double)).at<double>({}), 0.0);
      EXPECT_THROW(linalg::Min(real), std::logic_error);
      EXPECT_THROW(linalg::Max(real), std::logic_error);

      Tensor traced = linalg::Trace(Tensor({2, 0, 0, 3}, Type.Double), 1, 2);
      EXPECT_EQ(traced.shape(), (std::vector<cytnx_uint64>{2, 3}));
      ExpectAllEqual(traced, 0.0);
    }

    TEST(ZeroExtentLinalgTest, ProductsReturnEmptyOrAdditiveIdentity) {
      Tensor zero_inner = linalg::Matmul(Tensor({2, 0}, Type.Float), Tensor({0, 3}, Type.Double));
      EXPECT_EQ(zero_inner.shape(), (std::vector<cytnx_uint64>{2, 3}));
      EXPECT_EQ(zero_inner.dtype(), Type.Double);
      ExpectAllEqual(zero_inner, 0.0);

      ExpectEmpty(linalg::Matmul(Tensor({0, 2}, Type.Double), Tensor({2, 3}, Type.Double)), {0, 3},
                  Type.Double);
      Tensor dot = linalg::Dot(Tensor({2, 0}, Type.Double), Tensor({0}, Type.Double));
      EXPECT_EQ(dot.shape(), (std::vector<cytnx_uint64>{2}));
      ExpectAllEqual(dot, 0.0);

      ExpectEmpty(linalg::Diag(Tensor({0}, Type.Double)), {0, 0}, Type.Double);
      ExpectEmpty(linalg::Diag(Tensor({0, 0}, Type.Double)), {0}, Type.Double);
      ExpectEmpty(linalg::Outer(Tensor({0}, Type.Double), Tensor({3}, Type.Double)), {0, 3},
                  Type.Double);
      ExpectEmpty(linalg::Ger(Tensor({0}, Type.Double), Tensor({3}, Type.Double)), {0, 3},
                  Type.Double);
      ExpectEmpty(linalg::Kron(Tensor({0, 2}, Type.Double), Tensor({3, 4}, Type.Double)), {0, 8},
                  Type.Double);
      ExpectEmpty(linalg::Matmul_dg(Tensor({0}, Type.Double), Tensor({0, 3}, Type.Double)), {0, 3},
                  Type.Double);

      Tensor contracted =
        linalg::Tensordot(Tensor({2, 0, 3}, Type.Double), Tensor({4, 0, 5}, Type.Double), {1}, {1});
      EXPECT_EQ(contracted.shape(), (std::vector<cytnx_uint64>{2, 3, 4, 5}));
      ExpectAllEqual(contracted, 0.0);

      Tensor scalar =
        linalg::Tensordot(Tensor({0}, Type.Double), Tensor({0}, Type.Double), {0}, {0});
      EXPECT_TRUE(scalar.is_scalar());
      EXPECT_DOUBLE_EQ(scalar.at<double>({}), 0.0);

      Tensor weighted_trace = linalg::Tensordot_dg(
        Tensor({0}, Type.Double), Tensor({0, 0, 2}, Type.Double), {0, 1}, {0, 1}, true);
      EXPECT_EQ(weighted_trace.shape(), (std::vector<cytnx_uint64>{2}));
      ExpectAllEqual(weighted_trace, 0.0);

      ExpectEmpty(linalg::Directsum(Tensor({0, 2}, Type.Double), Tensor({0, 3}, Type.Double), {0}),
                  {0, 5}, Type.Double);
    }

    TEST(ZeroExtentLinalgTest, MatrixFunctionsUseCanonicalEmptyResults) {
      Tensor empty_matrix({0, 0}, Type.Double);
      EXPECT_DOUBLE_EQ(linalg::Det(empty_matrix).at<double>({}), 1.0);
      ExpectEmpty(linalg::InvM(empty_matrix), {0, 0}, Type.Double);
      linalg::InvM_(empty_matrix);
      ExpectEmpty(empty_matrix, {0, 0}, Type.Double);
      ExpectEmpty(linalg::ExpH(empty_matrix), {0, 0}, Type.Double);
      ExpectEmpty(linalg::ExpM(empty_matrix), {0, 0}, Type.ComplexDouble);

      Tensor gemm =
        linalg::Gemm(Scalar(3.0), Tensor({2, 0}, Type.Double), Tensor({0, 3}, Type.Double));
      EXPECT_EQ(gemm.shape(), (std::vector<cytnx_uint64>{2, 3}));
      ExpectAllEqual(gemm, 0.0);

      Tensor c = ones({2, 3}, Type.Double);
      linalg::Gemm_(Scalar(3.0), Tensor({2, 0}, Type.Double), Tensor({0, 3}, Type.Double),
                    Scalar(2.0), c);
      ExpectAllEqual(c, 2.0);

      c.fill(std::numeric_limits<double>::quiet_NaN());
      linalg::Gemm_(Scalar(3.0), Tensor({2, 0}, Type.Double), Tensor({0, 3}, Type.Double),
                    Scalar(0.0), c);
      ExpectAllEqual(c, 0.0);
    }

    TEST(ZeroExtentLinalgTest, BatchedGemmFallsBackForZeroExtents) {
      std::vector<Tensor> a{Tensor({2, 0}, Type.Double), eye(2, Type.Double)};
      std::vector<Tensor> b{Tensor({0, 3}, Type.Double), eye(2, Type.Double)};
      std::vector<Tensor> c{ones({2, 3}, Type.Double), ones({2, 2}, Type.Double)};

      linalg::Gemm_Batch({Scalar(2.0), Scalar(3.0)}, a, b, {Scalar(4.0), Scalar(5.0)}, c, {1, 1});

      ExpectAllEqual(c[0], 4.0);
      EXPECT_DOUBLE_EQ(c[1].at<double>({0, 0}), 8.0);
      EXPECT_DOUBLE_EQ(c[1].at<double>({0, 1}), 5.0);
      EXPECT_DOUBLE_EQ(c[1].at<double>({1, 0}), 5.0);
      EXPECT_DOUBLE_EQ(c[1].at<double>({1, 1}), 8.0);
    }

    TEST(ZeroExtentLinalgTest, ThinFactorizationsHaveZeroAuxiliaryDimension) {
      Tensor matrix({3, 0}, Type.Float);

      std::vector<Tensor> svd = linalg::Svd(matrix);
      ASSERT_EQ(svd.size(), 3);
      ExpectEmpty(svd[0], {0}, Type.Float);
      ExpectEmpty(svd[1], {3, 0}, Type.Float);
      ExpectEmpty(svd[2], {0, 0}, Type.Float);

      std::vector<Tensor> gesvd = linalg::Gesvd(matrix);
      ASSERT_EQ(gesvd.size(), 3);
      ExpectEmpty(gesvd[0], {0}, Type.Float);
      ExpectEmpty(gesvd[1], {3, 0}, Type.Float);
      ExpectEmpty(gesvd[2], {0, 0}, Type.Float);

      std::vector<Tensor> qr = linalg::Qr(matrix, true);
      ASSERT_EQ(qr.size(), 3);
      ExpectEmpty(qr[0], {3, 0}, Type.Float);
      ExpectEmpty(qr[1], {0, 0}, Type.Float);
      ExpectEmpty(qr[2], {0}, Type.Float);

      std::vector<Tensor> qdr = linalg::Qdr(matrix, true);
      ASSERT_EQ(qdr.size(), 4);
      ExpectEmpty(qdr[0], {3, 0}, Type.Float);
      ExpectEmpty(qdr[1], {0}, Type.Float);
      ExpectEmpty(qdr[2], {0, 0}, Type.Float);
      ExpectEmpty(qdr[3], {0}, Type.Float);

      std::vector<Tensor> truncated = linalg::Svd_truncate(matrix, 2, 0.0, true, 2);
      ASSERT_EQ(truncated.size(), 4);
      ExpectEmpty(truncated[3], {0}, Type.Float);

      std::vector<Tensor> gesvd_truncated = linalg::Gesvd_truncate(matrix, 2, 0.0, true, true, 2);
      ASSERT_EQ(gesvd_truncated.size(), 4);
      ExpectEmpty(gesvd_truncated[3], {0}, Type.Float);

      std::vector<Tensor> randomized = linalg::Rsvd(matrix, 2, 0.0, true, true, 2);
      ASSERT_EQ(randomized.size(), 4);
      ExpectEmpty(randomized[0], {0}, Type.Float);
      ExpectEmpty(randomized[3], {0}, Type.Float);

      Tensor wide_matrix({0, 3}, Type.Float);
      svd = linalg::Svd(wide_matrix);
      ExpectEmpty(svd[0], {0}, Type.Float);
      ExpectEmpty(svd[1], {0, 0}, Type.Float);
      ExpectEmpty(svd[2], {0, 3}, Type.Float);

      qr = linalg::Qr(wide_matrix);
      ExpectEmpty(qr[0], {0, 0}, Type.Float);
      ExpectEmpty(qr[1], {0, 3}, Type.Float);
    }

    TEST(ZeroExtentLinalgTest, EmptyEigenproblemsAndTridiagonalProblemReturnEmptyFactors) {
      Tensor matrix({0, 0}, Type.Double);

      std::vector<Tensor> eig = linalg::Eig(matrix);
      ASSERT_EQ(eig.size(), 2);
      ExpectEmpty(eig[0], {0}, Type.ComplexDouble);
      ExpectEmpty(eig[1], {0, 0}, Type.ComplexDouble);

      std::vector<Tensor> eigh = linalg::Eigh(matrix);
      ASSERT_EQ(eigh.size(), 2);
      ExpectEmpty(eigh[0], {0}, Type.Double);
      ExpectEmpty(eigh[1], {0, 0}, Type.Double);

      std::vector<Tensor> tridiag =
        linalg::Tridiag(Tensor({0}, Type.Double), Tensor({0}, Type.Double));
      ASSERT_EQ(tridiag.size(), 2);
      ExpectEmpty(tridiag[0], {0}, Type.Double);
      ExpectEmpty(tridiag[1], {0, 0}, Type.Double);
      EXPECT_THROW(linalg::Tridiag(Tensor({2}, Type.Double), Tensor({0}, Type.Double)),
                   std::logic_error);

      ExpectEmpty(linalg::Rand_isometry(Tensor({3, 0}, Type.Float), 2), {3, 0}, Type.Float);
      EXPECT_THROW(linalg::Lstsq(Tensor({0, 2}, Type.Double), Tensor({0, 1}, Type.Double)),
                   std::logic_error);
    }

  }  // namespace
}  // namespace cytnx
