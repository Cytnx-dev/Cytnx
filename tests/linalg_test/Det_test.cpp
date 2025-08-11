#include <exception>
#include <iomanip>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "test_tools.h"
#include "Device.hpp"
#include "linalg.hpp"
#include "Type.hpp"

namespace cytnx {

  using TestTools::dtype_list;
  using TestTools::InitTensorUniform;

  Tensor CalculateDeterminant(const Tensor& T) {
    // Regardless of whether the input tensor is on the CPU or CPU, the result tensor of Det is
    // always on the CPU, so we always initialize `determinant` on the CPU to avoid the error of
    // adding two tensors on different devices.
    Tensor determinant = zeros(1, T.dtype(), Device.cpu);
    size_t n = T.shape()[0];
    if (n == 1) {
      determinant.at({0}) = T.at({0, 0});
      return determinant;
    } else if (n == 2) {
      determinant.at({0}) = T.at({0, 0}) * T.at({1, 1}) - T.at({0, 1}) * T.at({1, 0});
      return determinant;
    }
    for (cytnx_uint64 a = 0; a < n; a++) {
      Tensor T2 = zeros({T.shape()[0] - 1, T.shape()[1] - 1}, T.dtype(), T.device());
      for (cytnx_uint64 i = 0; i < n - 1; i++) {
        for (cytnx_uint64 j = 0; j < n - 1; j++) {
          cytnx_uint64 ii = i + (i >= 0);
          cytnx_uint64 jj = j + (j >= a);
          T2.at({i, j}) = T.at({ii, jj});
        }
      }
      determinant({0}) += (a % 2 == 0 ? 1 : -1) * T.at({0, a}) * linalg::Det(T2);
    }
    return determinant;
  }

  cytnx_complex128 ExpectedDeterminant(const Tensor& T) {
    return CalculateDeterminant(Type_class::is_float(T.dtype()) ? T : T.astype(Type.Double))
      .astype(Type.ComplexDouble)
      .at<cytnx_complex128>({0});
  }

  cytnx_complex128 TestingDeterminant(const Tensor& T) {
    return linalg::Det(T).astype(Type.ComplexDouble).at<cytnx_complex128>({0});
  }

  cytnx_double Tolerance(cytnx_double testing_value) { return std::abs(testing_value) / 1.0e6; }

  MATCHER_P(ComplexDoubleEq, n, ::testing::PrintToString(n) + " (of type std::complex<double>)") {
    return ExplainMatchResult(::testing::DoubleNear(n.real(), Tolerance(arg.real())), arg.real(),
                              result_listener) &&
           ExplainMatchResult(::testing::DoubleNear(n.imag(), Tolerance(arg.imag())), arg.imag(),
                              result_listener);
  }

  /*=====test info=====
  describe:Test all possible data type and check the results.
  input:
    T:Tensor with shape {6, 6} or {2, 2}, {1, 1}, {3, 3}, {4, 4} and test for all
  possilbe data type.
  ====================*/
  TEST(Det, HandleAllDtype) {
    for (auto dtype : dtype_list) {
      Tensor T = Tensor({6, 6}, dtype, Device.cpu);
      InitTensorUniform(T, /*rand_seed=*/3);
      EXPECT_THAT(TestingDeterminant(T), ComplexDoubleEq(ExpectedDeterminant(T))) << T;

      T = Tensor({2, 2}, dtype, Device.cpu);
      InitTensorUniform(T, /*rand_seed=*/3);
      EXPECT_THAT(TestingDeterminant(T), ComplexDoubleEq(ExpectedDeterminant(T))) << T;

      T = Tensor({1, 1}, dtype, Device.cpu);
      InitTensorUniform(T, /*rand_seed=*/3);
      EXPECT_THAT(TestingDeterminant(T), ComplexDoubleEq(ExpectedDeterminant(T))) << T;

      T = Tensor({3, 3}, dtype, Device.cpu);
      InitTensorUniform(T, /*rand_seed=*/3);
      EXPECT_THAT(TestingDeterminant(T), ComplexDoubleEq(ExpectedDeterminant(T))) << T;

      T = Tensor({4, 4}, dtype, Device.cpu);
      InitTensorUniform(T, /*rand_seed=*/3);
      EXPECT_THAT(TestingDeterminant(T), ComplexDoubleEq(ExpectedDeterminant(T))) << T;
    }
  }

  /*=====test info=====
  describe:Test the tensor only 1 have one element. Test for all possible data type.
  input:
    T:Tensor with shape {1} on the CPU, testing for all possible data type.
  ====================*/
  TEST(Det, HandleSingleElementTensor) {
    for (auto dtype : dtype_list) {
      Tensor T = Tensor({1, 1}, dtype, Device.cpu);
      InitTensorUniform(T, /*rand_seed=*/6);
      EXPECT_THAT(TestingDeterminant(T), ComplexDoubleEq(ExpectedDeterminant(T))) << T;
    }
  }

  /*=====test info=====
  describe:Test for not contiguous tensor.
  input:
    T:Double data type not contiguous tensor with shape {9,9} on the CPU.
  ====================*/
  TEST(Det, HandleNotContiguous) {
    Tensor T = Tensor({9, 9}, Type.Double, Device.cpu);
    InitTensorUniform(T, /*rand_seed=*/1);
    // permute then they will not contiguous
    T.permute_({1, 0});  // shape:[9,9]
    EXPECT_THAT(TestingDeterminant(T), ComplexDoubleEq(ExpectedDeterminant(T))) << T;
  }

  // error test
  /*=====test info=====
  describe:Test the input tensors are both void tensor.
  input:
    T:void tensor
  ====================*/
  TEST(Det, ThrowsOnNonInitializedTensor) {
    Tensor T = Tensor();
    EXPECT_THROW(linalg::Det(T), std::logic_error);
  }

  /*=====test info=====
  describe:Test contains shared axis of the tensors are not same.
  input:
    T:double type tensor with shape {2, 3} on the CPU.
  ====================*/
  TEST(Det, ThrowsOnNonSquareTensor) {
    Tensor T = Tensor({2, 3}, Type.Double, Device.cpu);
    InitTensorUniform(T, /*rand_seed=*/0);
    EXPECT_THROW(linalg::Det(T), std::logic_error);
  }

}  // namespace cytnx
