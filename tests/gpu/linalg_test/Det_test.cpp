#include <gtest/gtest.h>

#include "../test_tools.h"

using namespace cytnx;
using namespace testing;
using namespace TestTools;

namespace DetTest {

  static cytnx_uint64 rand_seed1, rand_seed2;

  void ExcuteDetTest(const Tensor& T);

  void ErrorTestExcute(const Tensor& T);

  /*=====test info=====
  describe:Test all possible data type and check the results.
  input:
    T:Tensor with shape {6, 6} or {2, 2}, {1, 1}, {3, 3}, {4, 4} and test for all
  possilbe data type.
  ====================*/
  TEST(Det, allDType) {
    for (auto device : device_list) {
      for (auto dtype : dtype_list) {
        // The GPU version of Det is just too slow.
        // So we lower the size of the tensor from 6,6 to 3,3.
        Tensor T = Tensor({3, 3}, dtype, device);
        InitTensorUniform(T, rand_seed1 = 3);
        ExcuteDetTest(T);

        T = Tensor({2, 2}, dtype, device);
        InitTensorUniform(T, rand_seed1 = 3);
        ExcuteDetTest(T);

        T = Tensor({1, 1}, dtype, device);
        InitTensorUniform(T, rand_seed1 = 3);
        ExcuteDetTest(T);

        T = Tensor({3, 3}, dtype, device);
        InitTensorUniform(T, rand_seed1 = 3);
        ExcuteDetTest(T);

        T = Tensor({4, 4}, dtype, device);
        InitTensorUniform(T, rand_seed1 = 3);
        ExcuteDetTest(T);
      }
    }
  }

  /*=====test info=====
  describe:Test the tensor only 1 have one element. Test for all possible data type.
  input:
    T:Tensor with shape {1} on cpu, testing for all possible data type.
  ====================*/
  TEST(Det, one_elem_tens) {
    for (auto dtype : dtype_list) {
      Tensor T = Tensor({1, 1}, dtype);
      InitTensorUniform(T, rand_seed1 = 3);
      ExcuteDetTest(T);
    }
  }

  /*=====test info=====
  describe:Test for not contiguous tensor.
  input:
    T:Double data type not contiguous tensor with shape {9,9} on cpu.
  ====================*/
  TEST(Det, not_contiguous) {
    Tensor T = Tensor({9, 9}, Type.Double);
    InitTensorUniform(T, rand_seed1 = 1);
    // permute then they will not contiguous
    T.permute_({1, 0});  // shape:[9,9]
    ExcuteDetTest(T);
  }

  // error test
  /*=====test info=====
  describe:Test the input tensors are both void tensor.
  input:
    T:void tensor
  ====================*/
  TEST(Det, err_void_tens) {
    Tensor T = Tensor();
    ErrorTestExcute(T);
  }

  /*=====test info=====
  describe:Test contains shared axis of the tensors are not same.
  input:
    T:double type tensor with shape {2, 3} on cpu.
  ====================*/
  TEST(Det, err_axis_dim_wrong) {
    Tensor T = Tensor({2, 3});
    InitTensorUniform(T, rand_seed1 = 0);
    ErrorTestExcute(T);
  }

  Tensor ConstructExpectTens(const Tensor& T) {
    Tensor dst_T = zeros(1, T.dtype(), T.device());
    int n = T.shape()[0];
    if (n == 1) {
      dst_T.at({0}) = T.at({0, 0});
      return dst_T;
    } else if (n == 2) {
      dst_T.at({0}) = T.at({0, 0}) * T.at({1, 1}) - T.at({0, 1}) * T.at({1, 0});
      return dst_T;
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
      dst_T({0}) += (a % 2 == 0 ? 1 : -1) * T.at({0, a}) * ConstructExpectTens(T2);
    }
    return dst_T;
  }

  void ExcuteDetTest(const Tensor& T) {
    Tensor det_T = linalg::Det(T);
    Tensor expect_T;
    expect_T =
      ConstructExpectTens(T.dtype() > 4 ? T.astype(Type.Double).to(-1) : T.to(-1)).to(T.device());
    const double tolerance =
      std::pow(10.0, std::max((int)std::abs(std::log10(det_T(0).item<cytnx_double>())) - 6, 0));
    // Because the determinant will be casted to at least double, so we just cast the result to
    // ComplexDouble for comparison.
    EXPECT_TRUE(AreNearlyEqTensor(det_T.astype(Type.ComplexDouble),
                                  expect_T.astype(Type.ComplexDouble), tolerance));
  }

  void ErrorTestExcute(const Tensor& T) {
    try {
      auto dirsum_T = linalg::Det(T);
      std::cerr << "[Test Error] This test should throw error but not !" << std::endl;
      FAIL();
    } catch (const std::exception& ex) {
      auto err_msg = ex.what();
      std::cerr << err_msg << std::endl;
      SUCCEED();
    }
  }

}  // namespace DetTest
