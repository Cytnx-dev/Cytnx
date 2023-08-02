#include "cytnx.hpp"
#include <gtest/gtest.h>
#include "../test_tools.h"

using namespace cytnx;
using namespace testing;
using namespace TestTools;

namespace HstackTest {

  void CheckResult(const Tensor& hstack_tens, const std::vector<Tensor>& input_tens);

  /*=====test info=====
  describe:Input only one tensor. Test all possible data type on cpu device.
  input:One tensor with shape = (4, 3), all possible type on cpu device.
  ====================*/
  TEST(Hstack, gpu_only_one_tensor) {
    for (auto dtype : dtype_list) {
      if (dtype == Type.Bool)  // if both bool type, it will throw error
        continue;
      std::vector<Tensor> Ts = {Tensor({4, 3}, dtype).to(cytnx::Device.cuda)};
      InitTensorUniform(Ts);
      Tensor hstack_tens = algo::Hstack(Ts);
      CheckResult(hstack_tens, Ts);
    }
  }

  /*=====test info=====
  describe:Input multiple tensor with same type. Test all possible data type on cpu device.
  input:Three tensor with the shapes [(4, 3), (4, 2), (4, 5)] and same data type. Test all possible
  type on cpu device.
  ====================*/
  TEST(Hstack, gpu_multi_tensor) {
    for (auto dtype : dtype_list) {
      if (dtype == Type.Bool)  // if both bool type, it will throw error
        continue;
      std::vector<Tensor> Ts = {Tensor({4, 3}, dtype).to(cytnx::Device.cuda),
                                Tensor({4, 2}, dtype).to(cytnx::Device.cuda),
                                Tensor({4, 5}, dtype).to(cytnx::Device.cuda)};
      InitTensorUniform(Ts);
      Tensor hstack_tens = algo::Hstack(Ts);
      CheckResult(hstack_tens, Ts);
    }
  }

  /*=====test info=====
  describe:Test arbitray two data type.
  input:Three tensor on cpu device with the shapes [(4, 3), (4, 2)] but the data type may be
  different. Test all possible data type with arbitrary combination.
  ====================*/
  TEST(Hstack, gpu_two_type_tensor) {
    for (auto dtype1 : dtype_list) {
      for (auto dtype2 : dtype_list) {
        if (dtype1 == Type.Bool &&
            dtype2 == Type.Bool)  // if both are bool type, it will throw error.
          continue;
        std::vector<Tensor> Ts = {Tensor({4, 3}, dtype1).to(cytnx::Device.cuda),
                                  Tensor({4, 2}, dtype2).to(cytnx::Device.cuda)};
        InitTensorUniform(Ts);
        Tensor hstack_tens = algo::Hstack(Ts);
        CheckResult(hstack_tens, Ts);
      }
    }
  }

  /*=====test info=====
  describe:Test multiple tensor with different data type.
  input:Five tensors as following
    T1:shape (4, 2) with bool type on cpu device.
    T2:shape (4, 3) with complex double type on cpu device.
    T3:shape (4, 2) with double type on cpu device.
    T4:shape (4, 1) with double type on cpu device.
    T5:shape (4, 5) with uint64 type on cpu device.
  ====================*/
  TEST(Hstack, gpu_diff_type_tensor) {
    std::vector<Tensor> Ts = {Tensor({4, 2}, Type.Bool).to(cytnx::Device.cuda),
                              Tensor({4, 3}, Type.ComplexDouble).to(cytnx::Device.cuda),
                              Tensor({4, 2}, Type.Double).to(cytnx::Device.cuda),
                              Tensor({4, 1}, Type.Double).to(cytnx::Device.cuda),
                              Tensor({4, 5}, Type.Uint64).to(cytnx::Device.cuda)};
    InitTensorUniform(Ts);
    Tensor hstack_tens = algo::Hstack(Ts);
    CheckResult(hstack_tens, Ts);
  }

  /*=====test info=====
  describe:Test non contiguous tensor.
  input:Three tensors as following
    T1:shape (4, 2) with non contiguous, bool type on cpu device.
    T2:shape (4, 3) with non contiguous complex double type on cpu device.
    T3:shape (4, 1) with non contiguous double type on cpu device.
  ====================*/
  TEST(Hstack, gpu_non_contiguous_tensor) {
    std::vector<Tensor> Ts = {
      Tensor({2, 4}, Type.Bool).to(cytnx::Device.cuda),
      Tensor({3, 4}, Type.ComplexDouble).to(cytnx::Device.cuda),
      Tensor({1, 4}, Type.Double).to(cytnx::Device.cuda),
    };
    InitTensorUniform(Ts);
    // permute the it will not be contiguous
    for (auto& T : Ts) {
      T.permute_({1, 0});
      ASSERT_FALSE(T.is_contiguous());
    }
    Tensor hstack_tens = algo::Hstack(Ts);
    CheckResult(hstack_tens, Ts);
  }

  // error test

  /*=====test info=====
  describe:Test input empty vector.
  input:empty vector, cpu
  ====================*/
  TEST(Hstack, gpu_err_empty) {
    std::vector<Tensor> empty;
    EXPECT_THROW({ Tensor tens = algo::Hstack(empty); }, std::logic_error);
  }

  /*=====test info=====
  describe:Test input is void tensor
  input:void tensor, cpu
  ====================*/
  TEST(Hstack, gpu_err_tensor_void) {
    std::vector<Tensor> Ts = {Tensor()};
    InitTensorUniform(Ts);
    EXPECT_THROW({ Tensor tens = algo::Hstack(Ts); }, std::logic_error);
  }

  /*=====test info=====
  describe:Test input contains void tensor
  input:
    T1:shape (4, 3), data type is double, cpu
    T2:void tensor
  ====================*/
  TEST(Hstack, gpu_err_contains_void) {
    std::vector<Tensor> Ts = {Tensor({4, 3}, Type.Double).to(cytnx::Device.cuda), Tensor()};
    InitTensorUniform(Ts);
    EXPECT_THROW({ Tensor tens = algo::Hstack(Ts); }, std::logic_error);
  }

  /*=====test info=====
  describe:Test input different row matrix.
  input:
    T1:shape (4, 3), data type is double, cpu
    T2:shape (2, 3), data type is double, cpu
  ====================*/
  TEST(Hstack, gpu_err_row_not_eq) {
    std::vector<Tensor> Ts = {Tensor({4, 3}, Type.Double).to(cytnx::Device.cuda),
                              Tensor({2, 3}, Type.Double).to(cytnx::Device.cuda)};
    InitTensorUniform(Ts);
    EXPECT_THROW({ Tensor tens = algo::Hstack(Ts); }, std::logic_error);
  }

  /*=====test info=====
  describe:Test input bool type.
  input:
    T:shape (2, 3), data type is bool, cpu
  ====================*/
  TEST(Hstack, gpu_err_a_bool_type) {
    std::vector<Tensor> Ts = {Tensor({2, 3}, Type.Bool).to(cytnx::Device.cuda)};
    InitTensorUniform(Ts);
    EXPECT_THROW({ Tensor tens = algo::Hstack(Ts); }, std::logic_error);
  }

  /*=====test info=====
  describe:Test input types are all bool.
  input:
    T1:shape (4, 3), data type is bool, cpu
    T1:shape (4, 2), data type is bool, cpu
  ====================*/
  TEST(Hstack, gpu_err_multi_bool_type) {
    std::vector<Tensor> Ts = {Tensor({4, 3}, Type.Bool).to(cytnx::Device.cuda),
                              Tensor({4, 2}, Type.Bool).to(cytnx::Device.cuda)};
    InitTensorUniform(Ts);
    EXPECT_THROW({ Tensor tens = algo::Hstack(Ts); }, std::logic_error);
  }

  void CheckResult(const Tensor& hstack_tens, const std::vector<Tensor>& input_tens) {
    // 1. check tensor data type
    std::vector<unsigned int> input_types;
    for (size_t i = 0; i < input_tens.size(); ++i) {
      input_types.push_back(input_tens[i].dtype());
    }
    // since complex double < complex float < double < ... < bool, we need to
    //   find the min value type as the final converted type.
    auto expect_dtype = *std::min_element(input_types.begin(), input_types.end());
    EXPECT_EQ(expect_dtype, hstack_tens.dtype());

    // 2. check tensor shape
    auto hstack_shape = hstack_tens.shape();
    EXPECT_EQ(hstack_shape.size(), 2);  // need to be matrix
    int D_share = input_tens[0].shape()[0];
    int D_total = 0;
    for (auto tens : input_tens) {
      ASSERT_EQ(tens.shape()[0], D_share);  // all column need to be same
      D_total += tens.shape()[1];
    }
    EXPECT_EQ(hstack_shape[0], D_share);
    EXPECT_EQ(hstack_shape[1], D_total);

    // 3. check tensor elements
    EXPECT_TRUE(hstack_tens.is_contiguous());
    int block_col_shift = 0;
    bool is_same_elem = true;
    for (auto tens : input_tens) {
      auto cvt_tens = tens.astype(expect_dtype);
      auto src_shape = tens.shape();
      auto r_num = src_shape[0];  // row number
      auto c_num = src_shape[1];  // column number
      for (int r = 0; r < r_num; ++r) {
        for (int c = 0; c < c_num; ++c) {
          auto dst_c = c + block_col_shift;
          is_same_elem = AreElemSame(cvt_tens, {r, c}, hstack_tens, {r, dst_c});
          if (!is_same_elem) break;
        }  // end col
        if (!is_same_elem) break;
      }  // end row
      if (!is_same_elem) break;
      block_col_shift += c_num;
    }  // end input tens vec
    EXPECT_TRUE(is_same_elem);
  }  // fucn:CheckResult

}  // namespace HstackTest
