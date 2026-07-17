#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "gpu_test_tools.h"
namespace cytnx {
  namespace {

    namespace VsplitTest {

      void CheckResult(const Tensor& T_in, const std::vector<Tensor>& vsplit_tens,
                       const std::vector<cytnx_uint64>& dims);

      /*=====test info=====
      describe:Test 'dims' only one element.
      input:
        Tin:shape (7, 10), all possible data type.
        dims:[7]
      ====================*/
      TEST(Vsplit, GpuDimOneElem) {
        for (const auto& dtype : dtype_list) {
          if (dtype == Type.Bool) continue;
          Tensor T = Tensor({7, 10}, dtype).to(Device.cuda);
          InitTensorUniform(T);
          std::vector<cytnx_uint64> dims = {7};
          std::vector<Tensor> Ts = algo::Vsplit(T, dims);
          CheckResult(T, Ts, dims);
        }
      }

      /*=====test info=====
      describe:normal test.
      input:
        Tin:shape (7, 10), all possible data type.
        dims:[2, 3, 2]
      ====================*/
      TEST(Vsplit, GpuNormalTest) {
        for (const auto& dtype : dtype_list) {
          if (dtype == Type.Bool) continue;
          Tensor T = Tensor({7, 10}, dtype).to(Device.cuda);
          InitTensorUniform(T);
          std::vector<cytnx_uint64> dims = {2, 3, 2};
          std::vector<Tensor> Ts = algo::Vsplit(T, dims);
          CheckResult(T, Ts, dims);
        }
      }

      /*=====test info=====
      describe:Test 'dims' elements all 1.
      input:
        Tin:shape (7, 10), all possible data type.
        dims:[1, 1, 1, 1, 1, 1, 1]
      ====================*/
      TEST(Vsplit, GpuDimAllOne) {
        for (const auto& dtype : dtype_list) {
          if (dtype == Type.Bool) continue;
          Tensor T = Tensor({7, 10}, dtype).to(Device.cuda);
          InitTensorUniform(T);
          std::vector<cytnx_uint64> dims = {1, 1, 1, 1, 1, 1, 1};
          std::vector<Tensor> Ts = algo::Vsplit(T, dims);
          CheckResult(T, Ts, dims);
        }
      }

      /*=====test info=====
      describe:Test input tensor is non contiguous.
      input:
        Tin:not contiguous, shape (7, 10), all possible data type.
        dims:[2, 3, 2]
      ====================*/
      TEST(Vsplit, GpuNotContiguous) {
        for (const auto& dtype : dtype_list) {
          if (dtype == Type.Bool) continue;
          Tensor T = Tensor({10, 7}, dtype).to(Device.cuda);
          T.permute_(1, 0);
          InitTensorUniform(T);
          std::vector<cytnx_uint64> dims = {2, 3, 2};
          std::vector<Tensor> Ts = algo::Vsplit(T, dims);
          CheckResult(T, Ts, dims);
        }
      }

      /*=====test info=====
      describe:Test API void Vsplit_(...).
      input:
        Tout:empty.
        Tin:shape (7, 10), all possible data type.
        dims:[2, 3, 2]
      ====================*/
      TEST(Vsplit, GpuVoidAPI) {
        for (const auto& dtype : dtype_list) {
          if (dtype == Type.Bool) continue;
          Tensor T = Tensor({7, 10}, dtype).to(Device.cuda);
          InitTensorUniform(T);
          std::vector<cytnx_uint64> dims = {2, 3, 2};
          std::vector<Tensor> Ts;
          algo::Vsplit_(Ts, T, dims);
          CheckResult(T, Ts, dims);
        }
      }

      /*=====test info=====
      describe:Test API void Vsplit_(...), output not empty.
      input:
        Tout:not empty object.
        Tin:shape (7, 10), all possible data type.
        dims:[2, 3, 2]
      ====================*/
      TEST(Vsplit, GpuVoidAPIOutputNotEmpty) {
        for (const auto& dtype : dtype_list) {
          if (dtype == Type.Bool) continue;
          Tensor T = Tensor({7, 10}, dtype).to(Device.cuda);
          InitTensorUniform(T);
          std::vector<cytnx_uint64> dims = {2, 3, 2};
          int len;
          std::vector<Tensor> Ts = std::vector<Tensor>(len = 5, T.clone());
          algo::Vsplit_(Ts, T, dims);
          CheckResult(T, Ts, dims);
        }
      }

      // error test

      /*=====test info=====
      describe:Test input bool data type tensor.
      input:
        Tin:shape (7, 10), bool type.
        dims:[2, 3, 2]
      ====================*/
      TEST(Vsplit, GpuErrInputBoolType) {
        Tensor T = Tensor({7, 10}, Type.Bool).to(Device.cuda);
        InitTensorUniform(T);
        std::vector<cytnx_uint64> dims = {2, 3, 2};
        EXPECT_THROW({ auto Ts = algo::Vsplit(T, dims); }, std::logic_error);
      }

      /*=====test info=====
      describe:Test input void tensor.
      input:
        Tin:void tensor.
        dims:[]
      ====================*/
      TEST(Vsplit, GpuErrVoidTensor) {
        Tensor T = Tensor();
        std::vector<cytnx_uint64> dims = {};
        EXPECT_THROW({ auto Ts = algo::Vsplit(T, dims); }, std::logic_error);
      }

      /*=====test info=====
      describe:Test input not matrix tensor.
      input:
        Tin:tensor with shape [1].
        dims:[2, 3, 2]
      ====================*/
      TEST(Vsplit, GpuErrNotMat) {
        Tensor T = Tensor({7}, Type.Bool).to(Device.cuda);
        std::vector<cytnx_uint64> dims = {2, 3, 2};
        EXPECT_THROW({ auto Ts = algo::Vsplit(T, dims); }, std::logic_error);
      }

      /*=====test info=====
      describe:Test empty dims.
      input:
        Tin:shape (7, 10), double type.
        dims:[]
      ====================*/
      TEST(Vsplit, GpuErrDimsEmpty) {
        Tensor T = Tensor({7, 10}, Type.Double).to(Device.cuda);
        InitTensorUniform(T);
        std::vector<cytnx_uint64> dims = {};
        EXPECT_THROW({ auto Ts = algo::Vsplit(T, dims); }, std::logic_error);
      }

      /*=====test info=====
      describe:Test dims contains zero.
      input:
        Tin:shape (7, 10), double type.
        dims:[3, 4, 0]
      ====================*/
      TEST(Vsplit, GpuErrDimsContainsZero) {
        Tensor T = Tensor({7, 10}, Type.Double).to(Device.cuda);
        InitTensorUniform(T);
        std::vector<cytnx_uint64> dims = {3, 4, 0};
        EXPECT_THROW({ auto Ts = algo::Vsplit(T, dims); }, std::logic_error);
      }

      /*=====test info=====
      describe:Test dims not match.
      input:
        Tin:shape (7, 10), double type.
        dims:[3, 4, 1]
      ====================*/
      TEST(Vsplit, GpuErrDimsNotMatch) {
        Tensor T = Tensor({7, 10}, Type.Double).to(Device.cuda);
        InitTensorUniform(T);
        std::vector<cytnx_uint64> dims = {3, 4, 0};
        EXPECT_THROW({ auto Ts = algo::Vsplit(T, dims); }, std::logic_error);
      }

      void CheckResult(const Tensor& T_in, const std::vector<Tensor>& vsplit_tens,
                       const std::vector<cytnx_uint64>& dims) {
        // 1. check tensor data type
        for (const auto& tens : vsplit_tens) {
          EXPECT_EQ(T_in.dtype(), tens.dtype());
        }

        // 2. check tensor shape
        std::vector<cytnx_uint64> in_shape = T_in.shape();
        ASSERT_EQ(in_shape.size(), 2);  // need to be matrix
        EXPECT_EQ(vsplit_tens.size(), dims.size());
        int D_total = in_shape[0];
        int D_share = in_shape[1];
        int D_accum = 0;
        int i = 0;
        for (auto tens : vsplit_tens) {
          auto split_shape = tens.shape();
          EXPECT_EQ(split_shape.size(), 2);  // need to be matrix
          EXPECT_EQ(split_shape[1], D_share);  // all column need to be same
          EXPECT_EQ(split_shape[0], dims[i++]);
          D_accum += split_shape[0];
        }
        EXPECT_EQ(D_accum, D_total);

        // 3. check tensor elements
        int block_row_shift = 0;
        bool is_same_elem = true;
        for (auto tens : vsplit_tens) {
          EXPECT_TRUE(tens.is_contiguous());
          auto split_shape = tens.shape();
          auto r_num = split_shape[0];  // row number
          auto c_num = split_shape[1];  // column number
          for (cytnx_uint64 r = 0; r < r_num; ++r) {
            auto in_r = r + block_row_shift;
            for (cytnx_uint64 c = 0; c < c_num; ++c) {
              is_same_elem = AreElemSame(T_in, {in_r, c}, tens, {r, c});
              if (!is_same_elem) break;
            }  // end col
            if (!is_same_elem) break;
          }  // end row
          if (!is_same_elem) break;
          block_row_shift += r_num;
        }  // end input tens vec
        EXPECT_TRUE(is_same_elem);
      }  // fucn:CheckResult

    }  // namespace VsplitTest
  }  // namespace
}  // namespace cytnx
