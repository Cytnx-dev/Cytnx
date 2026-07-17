#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "gpu_test_tools.h"

namespace cytnx {
  namespace gpu_test {

    static cytnx_uint64 rand_seed1, rand_seed2;

    static void ExcuteDirectsumTest(const Tensor& T1, const Tensor& T2,
                                    const std::vector<cytnx_uint64> shared_axes);

    static void ErrorTestExcute(const Tensor& T1, const Tensor& T2,
                                const std::vector<cytnx_uint64> shared_axes);

    /*=====test info=====
    describe:Test all possible data type and check the results.
    input:
      T1:Tensor with shape {12, 5, 7} and test for all possilbe data type.
      T2:Tensor with shape {12, 5, 8} and test for all possilbe data type.
      axes:{1}
    ====================*/
    TEST(Directsum, GpuAllDType) {
      for (auto device : device_list) {
        for (auto dtype1 : dtype_list) {
          for (auto dtype2 : dtype_list) {
            Tensor T1 = Tensor({12, 5, 7}, dtype1, device).to(Device.cuda);
            Tensor T2 = Tensor({12, 5, 8}, dtype2, device).to(Device.cuda);
            InitTensorUniform(T1, rand_seed1 = 0);
            InitTensorUniform(T2, rand_seed2 = 1);
            std::vector<cytnx_uint64> shared_axes = {1};
            ExcuteDirectsumTest(T1, T2, shared_axes);
          }
        }
      }
    }

    /*=====test info=====
    describe:Test all possible combination (and permutation) of share axes.
    input:
      T1:double data type tensor with shape {7, 5, 3, 3} on gpu.
      T2:double data type tensor with shape {7, 9, 3, 3} on gpu.
      axes:test for all possible combination and permutation of the index {0, 2, 3}
    ====================*/
    TEST(Directsum, GpuSharedAxesCombination) {
      Tensor T1 = Tensor({7, 5, 3, 3}).to(Device.cuda);
      Tensor T2 = Tensor({7, 9, 3, 3}).to(Device.cuda);
      InitTensorUniform(T1, rand_seed1 = 0);
      InitTensorUniform(T2, rand_seed2 = 1);
      std::vector<std::vector<cytnx_uint64>> shared_axes_list = {
        {},  // empty
        {0},       {2},       {3},  // 1 elem
        {0, 2},    {2, 0},    {2, 3},    {3, 2},    {0, 3},    {3, 0},  // 2 elem
        {0, 2, 3}, {0, 3, 2}, {2, 0, 3}, {2, 3, 0}, {3, 0, 2}, {3, 2, 0}  // 3 elem
      };
      for (auto& shared_axes : shared_axes_list) {
        ExcuteDirectsumTest(T1, T2, shared_axes);
      }
    }

    /*=====test info=====
    describe:Test for share axes is empty vector.
    input:
      T1:double data type tensor with shape {2, 1, 2} on gpu.
      T2:double data type tensor with shape {2, 4, 2} on gpu.
      axes:empty
    ====================*/
    TEST(Directsum, GpuSharedAxesEmpty) {
      Tensor T1 = Tensor({2, 1, 2}).to(Device.cuda);
      Tensor T2 = Tensor({2, 4, 3}).to(Device.cuda);
      InitTensorUniform(T1, rand_seed1 = 0);
      InitTensorUniform(T2, rand_seed2 = 1);
      std::vector<cytnx_uint64> shared_axes = {};
      ExcuteDirectsumTest(T1, T2, shared_axes);
    }

    /*=====test info=====
    describe:Test the tensor only 1 have one element. Test for all possible data type.
    input:
      T1:Tensor with shape {1} on gpu, testing for all possible data type.
      T2:Tensor with shape {1} on gpu, testing for all possible data type.
      axes:test empty.
    ====================*/
    TEST(Directsum, GpuOneElemTens) {
      for (auto dtype : dtype_list) {
        Tensor T1 = Tensor({1}, dtype).to(Device.cuda);
        Tensor T2 = Tensor({1}, dtype).to(Device.cuda);
        InitTensorUniform(T1, rand_seed1 = 0);
        InitTensorUniform(T2, rand_seed2 = 1);
        std::vector<cytnx_uint64> shared_axes = {};
        ExcuteDirectsumTest(T1, T2, shared_axes);
      }
    }

    /*=====test info=====
    describe:Test for matrix case.
    input:
      T1:Tensor with shape {3, 2} on gpu, testing for all possible data type.
      T2:Tensor with shape {3, 2} on gpu, testing for all possible data type.
      axes:empty, {0}, {1}.
    ====================*/
    TEST(Directsum, GpuMatrixCase) {
      std::vector<std::vector<cytnx_uint64>> shared_axes_list = {{}, {0}, {1}, {0, 1}, {1, 0}};
      for (auto dtype : dtype_list) {
        Tensor T1 = Tensor({3, 2}, dtype).to(Device.cuda);
        Tensor T2 = Tensor({3, 2}, dtype).to(Device.cuda);
        InitTensorUniform(T1, rand_seed1 = 0);
        InitTensorUniform(T2, rand_seed2 = 1);
        for (auto& shared_axes : shared_axes_list) {
          ExcuteDirectsumTest(T1, T2, shared_axes);
        }
      }
    }

    /*=====test info=====
    describe:Test two tensor are reference copy.
    input:
      T1:Tensor with shape {3, 2} on gpu, testing for all possible data type.
      T2:T2=T1
      axes:empty, {0}, {1}.
    ====================*/
    TEST(Directsum, GpuTensShareMemory) {
      std::vector<std::vector<cytnx_uint64>> shared_axes_list = {{}, {0}, {1}};
      for (auto dtype : dtype_list) {
        Tensor T1 = Tensor({3, 2}, dtype).to(Device.cuda);
        InitTensorUniform(T1);
        auto T2 = T1;
        for (auto& shared_axes : shared_axes_list) {
          ExcuteDirectsumTest(T1, T2, shared_axes);
        }
      }
    }

    /*=====test info=====
    describe:Test the shared axes contain all axes.
    input:
      T1:complex double type tensor with shape {2, 3} on gpu.
      T2:double type tensor with shape {2, 3} on gpu.
      axes:{0, 1}
    ====================*/
    TEST(Directsum, GpuSharedAxisContainsAll) {
      Tensor T1 = Tensor({2, 3}, Type.ComplexDouble).to(Device.cuda);
      Tensor T2 = Tensor({2, 3}, Type.Double).to(Device.cuda);
      InitTensorUniform(T1, rand_seed1 = 0);
      InitTensorUniform(T2, rand_seed2 = 1);
      std::vector<cytnx_uint64> shared_axes = {0, 1};
      ExcuteDirectsumTest(T1, T2, shared_axes);
    }

    /*=====test info=====
    describe:Test the shared axes contain all axes. Input tensors have only one elem.
    input:
      T1:complex double type tensor with shape {1} on gpu.
      T2:double type tensor with shape {1} on gpu.
      axes:{0}
    ====================*/
    TEST(Directsum, GpuSharedAxisContainsAllTensOneElem) {
      Tensor T1 = Tensor({1}, Type.ComplexDouble).to(Device.cuda);
      Tensor T2 = Tensor({1}, Type.Double).to(Device.cuda);
      InitTensorUniform(T1, rand_seed1 = 0);
      InitTensorUniform(T2, rand_seed2 = 1);
      std::vector<cytnx_uint64> shared_axes = {0};
      ExcuteDirectsumTest(T1, T2, shared_axes);
    }

    /*=====test info=====
    describe:Test for not contiguous tensor.
    input:
      T1:int32 data type not contiguous tensor with shape {5, 7, 3, 3} on gpu.
      T2:double data type not contiguous tensor with shape {9, 7, 3, 3} on gpu.
      axes:empty
    ====================*/
    TEST(Directsum, GpuNotContiguous) {
      Tensor T1 = Tensor({7, 5, 3, 3}, Type.Int32).to(Device.cuda);
      Tensor T2 = Tensor({7, 9, 3, 3}, Type.Double).to(Device.cuda);
      InitTensorUniform(T1, rand_seed1 = 0);
      InitTensorUniform(T2, rand_seed2 = 1);
      // permute then they will not contiguous
      T1.permute_({1, 0, 2, 3});  // shape:[5, 7, 3, 3]
      T2.permute_({1, 0, 2, 3});  // shape:[9, 7, 3, 3]
      std::vector<std::vector<cytnx_uint64>> shared_axes_list = {
        {},  // empty
        {1},       {2},       {3},  // 1 elem
        {1, 2},    {2, 1},    {2, 3},    {3, 2},    {1, 3},    {3, 1},  // 2 elem
        {1, 2, 3}, {1, 3, 2}, {2, 1, 3}, {2, 3, 1}, {3, 1, 2}, {3, 2, 1}  // 3 elem
      };
      for (auto& shared_axes : shared_axes_list) {
        ExcuteDirectsumTest(T1, T2, shared_axes);
      }
    }

    // error test
    /*=====test info=====
    describe:Test the input tensors are both void tensor.
    input:
      T1:void tensor
      T2:void tensor
      axes:empty
    ====================*/
    TEST(Directsum, GpuErrVoidTens) {
      Tensor T1 = Tensor();
      Tensor T2 = Tensor();
      std::vector<cytnx_uint64> shared_axes = {};
      ErrorTestExcute(T1, T2, shared_axes);
    }

    /*=====test info=====
    describe:Test the rank of the input tensors are not same.
    input:
      T1:double type tensor with shape {2} on gpu.
      T2:double type tensor with shape {2, 1} on gpu.
      axes:empty
    ====================*/
    TEST(Directsum, GpuErrDiffRank) {
      Tensor T1 = Tensor({2}).to(Device.cuda);
      Tensor T2 = Tensor({2, 1}).to(Device.cuda);
      InitTensorUniform(T1, rand_seed1 = 0);
      InitTensorUniform(T2, rand_seed2 = 1);
      std::vector<cytnx_uint64> shared_axes = {};
      ErrorTestExcute(T1, T2, shared_axes);
    }

    /*=====test info=====
    describe:Test contains shared axis of the tensors are not same.
    input:
      T1:double type tensor with shape {2, 3, 3} on gpu.
      T2:double type tensor with shape {2, 1, 3} on gpu.
      axes:{2, 1}
    ====================*/
    TEST(Directsum, GpuErrSharedAxisDimWrong) {
      Tensor T1 = Tensor({2, 3, 3}).to(Device.cuda);
      Tensor T2 = Tensor({2, 1, 3}).to(Device.cuda);
      InitTensorUniform(T1, rand_seed1 = 0);
      InitTensorUniform(T2, rand_seed2 = 1);
      std::vector<cytnx_uint64> shared_axes = {2, 1};
      ErrorTestExcute(T1, T2, shared_axes);
    }

    /*=====test info=====
    describe:Test the shared axes out of the range.
    input:
      T1:double type tensor with shape {2, 3, 3} on gpu.
      T2:double type tensor with shape {2, 1, 3} on gpu.
      axes:{3}
    ====================*/
    TEST(Directsum, GpuErrSharedAxisOutRange) {
      Tensor T1 = Tensor({2, 3, 3}).to(Device.cuda);
      Tensor T2 = Tensor({2, 1, 3}).to(Device.cuda);
      InitTensorUniform(T1, rand_seed1 = 0);
      InitTensorUniform(T2, rand_seed2 = 1);
      std::vector<cytnx_uint64> shared_axes = {3};
      ErrorTestExcute(T1, T2, shared_axes);
    }

    /*=====test info=====
    describe:Test contains the shared axes out of the range.
    input:
      T1:double type tensor with shape {2, 3, 3} on gpu.
      T2:double type tensor with shape {2, 1, 3} on gpu.
      axes:{0, 3}
    ====================*/
    TEST(Directsum, GpuErrOneSharedAxisOutRange) {
      Tensor T1 = Tensor({2, 3, 3}).to(Device.cuda);
      Tensor T2 = Tensor({2, 1, 3}).to(Device.cuda);
      InitTensorUniform(T1, rand_seed1 = 0);
      InitTensorUniform(T2, rand_seed2 = 1);
      std::vector<cytnx_uint64> shared_axes = {3, 0};
      ErrorTestExcute(T1, T2, shared_axes);
    }

    /*=====test info=====
    describe:Test the shared axes not uniqe.
    input:
      T1:double type tensor with shape {2, 3, 3} on gpu.
      T2:double type tensor with shape {2, 1, 3} on gpu.
      axes:{0, 0}
    ====================*/
    TEST(Directsum, GpuErrSharedAxisNotUniqe) {
      Tensor T1 = Tensor({2, 3, 3}).to(Device.cuda);
      Tensor T2 = Tensor({2, 1, 3}).to(Device.cuda);
      InitTensorUniform(T1, rand_seed1 = 0);
      InitTensorUniform(T2, rand_seed2 = 1);
      std::vector<cytnx_uint64> shared_axes = {0, 0};
      ErrorTestExcute(T1, T2, shared_axes);
    }

    int CheckWhichRange(const std::vector<cytnx_uint64>& T1_shape,
                        const std::vector<cytnx_uint64>& idxs,
                        const std::vector<cytnx_uint64>& non_share_axes) {
      /*
       This function check, for the given output indices, which source tensor element should be
       take. For example, T1 shape = [2, 3, 3], non_share_axes = [1, 2], and case 1: if idxs (1,
       2, 2), then we need to find the element in T1. --> return 1. case 2: if idxs (1, 4, 4),
       then we need to find the element in T2. --> return 2. case 3: if idxs (1, 2, 4), then the
       element neither belons to T1 nor T2. --> return 0.
      */
      std::vector<int> check_list;
      for (auto& i : non_share_axes) {
        check_list.push_back(
          (static_cast<int>(idxs[i] + 1) - static_cast<int>(T1_shape[i])) <= 0 ? -1 : 1);
      }
      /*
        the elements of check list need to be all equal then the indices belongs to T1 or T2.
        Otherwise, non. For example, T1 shape = [2, 3, 3], non_share_axes = [1, 2], and case 1: if
        idxs (1, 2, 2), check_list = [-1, -1], check_list elements all equal to -1 --> belongs to
        T1. case 2: if idxs (1, 4, 4), check_list = [ 1,  1], check_list elements all equal to  1
        --> belongs to T2. case 3: if idxs (1, 2, 4), check_list = [-1,  1], check_list elements
        not all equal --> neither T1 nor T2.
      */
      bool all_eq = false;
      if (check_list.size() == 1) {
        all_eq = true;
      } else {
        all_eq = std::equal(check_list.begin() + 1, check_list.end(), check_list.begin());
      }
      if (all_eq) {
        return check_list[0] == -1 ? 1 : 2;
      } else {
        return 0;
      }
    }

    std::vector<cytnx_uint64> GetSrcIdxs(const std::vector<cytnx_uint64>& dst_axes,
                                         const std::vector<cytnx_uint64>& T1_shape,
                                         const std::vector<cytnx_uint64>& idxs,
                                         const std::vector<cytnx_uint64>& share_axes,
                                         int& which_T) {
      std::vector<cytnx_uint64> src_idxs(dst_axes.size());
      std::vector<cytnx_uint64> non_share_axes;
      for (auto& i : share_axes) {
        src_idxs[i] = idxs[i];
      }
      for (size_t i = 0; i < dst_axes.size(); ++i) {
        if (std::find(share_axes.begin(), share_axes.end(), i) == share_axes.end()) {  // not found
          non_share_axes.push_back(i);
        }
      }
      which_T = CheckWhichRange(T1_shape, idxs, non_share_axes);
      switch (which_T) {
        case 1:  // belongs to T1
          for (auto& i : non_share_axes) {
            src_idxs[i] = idxs[i];
          }
          break;
        case 2:  // belons to T2
          for (auto& i : non_share_axes) {
            src_idxs[i] = idxs[i] - T1_shape[i];
          }
          break;
        case 0:  // neither
          break;
      }
      return src_idxs;
    }

    void SetDstElem(const std::vector<cytnx_uint64>& dst_axes,
                    const std::vector<cytnx_uint64>& axes, const std::vector<cytnx_uint64>& idxs,
                    const Tensor& T1, const Tensor& T2, Tensor& dst_T) {
      int which_T = -1;
      auto src_idxs = GetSrcIdxs(dst_axes, T1.shape(), idxs, axes, which_T);
      if (which_T == 0) {
        dst_T.at(idxs) = 0;
      } else {
        auto val = which_T == 1 ? T1.at(src_idxs) : T2.at(src_idxs);
        dst_T.at(idxs) = val;
      }
    }

    void RankSetElem(const std::vector<cytnx_uint64>& dst_axes,
                     const std::vector<cytnx_uint64>& axes, const Tensor& T1, const Tensor& T2,
                     Tensor& dst_T) {
      auto rank = T1.rank();
      switch (rank) {
        case 0:
          break;
        case 1:
          for (cytnx_uint64 i1 = 0; i1 < dst_axes[0]; ++i1) {
            auto idx = std::vector<cytnx_uint64>{i1};
            SetDstElem(dst_axes, axes, idx, T1, T2, dst_T);
          }
          break;
        case 2:
          for (cytnx_uint64 i1 = 0; i1 < dst_axes[0]; ++i1) {
            for (cytnx_uint64 i2 = 0; i2 < dst_axes[1]; ++i2) {
              auto idx = std::vector<cytnx_uint64>{i1, i2};
              SetDstElem(dst_axes, axes, idx, T1, T2, dst_T);
            }
          }
          break;
        case 3:
          for (cytnx_uint64 i1 = 0; i1 < dst_axes[0]; ++i1) {
            for (cytnx_uint64 i2 = 0; i2 < dst_axes[1]; ++i2) {
              for (cytnx_uint64 i3 = 0; i3 < dst_axes[2]; ++i3) {
                auto idx = std::vector<cytnx_uint64>{i1, i2, i3};
                SetDstElem(dst_axes, axes, idx, T1, T2, dst_T);
              }
            }
          }
          break;
        case 4:
          for (cytnx_uint64 i1 = 0; i1 < dst_axes[0]; ++i1) {
            for (cytnx_uint64 i2 = 0; i2 < dst_axes[1]; ++i2) {
              for (cytnx_uint64 i3 = 0; i3 < dst_axes[2]; ++i3) {
                for (cytnx_uint64 i4 = 0; i4 < dst_axes[3]; ++i4) {
                  auto idx = std::vector<cytnx_uint64>{i1, i2, i3, i4};
                  SetDstElem(dst_axes, axes, idx, T1, T2, dst_T);
                }
              }
            }
          }
          break;
        case 5:
          for (cytnx_uint64 i1 = 0; i1 < dst_axes[0]; ++i1) {
            for (cytnx_uint64 i2 = 0; i2 < dst_axes[1]; ++i2) {
              for (cytnx_uint64 i3 = 0; i3 < dst_axes[2]; ++i3) {
                for (cytnx_uint64 i4 = 0; i4 < dst_axes[3]; ++i4) {
                  for (cytnx_uint64 i5 = 0; i5 < dst_axes[4]; ++i5) {
                    auto idx = std::vector<cytnx_uint64>{i1, i2, i3, i4, i5};
                    SetDstElem(dst_axes, axes, idx, T1, T2, dst_T);
                  }
                }
              }
            }
          }
          break;
        default:
          break;
      }  // switch
    }

    static Tensor ConstructExpectTens(const Tensor& T1, const Tensor& T2,
                                      const std::vector<cytnx_uint64> shared_axes) {
      auto rank = T1.rank();
      // promote across the real/complex boundary (Type.type_promote), not the lower-enum operand.
      auto expect_dtype = Type.type_promote(T1.dtype(), T2.dtype());
      auto device = T1.device();
      std::vector<cytnx_uint64> dst_axes(T1.rank());
      for (auto i = 0; i < T1.rank(); ++i) {
        if (std::find(shared_axes.begin(), shared_axes.end(), i) != shared_axes.end()) {
          dst_axes[i] = T1.shape()[i];
        } else {
          dst_axes[i] = T1.shape()[i] + T2.shape()[i];
        }
      }
      Tensor dst_T = zeros(dst_axes, expect_dtype, device);
      RankSetElem(dst_axes, shared_axes, T1, T2, dst_T);
      return dst_T;
    }

    static void ExcuteDirectsumTest(const Tensor& T1, const Tensor& T2,
                                    const std::vector<cytnx_uint64> shared_axes) {
      auto dirsum_T = linalg::Directsum(T1, T2, shared_axes);
      Tensor expect_T;
      // if shared axes contain all axes, the output is equal to T2 but convert to strongest type.
      if (shared_axes.size() == T1.rank()) {
        // convert T2 to the promoted type
        auto expect_dtype = Type.type_promote(T1.dtype(), T2.dtype());
        expect_T = T2.astype(expect_dtype);
      } else {
        expect_T = ConstructExpectTens(T1, T2, shared_axes);
      }
      EXPECT_TRUE(AreEqTensor(dirsum_T, expect_T));
    }

    static void ErrorTestExcute(const Tensor& T1, const Tensor& T2,
                                const std::vector<cytnx_uint64> shared_axes) {
      try {
        auto dirsum_T = linalg::Directsum(T1, T2, shared_axes);
        std::cerr << "[Test Error] This test should throw error but not !" << std::endl;
        FAIL();
      } catch (const std::exception& ex) {
        auto err_msg = ex.what();
        std::cerr << err_msg << std::endl;
        SUCCEED();
      }
    }

  }  // namespace gpu_test
}  // namespace cytnx
