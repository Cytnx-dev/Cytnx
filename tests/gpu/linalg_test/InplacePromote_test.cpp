#include "gtest/gtest.h"

#include "gpu_test_tools.h"
#include "cytnx.hpp"

// GPU in-place arithmetic promotion + non-contiguous tensor-tensor coverage (#1013).
// The per-op iadd/isub/imul tests only exercise same-dtype / non-promoting in-place
// (and reject non-contiguous tensor*=tensor / tensor/=tensor). These tests cover the
// behavior the typed in-place dispatch newly guarantees: promoting the LHS storage
// like the CPU path, and applying the layout mappers so non-contiguous tensor (op)=
// tensor works in place for every op.
namespace InplacePromoteTest {

  using namespace cytnx;

  inline void ApplyInplace(int op, Tensor& lhs, const Tensor& rhs) {
    switch (op) {
      case 0:
        linalg::iAdd(lhs, rhs);
        break;
      case 1:
        linalg::iMul(lhs, rhs);
        break;
      case 2:
        linalg::iSub(lhs, rhs);
        break;
      default:
        linalg::iDiv(lhs, rhs);
        break;
    }
  }

  inline const char* OpName(int op) {
    return op == 0 ? "iAdd" : op == 1 ? "iMul" : op == 2 ? "iSub" : "iDiv";
  }

  // GPU in-place promotes the LHS storage to the output dtype like CPU: Add/Sub/Mul
  // use type_promote, Div uses true division (make_floating_point of the promoted
  // type). Validate the GPU result dtype AND values against the CPU in-place path,
  // for the dtype pairs that PROMOTE (which the per-op tests skip). Deterministic
  // nonzero operands (arange) keep Div away from divide-by-zero.
  TEST(InplacePromoteTest, gpu_inplace_promotes_lhs_dtype) {
    for (int op = 0; op < 4; ++op) {
      for (auto ldtype : cytnx::TestTools::dtype_list) {
        if (ldtype == Type.Bool) continue;
        for (auto rdtype : cytnx::TestTools::dtype_list) {
          if (rdtype == Type.Bool) continue;
          // real (op)= complex is rejected: a complex result cannot be stored in a
          // real LHS.
          if (!Type.is_complex(ldtype) && Type.is_complex(rdtype)) continue;

          unsigned int expected = Type.type_promote(ldtype, rdtype);
          if (op == 3) expected = Type_class::make_floating_point_dtype(expected);
          if (expected == ldtype) continue;  // only the promoting cases

          SCOPED_TRACE(std::string(OpName(op)) + " ldtype=" + std::to_string(ldtype) +
                       " rdtype=" + std::to_string(rdtype));

          Tensor gpu_l = arange(1, 7, 1, ldtype).reshape({2, 3}).to(Device.cuda);
          Tensor gpu_r = arange(2, 8, 1, rdtype).reshape({2, 3}).to(Device.cuda);
          Tensor cpu_l = gpu_l.to(Device.cpu);
          Tensor cpu_r = gpu_r.to(Device.cpu);

          ApplyInplace(op, gpu_l, gpu_r);
          ApplyInplace(op, cpu_l, cpu_r);

          EXPECT_EQ(gpu_l.dtype(), expected);
          EXPECT_EQ(gpu_l.dtype(), cpu_l.dtype());
          EXPECT_TRUE(cytnx::TestTools::AreNearlyEqTensor(gpu_l.to(Device.cpu), cpu_l, 1e-5));
        }
      }
    }
  }

  // Non-contiguous tensor (op)= tensor: the typed dispatch's non-contiguous kernel
  // applies the layout mappers, so this now works in place for every op (Mul/Div
  // previously threw "not supported" because the legacy cuMul/cuDiv kernels ignored
  // the mappers, #988). Compare a permuted-LHS in-place op against the CPU path.
  TEST(InplacePromoteTest, gpu_inplace_noncontiguous_tensor_tensor) {
    for (int op = 0; op < 4; ++op) {
      for (auto dtype : cytnx::TestTools::dtype_list) {
        if (dtype == Type.Bool) continue;
        SCOPED_TRACE(std::string(OpName(op)) + " dtype=" + std::to_string(dtype));

        Tensor gpu_l = arange(1, 7, 1, dtype).reshape({2, 3}).permute({1, 0}).to(Device.cuda);
        Tensor gpu_r = arange(2, 8, 1, dtype).reshape({2, 3}).permute({1, 0}).to(Device.cuda);
        ASSERT_FALSE(gpu_l.is_contiguous());
        Tensor cpu_l = gpu_l.to(Device.cpu);
        Tensor cpu_r = gpu_r.to(Device.cpu);

        ApplyInplace(op, gpu_l, gpu_r);
        ApplyInplace(op, cpu_l, cpu_r);

        EXPECT_EQ(gpu_l.dtype(), cpu_l.dtype());
        EXPECT_TRUE(cytnx::TestTools::AreNearlyEqTensor(gpu_l.to(Device.cpu), cpu_l, 1e-5));
      }
    }
  }

  // Independent literal expected values (NOT the CPU oracle), per CLAUDE.md's test
  // guidance: pin the promoted result dtype AND element values to hand-computed
  // numbers so a bug shared by the CPU and GPU dispatch -- e.g. in the common
  // type_promote_t / true-division rule or the non-contiguous mapper math -- cannot
  // hide behind a GPU-vs-CPU comparison. Inputs are built with CPU arange (then
  // moved to the GPU) so the test data itself never depends on GPU kernels.
  TEST(InplacePromoteTest, gpu_inplace_literal_expected) {
    // Int16 += Int64 -> Int64: [1,2,3] += [10,20,30] = [11,22,33]
    {
      Tensor lhs = arange(1, 4, 1, Type.Int16).to(Device.cuda);
      Tensor rhs = arange(10, 40, 10, Type.Int64).to(Device.cuda);
      linalg::iAdd(lhs, rhs);
      EXPECT_EQ(lhs.dtype(), Type.Int64);
      Tensor got = lhs.to(Device.cpu);
      EXPECT_EQ(got.at<cytnx_int64>({0}), 11);
      EXPECT_EQ(got.at<cytnx_int64>({1}), 22);
      EXPECT_EQ(got.at<cytnx_int64>({2}), 33);
    }
    // Int64 /= Int64 -> Double (true division): [1,2,3] /= [2,2,2] = [0.5,1.0,1.5]
    {
      Tensor lhs = arange(1, 4, 1, Type.Int64).to(Device.cuda);
      Tensor rhs = zeros({3}, Type.Int64);
      rhs.fill(2);
      linalg::iDiv(lhs, rhs.to(Device.cuda));
      EXPECT_EQ(lhs.dtype(), Type.Double);
      Tensor got = lhs.to(Device.cpu);
      EXPECT_DOUBLE_EQ(got.at<cytnx_double>({0}), 0.5);
      EXPECT_DOUBLE_EQ(got.at<cytnx_double>({1}), 1.0);
      EXPECT_DOUBLE_EQ(got.at<cytnx_double>({2}), 1.5);
    }
    // Contiguous LHS + non-contiguous (permuted) RHS, so the logical elements sit at
    // DIFFERENT physical offsets in the two buffers (Lidx != Ridx): the mapper must
    // pair them correctly. a = [[1,2,3],[4,5,6]] (contiguous);
    // b logical = [[10,30,50],[20,40,60]] (arange 3x2 permuted); a += b ->
    // [[11,32,53],[24,45,66]].
    {
      Tensor a = arange(1, 7, 1, Type.Int64).reshape({2, 3}).to(Device.cuda);
      Tensor b = arange(10, 70, 10, Type.Int64).reshape({3, 2}).permute({1, 0}).to(Device.cuda);
      ASSERT_TRUE(a.is_contiguous());
      ASSERT_FALSE(b.is_contiguous());
      linalg::iAdd(a, b);
      Tensor got = a.to(Device.cpu);
      EXPECT_EQ(got.at<cytnx_int64>({0, 0}), 11);
      EXPECT_EQ(got.at<cytnx_int64>({0, 1}), 32);
      EXPECT_EQ(got.at<cytnx_int64>({0, 2}), 53);
      EXPECT_EQ(got.at<cytnx_int64>({1, 0}), 24);
      EXPECT_EQ(got.at<cytnx_int64>({1, 1}), 45);
      EXPECT_EQ(got.at<cytnx_int64>({1, 2}), 66);
    }
    // Permuted (non-contiguous) LHS - contiguous RHS, subtraction (operand order and
    // per-buffer indexing both matter). a logical = [[1,3,5],[2,4,6]] (arange 3x2
    // permuted, so the LHS output is written at permuted physical offsets);
    // b = [[0,10,20],[30,40,50]] (contiguous); a -= b -> [[1,-7,-15],[-28,-36,-44]].
    {
      Tensor a = arange(1, 7, 1, Type.Int64).reshape({3, 2}).permute({1, 0}).to(Device.cuda);
      Tensor b = arange(0, 60, 10, Type.Int64).reshape({2, 3}).to(Device.cuda);
      ASSERT_FALSE(a.is_contiguous());
      linalg::iSub(a, b);
      Tensor got = a.to(Device.cpu);
      EXPECT_EQ(got.at<cytnx_int64>({0, 0}), 1);
      EXPECT_EQ(got.at<cytnx_int64>({0, 1}), -7);
      EXPECT_EQ(got.at<cytnx_int64>({0, 2}), -15);
      EXPECT_EQ(got.at<cytnx_int64>({1, 0}), -28);
      EXPECT_EQ(got.at<cytnx_int64>({1, 1}), -36);
      EXPECT_EQ(got.at<cytnx_int64>({1, 2}), -44);
    }
  }

}  // namespace InplacePromoteTest
