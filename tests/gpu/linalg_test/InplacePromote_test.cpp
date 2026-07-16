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

}  // namespace InplacePromoteTest
