#ifndef CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUIARITHMETIC_DISPATCH_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUIARITHMETIC_DISPATCH_H_

#include <vector>

#include "Type.hpp"
#include "Tensor.hpp"

namespace cytnx {
  namespace linalg_internal {

    // In-place typed GPU dispatch for elementwise binary arithmetic. op_code:
    // 0=Add, 1=Mul, 2=Sub, 3=Div (true division). Promotes Lt's storage to the
    // output dtype (matching the CPU DispatchInplaceArithmeticCPU) and runs the op
    // in place; see cuiArithmeticDispatch.cuh. Defined only for CUDA builds.
    void cuiArithmeticDispatch(int op_code, Tensor &Lt, const Tensor &Rt, bool rhs_is_weak_scalar,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R);

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUIARITHMETIC_DISPATCH_H_
