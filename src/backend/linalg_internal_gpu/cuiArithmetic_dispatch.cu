#include "cuiArithmetic_dispatch.hpp"

#include "cuiArithmeticDispatch.cuh"

namespace cytnx {

  namespace linalg_internal {

    void cuiArithmeticDispatch(int op_code, Tensor &Lt, const Tensor &Rt, bool rhs_is_weak_scalar,
                               const std::vector<cytnx_uint64> &shape,
                               const std::vector<cytnx_uint64> &invmapper_L,
                               const std::vector<cytnx_uint64> &invmapper_R) {
      switch (op_code) {
        case 0:
          DispatchInplaceArithmeticGPU<0>(Lt, Rt, rhs_is_weak_scalar, shape, invmapper_L,
                                          invmapper_R);
          break;
        case 1:
          DispatchInplaceArithmeticGPU<1>(Lt, Rt, rhs_is_weak_scalar, shape, invmapper_L,
                                          invmapper_R);
          break;
        case 2:
          DispatchInplaceArithmeticGPU<2>(Lt, Rt, rhs_is_weak_scalar, shape, invmapper_L,
                                          invmapper_R);
          break;
        case 3:
          DispatchInplaceArithmeticGPU<3>(Lt, Rt, rhs_is_weak_scalar, shape, invmapper_L,
                                          invmapper_R);
          break;
        default:
          cytnx_error_msg(true, "[cuiArithmeticDispatch] invalid op_code %d%s", op_code, "\n");
      }
    }

  }  // namespace linalg_internal
}  // namespace cytnx
