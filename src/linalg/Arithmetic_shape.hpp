#ifndef CYTNX_LINALG_ARITHMETIC_SHAPE_HPP_
#define CYTNX_LINALG_ARITHMETIC_SHAPE_HPP_

#include "Tensor.hpp"

namespace cytnx {
  namespace linalg {
    namespace detail {

      inline void check_tensor_initialized(const Tensor &tensor, const char *op_name) {
        cytnx_error_msg(tensor.is_void(),
                        "[%s] cannot perform arithmetic on an uninitialized Tensor.%s", op_name,
                        "\n");
      }

      inline void check_binary_tensor_inputs(const Tensor &Lt, const Tensor &Rt,
                                             const char *op_name) {
        check_tensor_initialized(Lt, op_name);
        check_tensor_initialized(Rt, op_name);
      }

      inline bool is_singleton_tensor(const Tensor &tensor) { return tensor.size() == 1; }

      inline bool init_broadcast_binary_output(Tensor &out, const Tensor &Lt, const Tensor &Rt,
                                               const unsigned int dtype,
                                               const bool init_zero = true) {
        check_binary_tensor_inputs(Lt, Rt, "Tensor arithmetic");
        const bool lhs_is_singleton = is_singleton_tensor(Lt);
        const bool rhs_is_singleton = is_singleton_tensor(Rt);
        if (!lhs_is_singleton && !rhs_is_singleton) return false;

        const Tensor *meta = &Lt;
        if (lhs_is_singleton && (!rhs_is_singleton || Rt.rank() > Lt.rank())) {
          meta = &Rt;
        }
        out._impl = meta->_impl->_clone_meta_only();
        out._impl->storage() = Storage(meta->storage().size(), dtype, meta->device(), init_zero);
        return true;
      }

      inline Tensor host_singleton_for_gpu_broadcast(const Tensor &tensor, const int op_device) {
        if (op_device != Device.cpu && is_singleton_tensor(tensor) &&
            tensor.device() != Device.cpu) {
          return tensor.to(Device.cpu);
        }
        return tensor;
      }

    }  // namespace detail
  }  // namespace linalg
}  // namespace cytnx

#endif  // CYTNX_LINALG_ARITHMETIC_SHAPE_HPP_
