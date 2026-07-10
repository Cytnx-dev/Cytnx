#ifndef CYTNX_LINALG_ARITHMETIC_SHAPE_HPP_
#define CYTNX_LINALG_ARITHMETIC_SHAPE_HPP_

#include "Tensor.hpp"

namespace cytnx {
  namespace linalg {
    namespace detail {

      inline void check_binary_tensor_inputs(const Tensor &Lt, const Tensor &Rt,
                                             const char *op_name) {
        cytnx_error_msg(Lt.is_void() || Rt.is_void(),
                        "[%s] cannot perform arithmetic on an uninitialized Tensor.%s", op_name,
                        "\n");
      }

      inline bool init_broadcast_binary_output(Tensor &out, const Tensor &Lt, const Tensor &Rt,
                                               const unsigned int dtype,
                                               const bool init_zero = true) {
        check_binary_tensor_inputs(Lt, Rt, "Tensor arithmetic");
        if (!Lt.is_scalar() && !Rt.is_scalar()) return false;

        const Tensor &meta = Lt.is_scalar() ? Rt : Lt;
        out._impl = meta._impl->_clone_meta_only();
        out._impl->storage() = Storage(meta.storage().size(), dtype, meta.device(), init_zero);
        return true;
      }

      inline bool is_gpu_scalar_broadcast(const Tensor &Lt, const Tensor &Rt) {
        return Lt.device() != Device.cpu && (Lt.is_scalar() != Rt.is_scalar());
      }

      inline bool needs_gpu_size_one_dispatch_fallback(const Tensor &Lt, const Tensor &Rt) {
        if (!is_gpu_scalar_broadcast(Lt, Rt)) return false;
        // The legacy CUDA binary dispatchers infer scalar operands from Storage::size()==1.
        // For rank-0 scalar op rank-1 shape {1}, that loses the rank distinction and can
        // dereference a device pointer on the host. Route this tiny case through the CPU until
        // the CUDA dispatch API accepts explicit scalar flags.
        const Tensor &nonscalar = Lt.is_scalar() ? Rt : Lt;
        return nonscalar.storage().size() == 1;
      }

      inline Tensor host_scalar_for_gpu_broadcast(const Tensor &tensor, const int op_device) {
        if (op_device != Device.cpu && tensor.is_scalar() && tensor.device() != Device.cpu) {
          return tensor.to(Device.cpu);
        }
        return tensor;
      }

    }  // namespace detail
  }  // namespace linalg
}  // namespace cytnx

#endif  // CYTNX_LINALG_ARITHMETIC_SHAPE_HPP_
