#ifndef CYTNX_LINALG_ARITHMETIC_SHAPE_HPP_
#define CYTNX_LINALG_ARITHMETIC_SHAPE_HPP_

#include "Tensor.hpp"

namespace cytnx {
  namespace linalg {
    namespace detail {

      inline bool InitBroadcastBinaryOutput(Tensor &out, const Tensor &Lt, const Tensor &Rt,
                                            const unsigned int dtype, const bool init_zero = true) {
        if (!Lt.is_scalar() && !Rt.is_scalar()) return false;

        const Tensor &meta = Lt.is_scalar() ? Rt : Lt;
        out._impl = meta._impl->_clone_meta_only();
        out._impl->storage() = Storage(meta.storage().size(), dtype, meta.device(), init_zero);
        return true;
      }

      inline Tensor HostScalarForGpuBroadcast(const Tensor &tensor, const int op_device) {
        if (op_device != Device.cpu && tensor.is_scalar() && tensor.device() != Device.cpu) {
          return tensor.to(Device.cpu);
        }
        return tensor;
      }

    }  // namespace detail
  }  // namespace linalg
}  // namespace cytnx

#endif  // CYTNX_LINALG_ARITHMETIC_SHAPE_HPP_
