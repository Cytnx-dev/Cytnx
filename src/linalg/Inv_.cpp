#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include "Tensor.hpp"

namespace cytnx {
  namespace linalg {
    void Inv_(Tensor &Tin, const double &clip) {
      if (Tin.dtype() == Type.Void) {
        cytnx_error_msg(true, "[ERROR][Inv_] Cannot operate on un-initialize Tensor.%s", "\n");
      } else if (Tin.dtype() > 4) {
        Tin = Tin.astype(Type.Double);
      }

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Inv_inplace_ii[Tin.dtype()](
          Tin._impl->storage()._impl, Tin._impl->storage()._impl->size(), clip);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tin.device()));
        cytnx::linalg_internal::lii.cuInv_inplace_ii[Tin.dtype()](
          Tin._impl->storage()._impl, Tin._impl->storage()._impl->size(), clip);
#else
        cytnx_error_msg(true, "[Inv_] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }
    }
  }  // namespace linalg
}  // namespace cytnx
