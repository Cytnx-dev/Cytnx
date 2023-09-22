#include "linalg.hpp"

#include "Tensor.hpp"

#ifdef BACKEND_TORCH
#else
  #include "../backend/linalg_internal_interface.hpp"

namespace cytnx {
  namespace linalg {
    Tensor Min(const Tensor &Tin) {
      cytnx_error_msg(Tin.dtype() == Type.Void, "[Cannot have void (Uninitialize) Tensor]%s", "\n");
      Tensor out({1}, Tin.dtype(), Tin.device());

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.MM_ii[out.dtype()](out._impl->storage()._impl,
                                                       Tin._impl->storage()._impl,
                                                       Tin._impl->storage()._impl->size(), 'n');
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(out.device()));
        cytnx::linalg_internal::lii.cuMM_ii[out.dtype()](out._impl->storage()._impl,
                                                         Tin._impl->storage()._impl,
                                                         Tin._impl->storage()._impl->size(), 'n');
          // cytnx_error_msg(true, "[Min] Developing.%s", "\n");
  #else
        cytnx_error_msg(true, "[Min] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }
  }  // namespace linalg
}  // namespace cytnx
#endif  // BACKEND_TORCH
