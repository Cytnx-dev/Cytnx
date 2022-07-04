#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include "Tensor.hpp"
namespace cytnx {
  namespace linalg {
    Tensor Max(const Tensor &Tin) {
      cytnx_error_msg(Tin.dtype() == Type.Void, "[Cannot have void (Uninitialize) Tensor]%s", "\n");
      Tensor out({1}, Tin.dtype(), Tin.device());

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.MM_ii[out.dtype()](out._impl->storage()._impl,
                                                       Tin._impl->storage()._impl,
                                                       Tin._impl->storage()._impl->size(), 'x');
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(out.device()));
        // cytnx::linalg_internal::lii.cuMax_ii[out.dtype()](out._impl->storage()._impl,Tin._impl->storage()._impl,Tin._impl->storage()._impl->size(),'x');
        cytnx_error_msg(true, "[Max] Developing.%s", "\n");
#else
        cytnx_error_msg(true, "[Max] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }
  }  // namespace linalg
}  // namespace cytnx
