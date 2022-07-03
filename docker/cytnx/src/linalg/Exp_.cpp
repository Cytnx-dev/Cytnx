#include "linalg/linalg.hpp"

namespace cytnx {
  namespace linalg {
    void Exp_(Tensor &Tin) {
      if (Tin.dtype() > 4) Tin = Tin.astype(Type.Float);

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Exp_ii[Tin.dtype()](Tin._impl->storage()._impl,
                                                        Tin._impl->storage()._impl,
                                                        Tin._impl->storage()._impl->size());
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tin.device()));
        cytnx::linalg_internal::lii.cuExp_ii[Tin.dtype()](Tin._impl->storage()._impl,
                                                          Tin._impl->storage()._impl,
                                                          Tin._impl->storage()._impl->size());
#else
        cytnx_error_msg(true, "[Exp_] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }
    }
  }  // namespace linalg
}  // namespace cytnx
