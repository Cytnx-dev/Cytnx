#include "linalg/linalg.hpp"

namespace cytnx {
  namespace linalg {
    Tensor Exp(const Tensor &Tin) {
      Tensor out;
      if (Tin.dtype() > 4)
        out = Tin.astype(Type.Float);
      else
        out = Tin.clone();

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Exp_ii[out.dtype()](out._impl->storage()._impl,
                                                        Tin._impl->storage()._impl,
                                                        Tin._impl->storage()._impl->size());
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(out.device()));
        cytnx::linalg_internal::lii.cuExp_ii[out.dtype()](out._impl->storage()._impl,
                                                          Tin._impl->storage()._impl,
                                                          Tin._impl->storage()._impl->size());
#else
        cytnx_error_msg(true, "[Exp] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }
  }  // namespace linalg
}  // namespace cytnx
