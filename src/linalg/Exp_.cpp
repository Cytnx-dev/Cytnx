#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include "Tensor.hpp"

namespace cytnx {
  namespace linalg {
    void Exp_(Tensor &Tin) {
      if ((Tin.dtype() == Type.ComplexDouble) || (Tin.dtype() == Type.Double)) {
        ;
      } else if (Tin.dtype() == Type.ComplexFloat)
        Tin = Tin.astype(Type.ComplexDouble);
      else if (Tin.dtype() > 4)
        Tin = Tin.astype(Type.Double);
      else if (Tin.dtype() == Type.Float)
        Tin = Tin.astype(Type.Double);
      else
        cytnx_error_msg(true, "[Cannot have void (Uninitialize) Tensor]%s", "\n");

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
