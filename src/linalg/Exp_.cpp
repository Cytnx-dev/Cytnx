#include "linalg.hpp"
#include "Tensor.hpp"

#ifdef BACKEND_TORCH
#else
  #include "backend/linalg_internal_interface.hpp"
namespace cytnx {
  namespace linalg {
    void Exp_(Tensor &Tin) {
      // dtype-preserving: a floating/complex input keeps its own precision in place (Float stays
      // Float, ComplexFloat stays ComplexFloat), so Exp_ subsumes the former Expf_(); only
      // integer/bool inputs, which have no floating exp kernel, promote to Double.
      if ((Tin.dtype() == Type.ComplexDouble) || (Tin.dtype() == Type.Double) ||
          (Tin.dtype() == Type.ComplexFloat) || (Tin.dtype() == Type.Float)) {
        ;
      } else if (Tin.dtype() > 4)
        Tin = Tin.astype(Type.Double);
      else
        cytnx_error_msg(true, "[Cannot have void (Uninitialize) Tensor]%s", "\n");

      if (Tin.is_empty()) return;

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Exp_ii[Tin.dtype()](Tin._impl->storage()._impl,
                                                        Tin._impl->storage()._impl,
                                                        Tin._impl->storage()._impl->size());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tin.device()));
        cytnx::linalg_internal::cuExp_dispatch(Tin._impl->storage()._impl,
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
#endif  // BACKEND_TORCH
