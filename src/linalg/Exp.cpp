#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include "Tensor.hpp"
namespace cytnx {
  namespace linalg {
    Tensor Exp(const Tensor &Tin) {
      Tensor out;
      if ((Tin.dtype() == Type.ComplexDouble) || (Tin.dtype() == Type.Double))
        out = Tin.clone();
      else if (Tin.dtype() == Type.ComplexFloat)
        out = Tin.astype(Type.ComplexDouble);
      else if (Tin.dtype() > 4)
        out = Tin.astype(Type.Double);
      else if (Tin.dtype() == Type.Float)
        out = Tin.astype(Type.Double);
      else
        cytnx_error_msg(true, "[Cannot have void (Uninitialize) Tensor]%s", "\n");

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
