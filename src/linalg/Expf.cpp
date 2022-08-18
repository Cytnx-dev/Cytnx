#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include "Tensor.hpp"
namespace cytnx {
  namespace linalg {
    Tensor Expf(const Tensor &Tin) {
      Tensor out;
      if ((Tin.dtype() == Type.ComplexFloat) || (Tin.dtype() == Type.Float))
        out = Tin.clone();
      else if (Tin.dtype() == Type.ComplexDouble)
        out = Tin.astype(Type.ComplexFloat);
      else if (Tin.dtype() > 4)
        out = Tin.astype(Type.Float);
      else if (Tin.dtype() == Type.Double)
        out = Tin.astype(Type.Float);
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
        cytnx_error_msg(true, "[Expf] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }
  }  // namespace linalg
}  // namespace cytnx
