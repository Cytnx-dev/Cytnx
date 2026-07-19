#include "linalg.hpp"
#include "Tensor.hpp"

#ifdef BACKEND_TORCH
#else
  #include "backend/linalg_internal_interface.hpp"
namespace cytnx {
  namespace linalg {
    Tensor Exp(const Tensor &Tin) {
      Tensor out;
      // dtype-preserving: a floating/complex input keeps its own precision (Float stays Float,
      // ComplexFloat stays ComplexFloat), so Exp subsumes the former Expf(); only integer/bool
      // inputs, which have no floating exp kernel, promote to Double.
      if ((Tin.dtype() == Type.ComplexDouble) || (Tin.dtype() == Type.Double) ||
          (Tin.dtype() == Type.ComplexFloat) || (Tin.dtype() == Type.Float))
        out = Tin.clone();
      else if (Tin.dtype() > 4)
        out = Tin.astype(Type.Double);
      else
        cytnx_error_msg(true, "[Cannot have void (Uninitialize) Tensor]%s", "\n");

      if (out.is_empty()) return out;

      // `out` already holds Tin cast to the dispatch dtype, so feed it as the kernel input too:
      // dispatching Exp_ii[out.dtype()] with Tin's original storage would reinterpret e.g. a
      // float buffer as double*. The kernels support in == out (see Exp_.cpp).
      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Exp_ii[out.dtype()](out._impl->storage()._impl,
                                                        out._impl->storage()._impl,
                                                        out._impl->storage()._impl->size());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(out.device()));
        cytnx::linalg_internal::cuExp_dispatch(out._impl->storage()._impl,
                                               out._impl->storage()._impl,
                                               out._impl->storage()._impl->size());
  #else
        cytnx_error_msg(true, "[Exp] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }
  }  // namespace linalg
}  // namespace cytnx
#endif  // BACKEND_TORCH
