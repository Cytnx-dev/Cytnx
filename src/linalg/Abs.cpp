#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include "Tensor.hpp"

namespace cytnx {
  namespace linalg {
    Tensor Abs(const Tensor &Tin) {
      Tensor out;

      // if the type is unsigned, clone and return.
      if (Type.is_unsigned(Tin.dtype())) {
        out = Tin.clone();
        return out;
      }

      out._impl = Tin._impl->_clone_meta_only();
      if (Tin.dtype() == Type.ComplexDouble)
        out.storage() = Storage(Tin.storage().size(), Type.Double, Tin.device());
      else if (Tin.dtype() == Type.ComplexFloat)
        out.storage() = Storage(Tin.storage().size(), Type.Float, Tin.device());
      else
        out.storage() = Storage(Tin.storage().size(), Tin.dtype(), Tin.device());

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Abs_ii[Tin.dtype()](out._impl->storage()._impl,
                                                        Tin._impl->storage()._impl,
                                                        out._impl->storage()._impl->size());
      } else {
#ifdef UNI_GPU
        // checkCudaErrors(cudaSetDevice(out.device()));
        // cytnx::linalg_internal::lii.cuAbs_ii[Tin.dtype()](out._impl->storage()._impl,Tin._impl->storage()._impl,Tin._impl->storage()._impl->size(),p);
        cytnx_error_msg(true, "[Abs][GPU] developing%s", "\n");
#else
        cytnx_error_msg(true, "[Abs] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }
  }  // namespace linalg
}  // namespace cytnx
