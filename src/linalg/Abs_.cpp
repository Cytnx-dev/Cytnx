#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include "Tensor.hpp"

namespace cytnx {
  namespace linalg {
    void Abs_(Tensor &Tin) {
      // Tensor out;

      // if the type is unsigned, clone and return.
      if (!Type.is_unsigned(Tin.dtype())) {
        Tensor out;
        if (Tin.dtype() == Type.ComplexDouble || Tin.dtype() == Type.ComplexFloat) {
          out._impl = Tin._impl->_clone_meta_only();
          if (Tin.dtype() == Type.ComplexDouble)
            out.storage() = Storage(Tin.storage().size(), Type.Double, Tin.device());
          else
            out.storage() = Storage(Tin.storage().size(), Type.Float, Tin.device());

        } else {
          out = Tin;
        }

        if (Tin.device() == Device.cpu) {
          cytnx::linalg_internal::lii.Abs_ii[Tin.dtype()](out._impl->storage()._impl,
                                                          Tin._impl->storage()._impl,
                                                          out._impl->storage()._impl->size());
        } else {
#ifdef UNI_GPU
          // checkCudaErrors(cudaSetDevice(out.device()));
          // cytnx::linalg_internal::lii.cuAbs__ii[Tin.dtype()](out._impl->storage()._impl,Tin._impl->storage()._impl,Tin._impl->storage()._impl->size(),p);
          cytnx_error_msg(true, "[Abs_][GPU] developing%s", "\n");
#else
          cytnx_error_msg(true, "[Abs_] fatal error, the tensor is on GPU without CUDA support.%s",
                          "\n");
#endif
        }
        if (Tin.dtype() == Type.ComplexDouble || Tin.dtype() == Type.ComplexFloat) Tin = out;
      }
    }
  }  // namespace linalg
}  // namespace cytnx
