#include "linalg.hpp"
#include "Tensor.hpp"

#ifdef BACKEND_TORCH
#else

  #include "backend/linalg_internal_interface.hpp"
namespace cytnx {
  namespace linalg {
    Tensor Abs(const Tensor& Tin) {
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
        std::visit(
          [&](auto out_ptr) {
            using out_type = std::remove_pointer_t<decltype(out_ptr)>;
            static_assert(!std::is_same_v<out_type, void>);

            std::visit(
              [&](auto in_ptr) {
                using in_type = std::remove_pointer_t<decltype(in_ptr)>;
                static_assert(!std::is_same_v<in_type, void>);

                cytnx::linalg_internal::AbsInternalImpl<in_type, out_type>(
                  out._impl->storage()._impl, Tin._impl->storage()._impl,
                  out._impl->storage()._impl->size());
              },
              Tin.ptr());
          },
          out.ptr());
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(out.device()));
        cytnx::linalg_internal::lii.cuAbs_ii[Tin.dtype()](out._impl->storage()._impl,
                                                          Tin._impl->storage()._impl,
                                                          Tin._impl->storage()._impl->size());
          // cytnx_error_msg(true, "[Abs][GPU] developing%s", "\n");
  #else
        cytnx_error_msg(true, "[Abs] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }
  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
