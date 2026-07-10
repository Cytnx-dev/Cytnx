#include "linalg.hpp"
#include "utils/utils.hpp"
#include "Tensor.hpp"

#ifdef BACKEND_TORCH
#else
  #include "backend/linalg_internal_interface.hpp"
namespace cytnx {

  namespace linalg {
    Tensor Vectordot(const Tensor &Tl, const Tensor &Tr, const bool &is_conj) {
      // checking:
      cytnx_error_msg(Tl.device() != Tr.device(),
                      "[ERROR] Two tensors for Vectordot cannot be on different devices.%s", "\n");
      cytnx_error_msg(Tl.shape().size() != 1,
                      "[ERROR][Tl] Tensor for Vectordot should be rank-1.%s", "\n");
      cytnx_error_msg(Tr.shape().size() != 1,
                      "[ERROR][Tr] Tensor for Vectordot should be rank-1.%s", "\n");
      cytnx_error_msg(Tr.shape()[0] != Tl.shape()[0],
                      "[ERROR] Two tensors for Vectordot should have the same length.%s", "\n");

      // Promote to a common dtype via the centralized rule so the output dtype agrees
      // with the elementwise ops (fixes mixed complex/real pairs, e.g. ComplexFloat
      // x Double -> ComplexDouble instead of the old lower-enum-index ComplexFloat).
      const unsigned int out_dtype = Type.type_promote(Tl.dtype(), Tr.dtype());
      // astype is a no-op (ref, no copy) when the dtype already matches - don't modify L/R!
      Tensor L = Tl.astype(out_dtype);
      Tensor R = Tr.astype(out_dtype);
      Tensor out;
      out.Init({1}, out_dtype, Tl.device());

      if (out.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Vd_ii[out.dtype()](
          out._impl->storage()._impl, L._impl->storage()._impl, R._impl->storage()._impl,
          L._impl->storage()._impl->size(), is_conj);
        return out;
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tl.device()));
        cytnx::linalg_internal::lii.cuVd_ii[out.dtype()](
          out._impl->storage()._impl, L._impl->storage()._impl, R._impl->storage()._impl,
          L._impl->storage()._impl->size(), is_conj);

        return out;
  #else
        cytnx_error_msg(true, "[Vectordot] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
        return Tensor();
  #endif
      }
    }

  }  // namespace linalg
}  // namespace cytnx
#endif  // BACKEND_TORCH
