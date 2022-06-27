#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include "Tensor.hpp"
namespace cytnx {
  namespace linalg {
    Tensor Inv(const Tensor &Tin, const double &clip) {
      Tensor out;
      if (Tin.dtype() == Type.Void) {
        cytnx_error_msg(true, "[ERROR][Inv] Cannot operate on un-initialize Tensor.%s", "\n");
      } else if (Tin.dtype() > 4) {
        out = Tin.astype(Type.Double);
      } else {
        out = Tin.clone();
      }

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Inv_inplace_ii[out.dtype()](
          out._impl->storage()._impl, out._impl->storage()._impl->size(), clip);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tin.device()));
        cytnx::linalg_internal::lii.cuInv_inplace_ii[out.dtype()](
          out._impl->storage()._impl, out._impl->storage()._impl->size(), clip);
#else
        cytnx_error_msg(true, "[Inv] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }
      return out;
    }
  }  // namespace linalg
}  // namespace cytnx
