#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include "Tensor.hpp"
namespace cytnx {
  namespace linalg {
    Tensor InvM(const Tensor &Tin) {
      cytnx_error_msg(Tin.shape().size() != 2,
                      "[InvM] error, InvM can only operate on rank-2 Tensor.%s", "\n");
      // cytnx_error_msg(!Tin.is_contiguous(), "[InvM] error tensor must be contiguous. Call
      // Contiguous_() or Contiguous() first%s","\n");

      cytnx_error_msg(Tin.shape()[0] != Tin.shape()[1],
                      "[InvM] error, the size of last two rank should be the same.%s", "\n");

      Tensor out;
      if (!Tin.is_contiguous())
        out = Tin.contiguous();
      else
        out = Tin.clone();

      if (Tin.dtype() > 4) out = out.astype(Type.Double);

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.InvM_inplace_ii[out.dtype()](out._impl->storage()._impl,
                                                                 out.shape().back());

        return out;

      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(out.device()));
        cytnx::linalg_internal::lii.cuInvM_inplace_ii[out.dtype()](out._impl->storage()._impl,
                                                                   out.shape().back());
        return out;
#else
        cytnx_error_msg(true, "[InvM] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
        return Tensor();
#endif
      }
    }

  }  // namespace linalg
}  // namespace cytnx
