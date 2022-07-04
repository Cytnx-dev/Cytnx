#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include "Tensor.hpp"

namespace cytnx {
  namespace linalg {
    void InvM_(Tensor &Tin) {
      cytnx_error_msg(Tin.shape().size() != 2,
                      "[InvM] error, InvM can only operate on rank-2 Tensor.%s", "\n");
      // cytnx_error_msg(!Tin.is_contiguous(), "[InvM] error tensor must be contiguous. Call
      // Contiguous_() or Contiguous() first%s","\n");

      cytnx_error_msg(Tin.shape()[0] != Tin.shape()[1],
                      "[InvM] error, the size of last two rank should be the same.%s", "\n");

      if (Tin.dtype() > 4) Tin = Tin.contiguous().astype(Type.Double);

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.InvM_inplace_ii[Tin.dtype()](Tin._impl->storage()._impl,
                                                                 Tin.shape().back());

      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tin.device()));
        cytnx::linalg_internal::lii.cuInvM_inplace_ii[Tin.dtype()](Tin._impl->storage()._impl,
                                                                   Tin.shape().back());

#else
        cytnx_error_msg(true, "[InvM] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
#endif
      }
    }

  }  // namespace linalg
}  // namespace cytnx
