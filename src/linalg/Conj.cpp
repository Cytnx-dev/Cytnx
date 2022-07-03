#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include <iostream>
#include "Tensor.hpp"

namespace cytnx {
  namespace linalg {
    Tensor Conj(const Tensor &Tin) {
      // cytnx_error_msg(Tin.shape().size() != 2,"[Inv] error, Inv can only operate on rank-2
      // Tensor.%s","\n"); cytnx_error_msg(!Tin.is_contiguous(), "[Inv] error tensor must be
      // contiguous. Call Contiguous_() or Contiguous() first%s","\n");

      // cytnx_error_msg(Tin.shape()[0] != Tin.shape()[1], "[Inv] error, the size of last two rank
      // should be the same.%s","\n");

      Tensor out;
      out = Tin.clone();

      if (Tin.device() == Device.cpu) {
        if (out.dtype() < 3)
          cytnx::linalg_internal::lii.Conj_inplace_ii[out.dtype()](out._impl->storage()._impl,
                                                                   out._impl->storage().size());

        return out;

      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tin.device()));
        if (out.dtype() < 3)
          cytnx::linalg_internal::lii.cuConj_inplace_ii[out.dtype()](out._impl->storage()._impl,
                                                                     out._impl->storage().size());
        return out;
#else
        cytnx_error_msg(true, "[Inv] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
        return Tensor();
#endif
      }
    }

  }  // namespace linalg
}  // namespace cytnx
