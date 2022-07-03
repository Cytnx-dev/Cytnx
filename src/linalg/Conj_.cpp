#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include "Tensor.hpp"

#include <iostream>

namespace cytnx {
  namespace linalg {
    void Conj_(Tensor &Tin) {
      // cytnx_error_msg(Tin.shape().size() != 2,"[Inv] error, Inv can only operate on rank-2
      // Tensor.%s","\n"); cytnx_error_msg(!Tin.is_contiguous(), "[Inv] error tensor must be
      // contiguous. Call Contiguous_() or Contiguous() first%s","\n");

      // cytnx_error_msg(Tin.shape()[0] != Tin.shape()[1], "[Inv] error, the size of last two rank
      // should be the same.%s","\n");

      if (Tin.device() == Device.cpu) {
        if (Tin.dtype() < 3)
          cytnx::linalg_internal::lii.Conj_inplace_ii[Tin.dtype()](Tin._impl->storage()._impl,
                                                                   Tin._impl->storage().size());

      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tin.device()));
        if (Tin.dtype() < 3)
          cytnx::linalg_internal::lii.cuConj_inplace_ii[Tin.dtype()](Tin._impl->storage()._impl,
                                                                     Tin._impl->storage().size());

#else
        cytnx_error_msg(true, "[Inv] fatal error,%s",
                        "try to call the gpu section withTin CUDA support.\n");
#endif
      }
    }

  }  // namespace linalg
}  // namespace cytnx
