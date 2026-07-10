#include "linalg.hpp"
#include <cmath>
#include <iostream>
#include "Tensor.hpp"
#include "cytnx.hpp"
#ifdef BACKEND_TORCH
#else
  #include "backend/linalg_internal_interface.hpp"

namespace cytnx {
  namespace linalg {
    Tensor Norm(const Tensor& Tl) {
      // cytnx_error_msg(Tl.shape().size() != 1,"[Norm] error, tensor Tl ,Norm can only operate on
      // rank-1 Tensor.%s","\n"); cytnx_error_msg(!Tl.is_contiguous(), "[Norm] error tensor Tl must
      // be contiguous. Call Contiguous_() or Contiguous() first%s","\n");

      // check type:
      Tensor _tl;
      Tensor out;

      if (Tl.dtype() > 4) {
        // do conversion:
        _tl = Tl.astype(Type.Double);

      } else {
        _tl = Tl;
      }

      if (Tl.dtype() == Type.ComplexDouble) {
        out.Init({}, Type.Double, _tl.device());
      } else if (Tl.dtype() == Type.ComplexFloat) {
        out.Init({}, Type.Float, _tl.device());
      } else {
        out.Init({}, _tl.dtype(), _tl.device());
      }

      if (Tl.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Norm_ii[_tl.dtype()](out._impl->storage()._impl->data(),
                                                         _tl._impl->storage()._impl);

        return out;

      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tl.device()));
        cytnx::linalg_internal::lii.cuNorm_ii[_tl.dtype()](out._impl->storage()._impl->data(),
                                                           _tl._impl->storage()._impl);

        return out;
  #else
        cytnx_error_msg(true, "[Norm] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
        return Tensor();
  #endif
      }
    }

    Tensor Norm(const UniTensor& uTl) { return uTl.Norm(); }

  }  // namespace linalg
}  // namespace cytnx
#endif  // BACKEND_TORCH
