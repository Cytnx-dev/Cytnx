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
      cytnx_error_msg(Tl.is_void(), "[Norm] cannot operate on an uninitialized Tensor.%s", "\n");
      // cytnx_error_msg(Tl.shape().size() != 1,"[Norm] error, tensor Tl ,Norm can only operate on
      // rank-1 Tensor.%s","\n"); cytnx_error_msg(!Tl.is_contiguous(), "[Norm] error tensor Tl must
      // be contiguous. Call Contiguous_() or Contiguous() first%s","\n");

      // check type: floating/complex inputs keep their precision; anything
      // else (integer/bool) is computed in double precision. The output
      // dtype policy lives in Type_class::norm_result_dtype, shared with
      // callers that pre-size norm accumulators (e.g. UniTensor's
      // normalize_()) so the two cannot drift apart.
      Tensor _tl;
      Tensor out;

      if (Type.is_float(Tl.dtype())) {
        _tl = Tl;
      } else {
        // do conversion:
        _tl = Tl.astype(Type.Double);
      }

      out.Init({}, Type_class::norm_result_dtype(Tl.dtype()), _tl.device());

      if (Tl.is_empty()) return out;

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
