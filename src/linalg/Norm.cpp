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
    // NOTE: Norm() below is [[deprecated]]. To avoid deprecation warnings from
    // internal callers, the actual computation lives in norm_impl(); both
    // Norm() and norm() delegate to it.
    static Tensor norm_impl(const Tensor& Tl) {
      cytnx_error_msg(Tl.is_void(), "[Norm] cannot operate on an uninitialized Tensor.%s", "\n");
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

      // the norm is a scalar: rank-0 output, carrying the tensor's own precision (Float for
      // Float/ComplexFloat input, Double otherwise).
      if (Tl.dtype() == Type.ComplexDouble) {
        out.Init({}, Type.Double, _tl.device());
      } else if (Tl.dtype() == Type.ComplexFloat) {
        out.Init({}, Type.Float, _tl.device());
      } else {
        out.Init({}, _tl.dtype(), _tl.device());
      }

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

    Tensor Norm(const Tensor& Tl) { return norm_impl(Tl); }

    // norm() returns the 2-norm as a Scalar carrying the tensor's own precision (Float for
    // Float/ComplexFloat input, Double otherwise) rather than a fixed double. This lets
    // x /= x.norm() stay dtype-preserving via linalg::Div's Tensor/Scalar path instead of
    // silently promoting a Float tensor to Double (#1000 review, ianmccul).
    Scalar norm(const Tensor& Tl) { return norm_impl(Tl).item(); }

    // The UniTensor overloads delegate to the per-subclass UniTensor::Norm()/norm(), which
    // aggregate the block norms. norm() uses the non-deprecated member; the deprecated Norm()
    // free function forwards to the deprecated member (suppressed locally, since both are being
    // removed together for one release).
  #if defined(__GNUC__) || defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  #endif
    Tensor Norm(const UniTensor& uTl) { return uTl.Norm(); }
  #if defined(__GNUC__) || defined(__clang__)
    #pragma GCC diagnostic pop
  #endif

    Scalar norm(const UniTensor& uTl) { return uTl.norm(); }

  }  // namespace linalg
}  // namespace cytnx
#endif  // BACKEND_TORCH
