#include "linalg.hpp"
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
        out.Init({1}, Type.Double, _tl.device());
      } else if (Tl.dtype() == Type.ComplexFloat) {
        out.Init({1}, Type.Float, _tl.device());
      } else {
        out.Init({1}, _tl.dtype(), _tl.device());
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

    static Tensor norm_impl(const UniTensor& uTl) {
      if (uTl.uten_type() == UTenType.Dense) {
        return norm_impl(uTl.get_block_());
      } else if ((uTl.uten_type() == UTenType.Block) ||
                 (uTl.uten_type() == UTenType.BlockFermionic)) {
        std::vector<Tensor> bks = uTl.get_blocks_();
        Tensor res = zeros(1);
        for (int i = 0; i < bks.size(); i++) {
          Tensor tmp = norm_impl(bks[i]);
          res.at({0}) = res.at({0}) + tmp.at({0}) * tmp.at({0});
        }
        res.at({0}) = sqrt(res.at({0}));
        return res;
      } else {
        cytnx_error_msg(true, "[ERROR][Norm] UniTensor type '%s' not supported\n",
                        uTl.uten_type_str().c_str());
        return Tensor();
      }
    }

    Tensor Norm(const Tensor& Tl) { return norm_impl(Tl); }

    // norm() returns the 2-norm as a Scalar carrying the tensor's own precision (Float for
    // Float/ComplexFloat input, Double otherwise) rather than a fixed double. This lets
    // x /= x.norm() stay dtype-preserving via linalg::Div's Tensor/Scalar path instead of
    // silently promoting a Float tensor to Double (#1000 review, ianmccul).
    Scalar norm(const Tensor& Tl) { return norm_impl(Tl).item(); }

    Tensor Norm(const UniTensor& uTl) { return norm_impl(uTl); }

    Scalar norm(const UniTensor& uTl) { return norm_impl(uTl).item(); }

  }  // namespace linalg
}  // namespace cytnx
#endif  // BACKEND_TORCH
