#include "linalg.hpp"
#include <iostream>
#include "Tensor.hpp"
#include "cytnx.hpp"
#ifdef BACKEND_TORCH
#else
  #include "../backend/linalg_internal_interface.hpp"

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
        out.Init({1}, Type.Double, _tl.device());
      } else if (Tl.dtype() == Type.ComplexFloat) {
        out.Init({1}, Type.Float, _tl.device());
      } else {
        out.Init({1}, _tl.dtype(), _tl.device());
      }

      if (Tl.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Norm_ii[_tl.dtype()](out._impl->storage()._impl->Mem,
                                                         _tl._impl->storage()._impl);

        return out;

      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tl.device()));
        cytnx::linalg_internal::lii.cuNorm_ii[_tl.dtype()](out._impl->storage()._impl->Mem,
                                                           _tl._impl->storage()._impl);

        return out;
  #else
        cytnx_error_msg(true, "[Norm] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
        return Tensor();
  #endif
      }
    }

    Tensor Norm(const UniTensor& uTl) {
      if (uTl.uten_type() == UTenType.Dense) {
        return Norm(uTl.get_block_());
      } else if (uTl.uten_type() == UTenType.Block) {
        std::vector<Tensor> bks = uTl.get_blocks_();
        Tensor res = zeros(1);
        for (int i = 0; i < bks.size(); i++) {
          Tensor tmp = Norm(bks[i]);
          res.at({0}) = res.at({0}) + tmp.at({0}) * tmp.at({0});
        }
        res.at({0}) = sqrt(res.at({0}));
        return res;
      } else {
        cytnx_error_msg(
          true,
          "[ERROR] Norm, unsupported type of UniTensor, only support Dense and Block. "
          "something wrong internal%s",
          "\n");
      }
    }

  }  // namespace linalg
}  // namespace cytnx
#endif  // BACKEND_TORCH
