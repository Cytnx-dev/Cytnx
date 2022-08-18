#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include <iostream>
#include <vector>
#include "Tensor.hpp"

namespace cytnx {
  namespace linalg {
    Tensor Dot(const Tensor &Tl, const Tensor &Tr) {
      // checking contiguous
      // cytnx_error_msg(!Tl.is_contiguous(), "[Dot] error tensor Tl must be contiguous. Call
      // Contiguous_() or Contiguous() first%s","\n"); cytnx_error_msg(!Tr.is_contiguous(), "[Dot]
      // error tensor Tr must be contiguous. Call Contiguous_() or Contiguous() first%s","\n");

      // auto Tl = _Tl.contiguous();
      // auto Tr = _Tr.contiguous();

      // check device:
      cytnx_error_msg(Tl.device() != Tr.device(),
                      "[Dot] error two tensor should be on same device.%s", "\n");

      // checking mode:
      if (Tl.shape().size() == 1 && Tr.shape().size() == 1) {
        // Vec-Vec:
        return Vectordot(Tl, Tr);

      } else if (Tl.shape().size() == 2 && Tr.shape().size() == 2) {
        // Mat-Mat:
        return Matmul(Tl, Tr);

      } else if (Tl.shape().size() >= 2 && Tr.shape().size() == 1) {
        // Mat-Vec or T-Vec:

        // check dimension:
        cytnx_error_msg(
          Tl.shape().back() != Tr.shape()[0],
          "[Dot] error. The last dimension of Tl must be the same as dimension of Tr.%s", "\n");

        // check type:
        Tensor _tl = Tl.contiguous();
        Tensor _tr = Tr.contiguous();
        // std::cout << "MV" << std::endl;
        Tensor out;
        std::vector<cytnx_uint64> newshape = Tl.shape();
        newshape.pop_back();
        cytnx_int32 lin_dim = 1;
        for (int i = 0; i < newshape.size(); i++) lin_dim *= newshape[i];

        if (Tl.dtype() != Tr.dtype()) {
          // do conversion:
          if (Tl.dtype() < Tr.dtype()) {
            _tr = _tr.astype(Tl.dtype());
            out.Init(newshape, Tl.dtype(), Tl.device());
          } else {
            _tl = _tl.astype(Tr.dtype());
            out.Init(newshape, Tr.dtype(), Tr.device());
          }
        } else {
          out.Init(newshape, Tr.dtype(), Tr.device());
        }

        if (Tl.device() == Device.cpu) {
          cytnx::linalg_internal::lii.Matvec_ii[_tl.dtype()](
            out._impl->storage()._impl, _tl._impl->storage()._impl, _tr._impl->storage()._impl,
            lin_dim, _tr.shape()[0]);

          return out;

        } else {
#ifdef UNI_GPU
          checkCudaErrors(cudaSetDevice(Tl.device()));
          cytnx::linalg_internal::lii.cuMatvec_ii[_tl.dtype()](
            out._impl->storage()._impl, _tl._impl->storage()._impl, _tr._impl->storage()._impl,
            lin_dim, _tr.shape()[0]);

          return out;
#else
          cytnx_error_msg(true, "[Dot] fatal error,%s",
                          "try to call the gpu section without CUDA support.\n");
          return Tensor();
#endif
        }

      } else {
        cytnx_error_msg(
          true,
          "[Dot] error, invalid ranks for Tensors to perform Tensordot. rank-Tl [%d], rank-Tr [%d]",
          Tl.shape().size(), Tr.shape().size());
      }
    }

  }  // namespace linalg
}  // namespace cytnx
