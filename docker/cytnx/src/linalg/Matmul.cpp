#include "linalg/linalg.hpp"
#include <iostream>

namespace cytnx {
  namespace linalg {
    Tensor Matmul(const Tensor &Tl, const Tensor &Tr) {
      cytnx_error_msg(Tl.shape().size() != 2,
                      "[Matmul] error, tensor Tl ,Matmul can only operate on rank-2 Tensor.%s",
                      "\n");
      cytnx_error_msg(Tr.shape().size() != 2,
                      "[Matmul] error, tensor Tr ,Matmul can only operate on rank-2 Tensor.%s",
                      "\n");

      cytnx_error_msg(
        !Tl.is_contiguous(),
        "[Matmul] error tensor Tl must be contiguous. Call Contiguous_() or Contiguous() first%s",
        "\n");

      cytnx_error_msg(
        !Tr.is_contiguous(),
        "[Matmul] error tensor Tr must be contiguous. Call Contiguous_() or Contiguous() first%s",
        "\n");

      // check device:
      cytnx_error_msg(Tl.device() != Tr.device(),
                      "[Matmul] error two tensor should be on same device.%s", "\n");

      // check dimension match
      cytnx_error_msg(Tl.shape()[1] != Tr.shape()[0], "[Matmul] error, dimension not match.%s",
                      "\n");

      // check type:
      Tensor _tl, _tr;
      Tensor out;
      if (Tl.dtype() != Tr.dtype()) {
        // do conversion:
        if (Tl.dtype() < Tr.dtype()) {
          _tr = Tr.astype(Tl.dtype());
          _tl = Tl;
          out.Init({Tl.shape()[0], Tr.shape()[1]}, Tl.dtype(), Tl.device());
        } else {
          _tl = Tl.astype(Tr.dtype());
          _tr = Tr;
          out.Init({Tl.shape()[0], Tr.shape()[1]}, Tr.dtype(), Tr.device());
        }
      } else {
        _tl = Tl;
        _tr = Tr;
        out.Init({Tl.shape()[0], Tr.shape()[1]}, Tr.dtype(), Tr.device());
      }

      if (Tl.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Matmul_ii[_tl.dtype()](
          out._impl->storage()._impl, _tl._impl->storage()._impl, _tr._impl->storage()._impl,
          _tl.shape()[0], _tl.shape()[1], _tr.shape()[1]);

        return out;

      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tl.device()));
        cytnx::linalg_internal::lii.cuMatmul_ii[_tl.dtype()](
          out._impl->storage()._impl, _tl._impl->storage()._impl, _tl._impl->storage()._impl,
          _tl.shape()[0], _tl.shape()[1], _tr.shape()[1]);

        return out;
#else
        cytnx_error_msg(true, "[Matmul] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
#endif
      }
    }

  }  // namespace linalg
}  // namespace cytnx
