#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include <iostream>

namespace cytnx {
  namespace linalg {
    Tensor Matmul_dg(const Tensor &Tl, const Tensor &Tr) {
      cytnx_error_msg(Tl.shape().size() > 2,
                      "[Matmul_dg] error, tensor Tl ,Matmul_dg can only operate on rank-2 x rank-1 "
                      "(rank-1 x rank-2) Tensor.%s",
                      "\n");
      cytnx_error_msg(Tr.shape().size() > 2,
                      "[Matmul_dg] error, tensor Tr ,Matmul_dg can only operate on rank-2 x rank-1 "
                      "(rank-1 x rank-2) Tensor.%s",
                      "\n");
      cytnx_error_msg(Tl.shape().size() == Tr.shape().size(),
                      "[Matmul_dg] error, tensor Tr:rank[%d] Tl:rank[%d] ,Matmul_dg can only "
                      "operate on rank-2 x rank-1 (rank-1 x rank-2) Tensor.\n",
                      Tl.shape().size(), Tr.shape().size());

      // check device:
      cytnx_error_msg(Tl.device() != Tr.device(),
                      "[Matmul_dg] error two tensor should be on same device.%s", "\n");

      // check dimension match
      cytnx_error_msg(Tl.shape().back() != Tr.shape()[0],
                      "[Matmul_dg] error, dimension not match.%s", "\n");

      int diag_L;
      if (Tl.shape().size() == 1)
        diag_L = 1;
      else
        diag_L = 0;

      // check type:
      Tensor _tl = Tl.contiguous(), _tr = Tr.contiguous();
      Tensor out;
      if (Tl.dtype() != Tr.dtype()) {
        // do conversion:
        if (Tl.dtype() < Tr.dtype()) {
          _tr = _tr.astype(Tl.dtype());
          out.Init({Tl.shape()[0], Tr.shape().back()}, Tl.dtype(), Tl.device());
        } else {
          _tl = _tl.astype(Tr.dtype());
          out.Init({Tl.shape()[0], Tr.shape().back()}, Tr.dtype(), Tr.device());
        }
      } else {
        out.Init({Tl.shape()[0], Tr.shape().back()}, Tr.dtype(), Tr.device());
      }
      // out.storage().set_zeros();

      if (Tl.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Matmul_dg_ii[_tl.dtype()](
          out._impl->storage()._impl, _tl._impl->storage()._impl, _tr._impl->storage()._impl,
          _tl.shape()[0], _tl.shape().back(), _tr.shape().back(), diag_L);

        // cytnx_error_msg(true,"[Developing][Matmul_dg][CPU]%s","\n");

        return out;

      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tl.device()));
        // std::cout << _tl << std::endl;
        // std::cout << _tr << std::endl;
        cytnx::linalg_internal::lii.cuMatmul_dg_ii[_tl.dtype()](
          out._impl->storage()._impl, _tl._impl->storage()._impl, _tr._impl->storage()._impl,
          _tl.shape()[0], _tl.shape().back(), _tr.shape().back(), diag_L);

        // std::cout << "GPU Matmul OUT" << std::endl;
        // std::cout << out <<std::endl;
        // std::cout << "xxxxxxxxxxxxxx\n";
        return out;
#else
        cytnx_error_msg(true, "[Matmul] fatal error,%s",
                        "try to call the gpu section without CUDA support.\n");
        return Tensor();
#endif
      }
    }

  }  // namespace linalg
}  // namespace cytnx
