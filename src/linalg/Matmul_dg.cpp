#include "linalg.hpp"
#include <iostream>

#ifdef BACKEND_TORCH
#else
  #include "backend/linalg_internal_interface.hpp"

namespace cytnx {
  namespace linalg {
    Tensor Matmul_dg(const Tensor &Tl, const Tensor &Tr) {
      const cytnx_uint64 rank_l = Tl.rank();
      const cytnx_uint64 rank_r = Tr.rank();
      cytnx_error_msg(rank_l > 2,
                      "[Matmul_dg] error, tensor Tl ,Matmul_dg can only operate on rank-2 x rank-1 "
                      "(rank-1 x rank-2) Tensor.%s",
                      "\n");
      cytnx_error_msg(rank_r > 2,
                      "[Matmul_dg] error, tensor Tr ,Matmul_dg can only operate on rank-2 x rank-1 "
                      "(rank-1 x rank-2) Tensor.%s",
                      "\n");
      cytnx_error_msg(rank_l == 0 || rank_r == 0,
                      "[Matmul_dg] error, Matmul_dg does not support rank-0 Tensor operands.%s",
                      "\n");
      cytnx_error_msg(rank_l == rank_r,
                      "[Matmul_dg] error, tensor Tr:rank[%d] Tl:rank[%d] ,Matmul_dg can only "
                      "operate on rank-2 x rank-1 (rank-1 x rank-2) Tensor.\n",
                      static_cast<int>(rank_l), static_cast<int>(rank_r));

      // check device:
      cytnx_error_msg(Tl.device() != Tr.device(),
                      "[Matmul_dg] error two tensor should be on same device.%s", "\n");

      // check dimension match
      cytnx_error_msg(Tl.shape().back() != Tr.shape()[0],
                      "[Matmul_dg] error, dimension not match.%s", "\n");

      int diag_L;
      if (rank_l == 1)
        diag_L = 1;
      else
        diag_L = 0;

      // check type: promote to a common dtype. The promoted dtype can differ
      // from both inputs (e.g. ComplexFloat x Double -> ComplexDouble), so
      // cast both operands; astype is a no-op when the dtype already matches.
      const unsigned int out_dtype = Type.type_promote(Tl.dtype(), Tr.dtype());
      Tensor out;
      out.Init({Tl.shape()[0], Tr.shape().back()}, out_dtype, Tl.device());
      if (out.is_empty()) return out;

      Tensor tl = Tl.contiguous().astype(out_dtype);
      Tensor tr = Tr.contiguous().astype(out_dtype);

      if (Tl.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Matmul_dg_ii[out.dtype()](
          out._impl->storage()._impl, tl._impl->storage()._impl, tr._impl->storage()._impl,
          tl.shape()[0], tl.shape().back(), tr.shape().back(), diag_L);

        // cytnx_error_msg(true,"[Developing][Matmul_dg][CPU]%s","\n");

        return out;

      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tl.device()));
        cytnx::linalg_internal::lii.cuMatmul_dg_ii[out.dtype()](
          out._impl->storage()._impl, tl._impl->storage()._impl, tr._impl->storage()._impl,
          tl.shape()[0], tl.shape().back(), tr.shape().back(), diag_L);
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
#endif  // BACKEND_TORCH
