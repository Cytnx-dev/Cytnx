#include "linalg.hpp"
#include <iostream>
#include <vector>
#include "Tensor.hpp"

#ifdef BACKEND_TORCH
#else
  #include "backend/linalg_internal_interface.hpp"

namespace cytnx {
  namespace linalg {
    Tensor Dot(const Tensor &Tl, const Tensor &Tr) {
      cytnx_error_msg(Tl.is_void() || Tr.is_void(),
                      "[Dot] cannot operate on an uninitialized Tensor.%s", "\n");
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

        // check type: promote to a common dtype. The promoted dtype can differ
        // from both inputs (e.g. ComplexFloat x Double -> ComplexDouble), so
        // cast both operands; astype is a no-op when the dtype already matches.
        const unsigned int out_dtype = Type.type_promote(Tl.dtype(), Tr.dtype());
        Tensor out;
        std::vector<cytnx_uint64> newshape = Tl.shape();
        newshape.pop_back();
        std::size_t lin_dim = 1;
        for (int i = 0; i < newshape.size(); i++) lin_dim *= newshape[i];

        const bool zero_inner_dimension = Tr.shape()[0] == 0;
        // A zero inner dimension gives a zero result. Other kernels fully overwrite out.
        out.Init(newshape, out_dtype, Tl.device(), zero_inner_dimension);
        if (zero_inner_dimension || out.is_empty()) return out;

        Tensor tl = Tl.contiguous().astype(out_dtype);
        Tensor tr = Tr.contiguous().astype(out_dtype);

        if (Tl.device() == Device.cpu) {
          cytnx::linalg_internal::lii.Matvec_ii[out.dtype()](
            out._impl->storage()._impl, tl._impl->storage()._impl, tr._impl->storage()._impl,
            lin_dim, tr.shape()[0]);

          return out;

        } else {
  #ifdef UNI_GPU
          checkCudaErrors(cudaSetDevice(Tl.device()));
          cytnx::linalg_internal::lii.cuMatvec_ii[out.dtype()](
            out._impl->storage()._impl, tl._impl->storage()._impl, tr._impl->storage()._impl,
            lin_dim, tr.shape()[0]);

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
#endif  // BACKEND_TORCH
