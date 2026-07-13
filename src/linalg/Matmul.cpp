#include "linalg.hpp"

#include <iostream>
#include "Tensor.hpp"

#ifdef BACKEND_TORCH
#else
  #include "backend/linalg_internal_interface.hpp"

namespace cytnx {
  namespace linalg {
    Tensor Matmul(const Tensor &Tl, const Tensor &Tr) {
      cytnx_error_msg(Tl.shape().size() != 2,
                      "[Matmul] error, tensor Tl ,Matmul can only operate on rank-2 Tensor.%s",
                      "\n");
      cytnx_error_msg(Tr.shape().size() != 2,
                      "[Matmul] error, tensor Tr ,Matmul can only operate on rank-2 Tensor.%s",
                      "\n");

      // cytnx_error_msg(!Tl.is_contiguous(), "[Matmul] error tensor Tl must be contiguous. Call
      // Contiguous_() or Contiguous() first%s","\n");

      // cytnx_error_msg(!Tr.is_contiguous(), "[Matmul] error tensor Tr must be contiguous. Call
      // Contiguous_() or Contiguous() first%s","\n");

      // check device:
      cytnx_error_msg(Tl.device() != Tr.device(),
                      "[Matmul] error two tensor should be on same device.%s", "\n");

      // check dimension match
      cytnx_error_msg(Tl.shape()[1] != Tr.shape()[0], "[Matmul] error, dimension not match.%s",
                      "\n");

      // check type: promote to a common dtype. The promoted dtype can differ
      // from both inputs (e.g. ComplexFloat x Double -> ComplexDouble), so
      // cast both operands; astype is a no-op when the dtype already matches.
      const unsigned int out_dtype = Type.type_promote(Tl.dtype(), Tr.dtype());
      Tensor out;
      const bool zero_inner_dimension = Tl.shape()[1] == 0;
      // A zero inner dimension produces a nonempty zero matrix when both outer
      // dimensions are nonzero. Other kernels fully overwrite their output.
      out.Init({Tl.shape()[0], Tr.shape()[1]}, out_dtype, Tl.device(), zero_inner_dimension);
      if (zero_inner_dimension || out.is_empty()) return out;

      Tensor tl = Tl.contiguous().astype(out_dtype);
      Tensor tr = Tr.contiguous().astype(out_dtype);

      if (Tl.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Matmul_ii[out.dtype()](
          out._impl->storage()._impl, tl._impl->storage()._impl, tr._impl->storage()._impl,
          tl.shape()[0], tl.shape()[1], tr.shape()[1]);

        return out;

      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tl.device()));
        cytnx::linalg_internal::lii.cuMatmul_ii[out.dtype()](
          out._impl->storage()._impl, tl._impl->storage()._impl, tr._impl->storage()._impl,
          tl.shape()[0], tl.shape()[1], tr.shape()[1]);
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
