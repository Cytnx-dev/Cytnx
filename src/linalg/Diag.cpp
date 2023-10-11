#include "linalg.hpp"

#include "Generator.hpp"
#include "Tensor.hpp"

#ifdef BACKEND_TORCH
#else

  #include "../backend/linalg_internal_interface.hpp"
namespace cytnx {
  namespace linalg {
    Tensor Diag(const Tensor &Tin) {
      cytnx_error_msg(Tin.shape().size() > 2,
                      "[ERROR] the input tensor shoud be rank-1 or rank-2 Tensor.%s", "\n");

      Tensor out;

      if (Tin.device() == Device.cpu) {
        if (Tin.shape().size() == 1) {
          out = zeros({Tin.shape()[0], Tin.shape()[0]}, Tin.dtype(), Tin.device());
          cytnx::linalg_internal::lii.Diag_ii[out.dtype()](
            out._impl->storage()._impl, Tin._impl->storage()._impl, Tin.shape()[0], 0);
        } else {
          cytnx_error_msg(Tin.shape()[0] != Tin.shape()[1],
                          "[ERROR] the input tensor is not a square matrix.%s", "\n");
          out = zeros({Tin.shape()[0]}, Tin.dtype(), Tin.device());
          cytnx::linalg_internal::lii.Diag_ii[out.dtype()](
            out._impl->storage()._impl, Tin._impl->storage()._impl, Tin.shape()[0], 1);
        }
      } else {
  #ifdef UNI_GPU
        // cytnx_error_msg(Tin.shape().size() != 1,
        //                 "[ERROR] the input tensor should be a rank-1 Tensor.%s", "\n");
        if (Tin.shape().size() == 1) {
          out = zeros({Tin.shape()[0], Tin.shape()[0]}, Tin.dtype(), Tin.device());
          checkCudaErrors(cudaSetDevice(out.device()));
          cytnx::linalg_internal::lii.cuDiag_ii[out.dtype()](
            out._impl->storage()._impl, Tin._impl->storage()._impl, Tin.shape()[0], 0);
        } else {
          out = zeros({Tin.shape()[0]}, Tin.dtype(), Tin.device());
          checkCudaErrors(cudaSetDevice(out.device()));
          cytnx::linalg_internal::lii.cuDiag_ii[out.dtype()](
            out._impl->storage()._impl, Tin._impl->storage()._impl, Tin.shape()[0], 1);
        }
  #else
        cytnx_error_msg(true, "[Diag] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }
  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
