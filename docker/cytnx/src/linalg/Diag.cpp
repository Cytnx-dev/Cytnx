#include "linalg/linalg.hpp"

namespace cytnx {
  namespace linalg {
    Tensor Diag(const Tensor &Tin) {
      cytnx_error_msg(Tin.shape().size() != 1,
                      "[ERROR] the input tensor shoud be a rank-1 Tensor.%s", "\n");
      Tensor out({Tin.shape()[0], Tin.shape()[0]}, Tin.dtype(), Tin.device());

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Diag_ii[out.dtype()](
          out._impl->storage()._impl, Tin._impl->storage()._impl, Tin.shape()[0]);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(out.device()));
        cytnx::linalg_internal::lii.cuDiag_ii[out.dtype()](
          out._impl->storage()._impl, Tin._impl->storage()._impl, Tin.shape()[0]);
#else
        cytnx_error_msg(true, "[Diag] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }

      return out;
    }
  }  // namespace linalg
}  // namespace cytnx
