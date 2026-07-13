#include "linalg.hpp"

#include "Generator.hpp"
#include "Tensor.hpp"

#ifdef BACKEND_TORCH
#else

  #include "backend/linalg_internal_interface.hpp"
namespace cytnx {
  namespace linalg {
    Tensor Diag(const Tensor &Tin) {
      cytnx_error_msg(Tin.is_void(), "[Diag] cannot operate on an uninitialized Tensor.%s", "\n");
      const int rank = Tin.rank();
      cytnx_error_msg(rank == 0 || rank > 2,
                      "[ERROR] the input tensor should be rank-1 or rank-2 Tensor.%s", "\n");

      if (rank == 2) {
        cytnx_error_msg(Tin.shape()[0] != Tin.shape()[1],
                        "[ERROR] the input tensor is not a square matrix.%s", "\n");
      }

      Tensor out = rank == 1 ? zeros({Tin.shape()[0], Tin.shape()[0]}, Tin.dtype(), Tin.device())
                             : zeros({Tin.shape()[0]}, Tin.dtype(), Tin.device());
      if (out.is_empty()) return out;

      if (Tin.device() == Device.cpu) {
        if (rank == 1) {
          cytnx::linalg_internal::lii.Diag_ii[out.dtype()](
            out._impl->storage()._impl, Tin._impl->storage()._impl, Tin.shape()[0], 0);
        } else {
          cytnx::linalg_internal::lii.Diag_ii[out.dtype()](
            out._impl->storage()._impl, Tin._impl->storage()._impl, Tin.shape()[0], 1);
        }
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(out.device()));
        if (rank == 1) {
          cytnx::linalg_internal::lii.cuDiag_ii[out.dtype()](
            out._impl->storage()._impl, Tin._impl->storage()._impl, Tin.shape()[0], 0);
        } else {
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
