#include "linalg.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"

#ifdef BACKEND_TORCH
#else

  #include "../backend/linalg_internal_interface.hpp"

namespace cytnx {
  namespace linalg {
    void Inv_(Tensor &Tio, const double &clip) {
      if (Tio.dtype() == Type.Void) {
        cytnx_error_msg(true, "[ERROR][Inv_] Cannot operate on un-initialize Tensor.%s", "\n");
      } else if (Tio.dtype() > 4) {
        Tio = Tio.astype(Type.Double);
      }

      if (Tio.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Inv_inplace_ii[Tio.dtype()](
          Tio._impl->storage()._impl, Tio._impl->storage()._impl->size(), clip);
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tio.device()));
        cytnx::linalg_internal::lii.cuInv_inplace_ii[Tio.dtype()](
          Tio._impl->storage()._impl, Tio._impl->storage()._impl->size(), clip);
  #else
        cytnx_error_msg(true, "[Inv_] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }
    }

    void Inv_(cytnx::UniTensor &Tio, double clip) {
      if (Tio.uten_type() == UTenType.Dense) {
        Tio.get_block_().Inv_(clip);
      } else if (Tio.uten_type() == UTenType.Block || Tio.uten_type() == UTenType.BlockFermionic) {
        for (auto &blk : Tio.get_blocks_()) {
          blk.Inv_(clip);
        }
      } else if (Tio.uten_type() == UTenType.Void) {
        cytnx_error_msg(
          true, "[ERROR][Inv_] fatal internal, cannot call on an un-initialized UniTensor_base%s",
          "\n");
      } else {
        cytnx_error_msg(true, "[Inv_]Unknown UniTensor type%s", "\n");
      }  // uten types
    }

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
