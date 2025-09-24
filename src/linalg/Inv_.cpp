#include "linalg.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"

#ifdef BACKEND_TORCH
#else

  #include "../backend/linalg_internal_interface.hpp"

namespace cytnx {
  namespace linalg {
    void Inv_(Tensor &Tin, const double &clip) {
      if (Tin.dtype() == Type.Void) {
        cytnx_error_msg(true, "[ERROR][Inv_] Cannot operate on un-initialize Tensor.%s", "\n");
      } else if (Tin.dtype() > 4) {
        Tin = Tin.astype(Type.Double);
      }

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Inv_inplace_ii[Tin.dtype()](
          Tin._impl->storage()._impl, Tin._impl->storage()._impl->size(), clip);
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tin.device()));
        cytnx::linalg_internal::lii.cuInv_inplace_ii[Tin.dtype()](
          Tin._impl->storage()._impl, Tin._impl->storage()._impl->size(), clip);
  #else
        cytnx_error_msg(true, "[Inv_] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }
    }

  }  // namespace linalg
}  // namespace cytnx

namespace cytnx {
  namespace linalg {
    void Inv_(cytnx::UniTensor &Tin, double clip) {
      if (Tin.uten_type() == UTenType.Dense) {
        Tin.get_block_().Inv_(clip);
      } else if (Tin.uten_type() == UTenType.Block || Tin.uten_type() == UTenType.BlockFermionic) {
        for (auto &blk : Tin.get_blocks_()) {
          blk.Inv_(clip);
        }
      } else if (Tin.uten_type() == UTenType.Void) {
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
