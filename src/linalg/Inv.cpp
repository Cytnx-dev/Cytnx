#include "linalg.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"

#ifdef BACKEND_TORCH
#else
  #include "backend/linalg_internal_interface.hpp"

namespace cytnx {
  namespace linalg {
    Tensor Inv(const Tensor &Tin, const double &clip) {
      Tensor out;
      if (Tin.dtype() == Type.Void) {
        cytnx_error_msg(true, "[ERROR][Inv] Cannot operate on un-initialize Tensor.%s", "\n");
      } else if (Tin.dtype() > 4) {
        out = Tin.astype(Type.Double);
      } else {
        out = Tin.clone();
      }

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Inv_inplace_ii[out.dtype()](
          out._impl->storage()._impl, out._impl->storage()._impl->size(), clip);
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tin.device()));
        cytnx::linalg_internal::lii.cuInv_inplace_ii[out.dtype()](
          out._impl->storage()._impl, out._impl->storage()._impl->size(), clip);
  #else
        cytnx_error_msg(true, "[Inv] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }
      return out;
    }

    cytnx::UniTensor Inv(const cytnx::UniTensor &Tin, double clip) {
      cytnx::UniTensor out;
      if (Tin.uten_type() == UTenType.Dense) {
        out = Tin.clone();
        out.get_block_().Inv_(clip);
      } else if (Tin.uten_type() == UTenType.Block || Tin.uten_type() == UTenType.BlockFermionic) {
        out = Tin.clone();
        for (auto &blk : out.get_blocks_()) {
          blk.Inv_(clip);
        }
      } else if (Tin.uten_type() == UTenType.Void) {
        cytnx_error_msg(
          true, "[ERROR][Inv] fatal internal, cannot call on an un-initialized UniTensor_base%s",
          "\n");
      } else {
        cytnx_error_msg(true, "[Inv]Unknown UniTensor type%s", "\n");
      }  // uten types
      return out;
    }

  }  // namespace linalg
}  // namespace cytnx
#endif  // BACKEND_TORCH
