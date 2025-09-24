#include "linalg.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"

#ifdef BACKEND_TORCH
#else
  #include "backend/linalg_internal_interface.hpp"

namespace cytnx {
  namespace linalg {
    Tensor Pow(const Tensor &Tin, const double &p) {
      Tensor out;
      if (Tin.dtype() > 4)
        out = Tin.astype(Type.Double);
      else
        out = Tin.clone();

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Pow_ii[out.dtype()](out._impl->storage()._impl,
                                                        out._impl->storage()._impl,
                                                        out._impl->storage()._impl->size(), p);
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(out.device()));
        cytnx::linalg_internal::lii.cuPow_ii[out.dtype()](out._impl->storage()._impl,
                                                          Tin._impl->storage()._impl,
                                                          Tin._impl->storage()._impl->size(), p);
          // cytnx_error_msg(true,"[Pow][GPU] developing%s","\n");
  #else
        cytnx_error_msg(true, "[Pow] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

    Tensor Pow(const Tensor &Tin, const Scalar &p) {
      Tensor out;
      if (Tin.dtype() > 4)
        out = Tin.astype(Type.Double);
      else
        out = Tin.clone();

      double dp = double(p);

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Pow_ii[out.dtype()](out._impl->storage()._impl,
                                                        out._impl->storage()._impl,
                                                        out._impl->storage()._impl->size(), dp);
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(out.device()));
        cytnx::linalg_internal::lii.cuPow_ii[out.dtype()](out._impl->storage()._impl,
                                                          Tin._impl->storage()._impl,
                                                          Tin._impl->storage()._impl->size(), dp);
          // cytnx_error_msg(true,"[Pow][GPU] developing%s","\n");
  #else
        cytnx_error_msg(true, "[Pow] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }

      return out;
    }

  }  // namespace linalg
}  // namespace cytnx

namespace cytnx {
  namespace linalg {
    cytnx::UniTensor Pow(const cytnx::UniTensor &Tin, const double &p) {
      cytnx::UniTensor out;
      if (Tin.uten_type() == UTenType.Dense) {
        out = Tin.clone();
        out.get_block_().Pow_(p);
      } else if (Tin.uten_type() == UTenType.Block || Tin.uten_type() == UTenType.BlockFermionic) {
        out = Tin.clone();
        for (auto &blk : out.get_blocks_()) {
          blk.Pow_(p);
        }
      } else if (Tin.uten_type() == UTenType.Void) {
        cytnx_error_msg(
          true, "[ERROR][Pow] fatal internal, cannot call on an un-initialized UniTensor_base%s",
          "\n");
      } else {
        cytnx_error_msg(true, "[Pow]Unknown UniTensor type%s", "\n");
      }  // uten types
      return out;
    }

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
