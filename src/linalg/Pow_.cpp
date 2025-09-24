#include "linalg.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"

#ifdef BACKEND_TORCH
#else
  #include "backend/linalg_internal_interface.hpp"

namespace cytnx {
  namespace linalg {
    void Pow_(Tensor &Tio, const double &p) {
      if (Tio.dtype() > 4) Tio = Tio.astype(Type.Double);

      if (Tio.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Pow_ii[Tio.dtype()](Tio._impl->storage()._impl,
                                                        Tio._impl->storage()._impl,
                                                        Tio._impl->storage()._impl->size(), p);
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tio.device()));
        cytnx::linalg_internal::lii.cuPow_ii[Tio.dtype()](Tio._impl->storage()._impl,
                                                          Tio._impl->storage()._impl,
                                                          Tio._impl->storage()._impl->size(), p);
          // cytnx_error_msg(true,"[Pow][GPU] developing%s","\n");
  #else
        cytnx_error_msg(true, "[Pow_] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }
    }

    void Pow_(Tensor &Tio, const Scalar &p) {
      if (Tio.dtype() > 4) Tio = Tio.astype(Type.Double);
      double dp = double(p);

      if (Tio.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Pow_ii[Tio.dtype()](Tio._impl->storage()._impl,
                                                        Tio._impl->storage()._impl,
                                                        Tio._impl->storage()._impl->size(), dp);
      } else {
  #ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tio.device()));
        cytnx::linalg_internal::lii.cuPow_ii[Tio.dtype()](Tio._impl->storage()._impl,
                                                          Tio._impl->storage()._impl,
                                                          Tio._impl->storage()._impl->size(), dp);
          // cytnx_error_msg(true,"[Pow][GPU] developing%s","\n");
  #else
        cytnx_error_msg(true, "[Pow_] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
  #endif
      }
    }

  }  // namespace linalg
}  // namespace cytnx

namespace cytnx {
  namespace linalg {
    void Pow_(cytnx::UniTensor &Tio, const double &p) {
      if (Tio.uten_type() == UTenType.Dense) {
        Tio.get_block_().Pow_(p);
      } else if (Tio.uten_type() == UTenType.Block || Tio.uten_type() == UTenType.BlockFermionic) {
        for (auto &blk : Tio.get_blocks_()) {
          blk.Pow_(p);
        }
      } else if (Tio.uten_type() == UTenType.Void) {
        cytnx_error_msg(
          true, "[ERROR][Pow_] fatal internal, cannot call on an un-initialized UniTensor_base%s",
          "\n");
      } else {
        cytnx_error_msg(true, "[Pow_]Unknown UniTensor type%s", "\n");
      }  // uten types
    }

  }  // namespace linalg
}  // namespace cytnx

#endif  // BACKEND_TORCH
