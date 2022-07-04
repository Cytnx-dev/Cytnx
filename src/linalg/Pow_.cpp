#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include "Tensor.hpp"
#include "UniTensor.hpp"
namespace cytnx {
  namespace linalg {
    void Pow_(Tensor &Tin, const double &p) {
      if (Tin.dtype() > 4) Tin = Tin.astype(Type.Double);

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Pow_ii[Tin.dtype()](Tin._impl->storage()._impl,
                                                        Tin._impl->storage()._impl,
                                                        Tin._impl->storage()._impl->size(), p);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tin.device()));
        cytnx::linalg_internal::lii.cuPow_ii[Tin.dtype()](Tin._impl->storage()._impl,
                                                          Tin._impl->storage()._impl,
                                                          Tin._impl->storage()._impl->size(), p);
        // cytnx_error_msg(true,"[Pow][GPU] developing%s","\n");
#else
        cytnx_error_msg(true, "[Pow_] fatal error, the tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }
    }

    void Pow_(Tensor &Tin, const Scalar &p) {
      if (Tin.dtype() > 4) Tin = Tin.astype(Type.Double);
      double dp = double(p);

      if (Tin.device() == Device.cpu) {
        cytnx::linalg_internal::lii.Pow_ii[Tin.dtype()](Tin._impl->storage()._impl,
                                                        Tin._impl->storage()._impl,
                                                        Tin._impl->storage()._impl->size(), dp);
      } else {
#ifdef UNI_GPU
        checkCudaErrors(cudaSetDevice(Tin.device()));
        cytnx::linalg_internal::lii.cuPow_ii[Tin.dtype()](Tin._impl->storage()._impl,
                                                          Tin._impl->storage()._impl,
                                                          Tin._impl->storage()._impl->size(), dp);
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
    void Pow_(UniTensor &Tin, const double &p) {
      if (Tin.is_blockform()) {
        // cytnx_error_msg(true,"[Pow][SparseUniTensor] Developing%s","\n");
        auto tmp = Tin.get_blocks_();
        for (int i = 0; i < tmp.size(); i++) {
          tmp[i].Pow_(p);
        }
      } else {
        Tin.get_block_().Pow_(p);
      }
    }
  }  // namespace linalg
}  // namespace cytnx
