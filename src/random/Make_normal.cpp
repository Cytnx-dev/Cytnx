#include "random.hpp"
#include "random_internal_interface.hpp"

namespace cytnx {
  namespace random {
    void Make_normal(Storage &Sin, const double &mean, const double &std,
                     const unsigned int &seed) {
      cytnx_error_msg(
        (Sin.dtype() < 1) || (Sin.dtype() > 4),
        "[ERROR][Random.Make_normal] Normal distribution only accept real/imag floating type.%s",
        "\n");
      if (Sin.device() == Device.cpu) {
        random_internal::rii.Normal[Sin.dtype()](Sin._impl, mean, std, seed);
      } else {
#ifdef UNI_GPU
        random_internal::rii.cuNormal[Sin.dtype()](Sin._impl, mean, std, seed);
#else
        cytnx_error_msg(true, "[ERROR][Make_normal] Tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }
    }
    void Make_normal(Tensor &Tin, const double &mean, const double &std, const unsigned int &seed) {
      cytnx_error_msg(
        (Tin.dtype() < 1) || (Tin.dtype() > 4),
        "[ERROR][Random.Make_normal] Normal distribution only accept real/imag floating type.%s",
        "\n");
      if (Tin.device() == Device.cpu) {
        random_internal::rii.Normal[Tin.dtype()](Tin._impl->storage()._impl, mean, std, seed);
      } else {
#ifdef UNI_GPU
        random_internal::rii.cuNormal[Tin.dtype()](Tin._impl->storage()._impl, mean, std, seed);
#else
        cytnx_error_msg(true, "[ERROR][Make_normal] Tensor is on GPU without CUDA support.%s",
                        "\n");
#endif
      }
    }

    void Make_normal(UniTensor &Tin, const double &mean, const double &std,
                     const unsigned int &seed) {
      if (Tin.uten_type() == UTenType.Sparse) {
        for (cytnx_int64 i = 0; i < Tin.get_blocks_().size(); i++) {
          Make_normal(Tin.get_blocks_()[i], mean, std, seed + i);
        }
      } else {
        Make_normal(Tin.get_block_(), mean, std, seed);
      }
    }

  }  // namespace random
}  // namespace cytnx
