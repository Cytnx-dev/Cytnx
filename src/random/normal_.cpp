#include "random.hpp"

#ifdef BACKEND_TORCH
#else

  #include "../backend/random_internal_interface.hpp"

namespace cytnx {
  namespace random {
    void normal_(Storage &Sin, const double &mean, const double &std, const unsigned int &seed) {
      cytnx_error_msg(
        (Sin.dtype() < 1) || (Sin.dtype() > 4),
        "[ERROR][Random.normal_] Normal distribution only accept real/imag floating type.%s", "\n");
      if (Sin.device() == Device.cpu) {
        random_internal::rii.Normal[Sin.dtype()](Sin._impl, mean, std, seed);
      } else {
  #ifdef UNI_GPU
        random_internal::rii.cuNormal[Sin.dtype()](Sin._impl, mean, std, seed);
  #else
        cytnx_error_msg(true, "[ERROR][normal_] Tensor is on GPU without CUDA support.%s", "\n");
  #endif
      }
    }
    void normal_(Tensor &Tin, const double &mean, const double &std, const unsigned int &seed) {
      cytnx_error_msg(
        (Tin.dtype() < 1) || (Tin.dtype() > 4),
        "[ERROR][Random.normal_] Normal distribution only accept real/imag floating type.%s", "\n");
      if (Tin.device() == Device.cpu) {
        random_internal::rii.Normal[Tin.dtype()](Tin._impl->storage()._impl, mean, std, seed);
      } else {
  #ifdef UNI_GPU
        random_internal::rii.cuNormal[Tin.dtype()](Tin._impl->storage()._impl, mean, std, seed);
  #else
        cytnx_error_msg(true, "[ERROR][normal_] Tensor is on GPU without CUDA support.%s", "\n");
  #endif
      }
    }

    void normal_(UniTensor &Tin, const double &mean, const double &std, const unsigned int &seed) {
      if (Tin.uten_type() != UTenType.Dense) {
        for (cytnx_int64 i = 0; i < Tin.get_blocks_().size(); i++) {
          normal_(Tin.get_blocks_()[i], mean, std, seed + i);
        }
      } else {
        normal_(Tin.get_block_(), mean, std, seed);
      }
    }

  }  // namespace random
}  // namespace cytnx
#endif  // BACKEND_TORCH
