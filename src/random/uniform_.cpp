#include "random.hpp"

#ifdef BACKEND_TORCH
#else

  #include "../backend/random_internal_interface.hpp"

namespace cytnx {
  namespace random {
    std::random_device __static_random_device;

    void uniform_(Storage &Sin, const double &low, const double &high, const unsigned int &seed) {
      cytnx_error_msg(
        (Sin.dtype() < 1) || (Sin.dtype() > 4),
        "[ERROR][Random.uniform_] Uniform distribution only accept real/imag floating type.%s",
        "\n");
      cytnx_error_msg(high <= low,
                      "[ERROR][Random.uniform_] higher-bound should be > lower-bound.%s", "\n");
      if (Sin.device() == Device.cpu) {
        random_internal::rii.Uniform[Sin.dtype()](Sin._impl, low, high, seed);
      } else {
  #ifdef UNI_GPU
        cytnx_error_msg(true, "[Developing.]%s", "\n");
        random_internal::rii.cuUniform[Sin.dtype()](Sin._impl, low, high, seed);
          // Sin = low + Sin*(high-low); // we need Storage arithmetic!

  #else
        cytnx_error_msg(true, "[ERROR][uniform_] Tensor is on GPU without CUDA support.%s", "\n");
  #endif
      }
    }
    void uniform_(Tensor &Tin, const double &low, const double &high, const unsigned int &seed) {
      cytnx_error_msg(
        (Tin.dtype() < 1) || (Tin.dtype() > 4),
        "[ERROR][Random.uniform_] Uniform distribution only accept real/imag floating type.%s",
        "\n");
      cytnx_error_msg(high <= low,
                      "[ERROR][Random.uniform_] higher-bound should be > lower-bound.%s", "\n");
      if (Tin.device() == Device.cpu) {
        random_internal::rii.Uniform[Tin.dtype()](Tin._impl->storage()._impl, low, high, seed);
      } else {
  #ifdef UNI_GPU
        // cytnx_error_msg(true, "[Developing]%s", "\n");
        random_internal::rii.cuUniform[Tin.dtype()](Tin._impl->storage()._impl, low, high, seed);
          // Tin = low + Tin*(high-low);
  #else
        cytnx_error_msg(true, "[ERROR][uniform_] Tensor is on GPU without CUDA support.%s", "\n");
  #endif
      }
    }

    void uniform_(UniTensor &Tin, const double &low, const double &high, const unsigned int &seed) {
      if (Tin.uten_type() != UTenType.Dense) {
        for (cytnx_int64 i = 0; i < Tin.get_blocks_().size(); i++) {
          uniform_(Tin.get_blocks_()[i], low, high, seed + i);
        }
      } else {
        uniform_(Tin.get_block_(), low, high, seed);
      }
    }

  }  // namespace random
}  // namespace cytnx
#endif
