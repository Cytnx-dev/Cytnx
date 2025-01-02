#include "random.hpp"

#ifndef BACKEND_TORCH
  #include <random>

  #include "backend/random_internal_interface.hpp"
  #include "backend/Storage.hpp"
  #include "Tensor.hpp"
  #include "Type.hpp"
  #include "UniTensor.hpp"

namespace cytnx {
  namespace random {
    std::random_device __static_random_device;

    /**
     * Equivalent to low + Tin * (high - low)
     *
     * Adding or multiplying a tensor with the double value will change the dtype of the tensor. To
     * keep the dtype of the tensor, we have to cast the double values before doing addition and
     * multiplication.
     *
     * @param Tin A tensor of which elements are in the range of (0.0, 1.0]. For the tensor holding
     * a complex type, both of the real part and the image part of each element are in (0.0, 1.0]
     * @param low the lower bound of the new range
     * @param hight the higher bound of the new range
     */
    void TransformToRange(Tensor &Tin, double low, double high) {
      if (high - low != 1.0) {
        if (Tin.dtype() == Type.ComplexDouble || Tin.dtype() == Type.Double) {
          Tin *= (high - low);
        } else if (Tin.dtype() == Type.ComplexFloat || Tin.dtype() == Type.Float) {
          Tin *= cytnx_float(high - low);
        } else {
          assert(false);
        }
      }
      if (low != 0) {
        if (Tin.dtype() == Type.ComplexDouble) {
          Tin += cytnx_complex128{low, low};
        } else if (Tin.dtype() == Type.ComplexFloat) {
          Tin += cytnx_complex64{static_cast<float>(low), static_cast<float>(low)};
        } else if (Tin.dtype() == Type.Double) {
          Tin += low;
        } else if (Tin.dtype() == Type.Float) {
          Tin += cytnx_float(low);
        } else {
          assert(false);
        }
      }
    }

    void uniform_(Storage &Sin, const double &low, const double &high, const unsigned int &seed) {
      cytnx_error_msg(
        (!Type.is_float(Sin.dtype())),
        "[ERROR][Random.uniform_] Uniform distribution only accept real/imag floating type.%s",
        "\n");
      cytnx_error_msg(high <= low,
                      "[ERROR][Random.uniform_] higher-bound should be > lower-bound.%s", "\n");
      if (Sin.device() == Device.cpu) {
        random_internal::rii.Uniform[Sin.dtype()](Sin._impl, low, high, seed);
      } else {
  #ifdef UNI_GPU
        random_internal::rii.cuUniform[Sin.dtype()](Sin._impl, low, high, seed);
        // TODO: The element-wise linear algebra functions should take iterators as the arguments
        // like `std` instead of the instances related to the Storage class. After landing that we
        // can refactor this workaround.
        Tensor wrapper = Tensor::from_storage(Sin);
        TransformToRange(wrapper, low, high);
        Sin._impl = wrapper.storage()._impl;
  #else
        cytnx_error_msg(true, "[ERROR][uniform_] Storage is on GPU without CUDA support.%s", "\n");
  #endif
      }
    }
    void uniform_(Tensor &Tin, const double &low, const double &high, const unsigned int &seed) {
      cytnx_error_msg(
        (!Type.is_float(Tin.dtype())),
        "[ERROR][Random.uniform_] Uniform distribution only accept real/imag floating type.%s",
        "\n");
      cytnx_error_msg(high <= low,
                      "[ERROR][Random.uniform_] higher-bound should be > lower-bound.%s", "\n");
      if (Tin.device() == Device.cpu) {
        random_internal::rii.Uniform[Tin.dtype()](Tin._impl->storage()._impl, low, high, seed);
      } else {
  #ifdef UNI_GPU
        random_internal::rii.cuUniform[Tin.dtype()](Tin._impl->storage()._impl, low, high, seed);
        // TODO: Use cublas's cublas<t>axpy() in the underlying function instead. Because
        // modifiying the underlying function will make conflicts with the #528 pull request,
        // leave this temporary solution here now.
        TransformToRange(Tin, low, high);
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
#endif  // BACKEND_TORCH
