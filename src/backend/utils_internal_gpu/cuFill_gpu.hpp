#ifndef CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUFILL_GPU_H_
#define CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUFILL_GPU_H_

#include "Type.hpp"

#ifdef UNI_GPU
  #include <complex>

  #include "cuda/std/complex"
#endif

namespace cytnx {
  namespace utils_internal {

#ifdef UNI_GPU
    /// @brief Maps a cytnx/std scalar type to the device arithmetic type used
    /// inside CUDA kernels. Complex types map to the bit-compatible
    /// cuda::std::complex, which provides device operator+ /
    /// construction-from-zero; every other type passes through unchanged.
    template <typename DType>
    struct ToCudaDType {
      typedef DType type;
    };

    template <typename DType>
    struct ToCudaDType<std::complex<DType>> {
      typedef cuda::std::complex<DType> type;
    };

    template <>
    struct ToCudaDType<cytnx_complex128> {
      typedef cuda::std::complex<double> type;
    };

    template <>
    struct ToCudaDType<cytnx_complex64> {
      typedef cuda::std::complex<float> type;
    };
#endif

    /**
     * @brief Assign the given value to the first `count` elements in the range beginning at
     * `first`.
     *
     * This function act the same as `std::fill_n` and is implemented in CUDA.
     *
     * @tparam DType the data type of the elements in the range
     *
     * @param first the beginning of the range
     * @param value the value to be assigned
     * @param count the number of elements to modify
     */
    template <typename DType>
    void FillGpu(void* first, const DType& value, cytnx_uint64 count);
  }  // namespace utils_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUFILL_GPU_H_
