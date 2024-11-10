#ifndef SRC_BACKEND_UTILS_INTERNAL_GPU_CUFILL_GPU_H_
#define SRC_BACKEND_UTILS_INTERNAL_GPU_CUFILL_GPU_H_

#include "Type.hpp"

namespace cytnx {
  namespace utils_internal {

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

#endif  // SRC_BACKEND_UTILS_INTERNAL_GPU_CUFILL_GPU_H_
