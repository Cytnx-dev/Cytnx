#ifndef CYTNX_BACKEND_UTILS_INTERNAL_CPU_FILL_CPU_H_
#define CYTNX_BACKEND_UTILS_INTERNAL_CPU_FILL_CPU_H_

#include "Type.hpp"

#ifdef UNI_OMP
  #include <omp.h>
#endif

namespace cytnx {
  namespace utils_internal {

    /**
     * @brief Assign the given value to the first `count` elements in the range beginning at
     * `first`.
     *
     * This function act the same as `std::fill_n`. The execution will be parallelized when OMP is
     * enabled.
     *
     * @tparam DType the data type of the elements in the range
     *
     * @param first the beginning of the range
     * @param value the value to be assigned
     * @param count the number of elements to modify
     */
    template <typename DType>
    void FillCpu(void *first, const DType &value, cytnx_uint64 count) {
      DType *typed_first = reinterpret_cast<DType *>(first);
#pragma omp parallel for schedule(static)
      for (cytnx_uint64 i = 0; i < count; i++) {
        typed_first[i] = value;
      }
    }
  }  // namespace utils_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_UTILS_INTERNAL_CPU_FILL_CPU_H_
