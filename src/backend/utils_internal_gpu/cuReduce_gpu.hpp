#ifndef SRC_BACKEND_UTILS_INTERNAL_GPU_CUREDUCE_GPU_H_
#define SRC_BACKEND_UTILS_INTERNAL_GPU_CUREDUCE_GPU_H_

#include "Type.hpp"

namespace cytnx {
  namespace utils_internal {
    template <class T>
    void cuReduce_gpu(T* out, T* in, const cytnx_uint64& Nelem);
  }  // namespace utils_internal
}  // namespace cytnx
#endif  // SRC_BACKEND_UTILS_INTERNAL_GPU_CUREDUCE_GPU_H_
