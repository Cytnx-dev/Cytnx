#ifndef CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUSETZEROS_GPU_H_
#define CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUSETZEROS_GPU_H_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "cytnx_error.hpp"
namespace cytnx {
  namespace utils_internal {

#ifdef UNI_GPU
    void cuSetZeros(void* c_ptr, const cytnx_uint64& bytes);
#endif
  }  // namespace utils_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUSETZEROS_GPU_H_
