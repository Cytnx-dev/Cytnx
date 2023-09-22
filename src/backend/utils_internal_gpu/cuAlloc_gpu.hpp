#ifndef _H_cuAlloc_gpu_
#define _H_cuAlloc_gpu_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "cytnx_error.hpp"
namespace cytnx {
  namespace utils_internal {

#ifdef UNI_GPU
    void* cuCalloc_gpu(const cytnx_uint64& N, const cytnx_uint64& perelem_bytes);
    void* cuMalloc_gpu(const cytnx_uint64& bytes);
#endif
  }  // namespace utils_internal
}  // namespace cytnx
#endif
