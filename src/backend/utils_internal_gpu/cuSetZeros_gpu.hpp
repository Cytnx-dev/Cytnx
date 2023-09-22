#ifndef _H_cuSetZeros_gpu_
#define _H_cuSetZeros_gpu_

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
#endif
