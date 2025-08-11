#ifndef CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUCOMPLEXMEM_GPU_H_
#define CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUCOMPLEXMEM_GPU_H_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "backend/Storage.hpp"
#include "cytnx_error.hpp"

namespace cytnx {
  namespace utils_internal {
#ifdef UNI_GPU
    void cuComplexmem_gpu_cdtd(void *out, void *in, const cytnx_uint64 &Nelem, const bool get_real);
    void cuComplexmem_gpu_cftf(void *out, void *in, const cytnx_uint64 &Nelem, const bool get_real);
#endif

  }  // namespace utils_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUCOMPLEXMEM_GPU_H_
