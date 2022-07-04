#ifndef _H_cuComplexmem_gpu_
#define _H_cuComplexmem_gpu_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "Storage.hpp"
#include "cytnx_error.hpp"

namespace cytnx {
  namespace utils_internal {
#ifdef UNI_GPU
    void cuComplexmem_gpu_cdtd(void *out, void *in, const cytnx_uint64 &Nelem, const bool get_real);
    void cuComplexmem_gpu_cftf(void *out, void *in, const cytnx_uint64 &Nelem, const bool get_real);
#endif

  }  // namespace utils_internal
}  // namespace cytnx
#endif
