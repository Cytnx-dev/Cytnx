#ifndef BACKEND_UTILS_INTERNAL_GPU_CUCOMPLEXMEM_GPU_H_
#define BACKEND_UTILS_INTERNAL_GPU_CUCOMPLEXMEM_GPU_H_

#include "Type.hpp"

namespace cytnx {
  namespace utils_internal {
#ifdef UNI_GPU
    void cuComplexmem_gpu_cdtd(void *out, void *in, const cytnx_uint64 &Nelem, const bool get_real);
    void cuComplexmem_gpu_cftf(void *out, void *in, const cytnx_uint64 &Nelem, const bool get_real);
#endif

  }  // namespace utils_internal
}  // namespace cytnx
#endif  // BACKEND_UTILS_INTERNAL_GPU_CUCOMPLEXMEM_GPU_H_
