#ifndef CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUFILL_GPU_H_
#define CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUFILL_GPU_H_

#include "Type.hpp"

namespace cytnx {
  namespace utils_internal {
    void cuFill_gpu_cd(void* in, void* val, const cytnx_uint64&);
    void cuFill_gpu_cf(void* in, void* val, const cytnx_uint64&);
    void cuFill_gpu_d(void* in, void* val, const cytnx_uint64&);
    void cuFill_gpu_f(void* in, void* val, const cytnx_uint64&);
    void cuFill_gpu_i64(void* in, void* val, const cytnx_uint64&);
    void cuFill_gpu_u64(void* in, void* val, const cytnx_uint64&);
    void cuFill_gpu_i32(void* in, void* val, const cytnx_uint64&);
    void cuFill_gpu_u32(void* in, void* val, const cytnx_uint64&);
    void cuFill_gpu_u16(void* in, void* val, const cytnx_uint64&);
    void cuFill_gpu_i16(void* in, void* val, const cytnx_uint64&);
    void cuFill_gpu_b(void* in, void* val, const cytnx_uint64&);
  }  // namespace utils_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUFILL_GPU_H_
