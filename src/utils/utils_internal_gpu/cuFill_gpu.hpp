#ifndef _H_cuFill_gpu_
#define _H_cuFill_gpu_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "Storage.hpp"
#include "cytnx_error.hpp"

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

#endif
