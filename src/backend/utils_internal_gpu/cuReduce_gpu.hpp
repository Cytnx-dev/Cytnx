#ifndef _H_cuReduce_gpu_
#define _H_cuReduce_gpu_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "cytnx_error.hpp"
#include "backend/Storage.hpp"

namespace cytnx {
  namespace utils_internal {

    void cuReduce_gpu_u16(cytnx_uint16* out, cytnx_uint16* in, const cytnx_uint64& Nelem);
    void cuReduce_gpu_i16(cytnx_int16* out, cytnx_int16* in, const cytnx_uint64& Nelem);
    void cuReduce_gpu_u32(cytnx_uint32* out, cytnx_uint32* in, const cytnx_uint64& Nelem);
    void cuReduce_gpu_i32(cytnx_int32* out, cytnx_int32* in, const cytnx_uint64& Nelem);
    void cuReduce_gpu_u64(cytnx_uint64* out, cytnx_uint64* in, const cytnx_uint64& Nelem);
    void cuReduce_gpu_i64(cytnx_int64* out, cytnx_int64* in, const cytnx_uint64& Nelem);
    void cuReduce_gpu_d(cytnx_double* out, cytnx_double* in, const cytnx_uint64& Nelem);
    void cuReduce_gpu_f(cytnx_float* out, cytnx_float* in, const cytnx_uint64& Nelem);
    void cuReduce_gpu_cf(cytnx_complex64* out, cytnx_complex64* in, const cytnx_uint64& Nelem);
    void cuReduce_gpu_cd(cytnx_complex128* out, cytnx_complex128* in, const cytnx_uint64& Nelem);

  }  // namespace utils_internal
}  // namespace cytnx
#endif
