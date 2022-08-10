#ifndef _H_cuReduce_gpu_
#define _H_cuReduce_gpu_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "cytnx_error.hpp"
#include "Storage.hpp"

namespace cytnx {
  namespace utils_internal {

    void cuReduce_gpu_d(cytnx_double* out, cytnx_double* in, const cytnx_uint64& Nelem);
    void cuReduce_gpu_f(cytnx_float* out, cytnx_float* in, const cytnx_uint64& Nelem);
    void cuReduce_gpu_cf(cytnx_complex64* out, cytnx_complex64* in, const cytnx_uint64& Nelem);
    void cuReduce_gpu_cd(cytnx_complex128* out, cytnx_complex128* in, const cytnx_uint64& Nelem);
  }  // namespace utils_internal
}  // namespace cytnx
#endif
