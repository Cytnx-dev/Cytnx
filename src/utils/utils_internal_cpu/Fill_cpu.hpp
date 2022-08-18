#ifndef _H_Fill_cpu_
#define _H_Fill_cpu_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "Storage.hpp"
#include "cytnx_error.hpp"
namespace cytnx {
  namespace utils_internal {

    void Fill_cpu_cd(void *in, void *val, const cytnx_uint64 &Nelem);
    void Fill_cpu_cf(void *in, void *val, const cytnx_uint64 &Nelem);
    void Fill_cpu_d(void *in, void *val, const cytnx_uint64 &Nelem);
    void Fill_cpu_f(void *in, void *val, const cytnx_uint64 &Nelem);
    void Fill_cpu_i64(void *in, void *val, const cytnx_uint64 &Nelem);
    void Fill_cpu_u64(void *in, void *val, const cytnx_uint64 &Nelem);
    void Fill_cpu_i32(void *in, void *val, const cytnx_uint64 &Nelem);
    void Fill_cpu_u32(void *in, void *val, const cytnx_uint64 &Nelem);
    void Fill_cpu_u16(void *in, void *val, const cytnx_uint64 &Nelem);
    void Fill_cpu_i16(void *in, void *val, const cytnx_uint64 &Nelem);
    void Fill_cpu_b(void *in, void *val, const cytnx_uint64 &Nelem);
  }  // namespace utils_internal

}  // namespace cytnx

#endif
