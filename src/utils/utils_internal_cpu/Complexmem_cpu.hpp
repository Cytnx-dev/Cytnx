#ifndef _H_Complexmem_cpu_
#define _H_Complexmem_cpu_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "Storage.hpp"
#include "cytnx_error.hpp"

namespace cytnx {
  namespace utils_internal {

    void Complexmem_cpu_cdtd(void *out, void *in, const cytnx_uint64 &Nelem, const bool get_real);
    void Complexmem_cpu_cftf(void *out, void *in, const cytnx_uint64 &Nelem, const bool get_real);

  }  // namespace utils_internal
}  // namespace cytnx
#endif
