#ifndef _H_Alloc_cpu_
#define _H_Alloc_cpu_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "../../Type.hpp"
#include "../../cytnx_error.hpp"

namespace cytnx {
  namespace utils_internal {

    void* Calloc_cpu(const cytnx_uint64& N, const cytnx_uint64& perelem_bytes);
    void* Malloc_cpu(const cytnx_uint64& bytes);

  }  // namespace utils_internal
}  // namespace cytnx
#endif
