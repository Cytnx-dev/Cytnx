#ifndef CYTNX_BACKEND_UTILS_INTERNAL_CPU_ALLOC_CPU_H_
#define CYTNX_BACKEND_UTILS_INTERNAL_CPU_ALLOC_CPU_H_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "cytnx_error.hpp"

namespace cytnx {
  namespace utils_internal {

    void* Calloc_cpu(const cytnx_uint64& N, const cytnx_uint64& perelem_bytes);
    void* Malloc_cpu(const cytnx_uint64& bytes);

  }  // namespace utils_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_UTILS_INTERNAL_CPU_ALLOC_CPU_H_
