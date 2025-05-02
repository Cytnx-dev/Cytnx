#ifndef CYTNX_BACKEND_UTILS_INTERNAL_CPU_SETZEROS_CPU_H_
#define CYTNX_BACKEND_UTILS_INTERNAL_CPU_SETZEROS_CPU_H_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "cytnx_error.hpp"

namespace cytnx {
  namespace utils_internal {

    void SetZeros(void* c_ptr, const cytnx_uint64& bytes);

  }
}  // namespace cytnx

#endif  // CYTNX_BACKEND_UTILS_INTERNAL_CPU_SETZEROS_CPU_H_
