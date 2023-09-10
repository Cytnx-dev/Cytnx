#ifndef _H_SetZeros_cpu_
#define _H_SetZeros_cpu_

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
#endif
