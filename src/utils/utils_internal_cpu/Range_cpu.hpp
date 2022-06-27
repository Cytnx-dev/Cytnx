#ifndef _H_Range_cpu_
#define _H_Range_cpu_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "Storage.hpp"
#include "cytnx_error.hpp"

namespace cytnx {
  namespace utils_internal {
    std::vector<cytnx_uint64> range_cpu(const cytnx_uint64 &len);
    std::vector<cytnx_uint64> range_cpu(const cytnx_uint64 &start, const cytnx_uint64 &end);

    template <class T>
    std::vector<T> range_cpu(const cytnx_int64 &len);
    template <class T>
    std::vector<T> range_cpu(const cytnx_int64 &start, const cytnx_int64 &end);

  }  // namespace utils_internal
}  // namespace cytnx
#endif
