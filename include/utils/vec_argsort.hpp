#ifndef CYTNX_UTILS_VEC_ARGSORT_H_
#define CYTNX_UTILS_VEC_ARGSORT_H_

#include <vector>
#include <algorithm>
#include <numeric>
#include "Type.hpp"

namespace cytnx {
  template <class T>
  std::vector<cytnx_uint64> vec_argsort(const std::vector<T>& in);
}  // namespace cytnx

#endif  // CYTNX_UTILS_VEC_ARGSORT_H_
