#ifndef CYTNX_UTILS_VEC_SORT_H_
#define CYTNX_UTILS_VEC_SORT_H_

#include <vector>
#include <algorithm>
#include <numeric>
#include "Type.hpp"

namespace cytnx {
  template <class T>
  std::vector<cytnx_uint64> vec_sort(std::vector<T>& in, const bool& return_order = true);

}  // namespace cytnx

#endif  // CYTNX_UTILS_VEC_SORT_H_
