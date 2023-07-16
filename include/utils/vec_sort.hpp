#ifndef __H_vec_sort_
#define __H_vec_sort_

#include <vector>
#include <algorithm>
#include <numeric>
#include "Type.hpp"

namespace cytnx {
  template <class T>
  std::vector<cytnx_uint64> vec_sort(std::vector<T>& in, const bool& return_order = true);

}  // namespace cytnx
#endif
