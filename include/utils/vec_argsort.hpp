#ifndef __H_vec_argsort_
#define __H_vec_argsort_

#include <vector>
#include <algorithm>
#include <numeric>
#include "Type.hpp"

namespace cytnx {
  template <class T>
  std::vector<cytnx_uint64> vec_argsort(const std::vector<T>& in);
}  // namespace cytnx
#endif
