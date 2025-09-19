#ifndef CYTNX_UTILS_VEC_MAP_H_
#define CYTNX_UTILS_VEC_MAP_H_

#include <vector>
#include "Type.hpp"
namespace cytnx {
  template <class T>
  std::vector<T> vec_map(const std::vector<T>& in, const std::vector<cytnx_uint64>& mapper);

};

#endif  // CYTNX_UTILS_VEC_MAP_H_
