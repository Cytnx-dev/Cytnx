#ifndef __H_vec_map_
#define __H_vec_map_

#include <vector>
#include "Type.hpp"
namespace cytnx {
  template <class T>
  std::vector<T> vec_map(const std::vector<T> &in, const std::vector<cytnx_uint64> &mapper);
};
#endif
