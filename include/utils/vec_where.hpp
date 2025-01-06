#ifndef CYTNX_UTILS_VEC_WHERE_H_
#define CYTNX_UTILS_VEC_WHERE_H_

#include <vector>
#include "Type.hpp"
namespace cytnx {

  template <class T>
  cytnx_uint64 vec_where(const std::vector<T>& in, const T& key);

  template <class T>
  std::vector<cytnx_uint64> vec_argwhere(std::vector<T> const& v, const T& target);

}  // namespace cytnx

#endif  // CYTNX_UTILS_VEC_WHERE_H_
