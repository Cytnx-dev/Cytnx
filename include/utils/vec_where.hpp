#ifndef __H_vec_where_
#define __H_vec_where_

#include <vector>
#include "Type.hpp"
namespace cytnx {

  template <class T>
  cytnx_uint64 vec_where(const std::vector<T>& in, const T& key);

  template <class T>
  std::vector<cytnx_uint64> vec_argwhere(std::vector<T> const& v, const T& target);

}  // namespace cytnx
#endif
