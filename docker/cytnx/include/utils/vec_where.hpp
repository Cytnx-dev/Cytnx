#ifndef __H_vec_where_
#define __H_vec_where_

#include <vector>
#include "Type.hpp"
namespace cytnx {

  template <class T>
  cytnx_uint64 vec_where(const std::vector<T>& in, const T& key);

}
#endif
