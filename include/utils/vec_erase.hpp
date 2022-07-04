#ifndef __H_vec_erase_
#define __H_vec_erase_

#include <vector>
#include "Type.hpp"
namespace cytnx {

  template <class T>
  std::vector<T> vec_erase(const std::vector<T>& in, const std::vector<cytnx_uint64>& eraseper);

  template <class T>
  void vec_erase_(std::vector<T>& in, const std::vector<cytnx_uint64>& eraseper);

}  // namespace cytnx
#endif
