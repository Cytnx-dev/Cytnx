#ifndef CYTNX_UTILS_VEC_ERASE_H_
#define CYTNX_UTILS_VEC_ERASE_H_

#include <vector>
#include "Type.hpp"
namespace cytnx {

  template <class T>
  std::vector<T> vec_erase(const std::vector<T>& in, const std::vector<cytnx_uint64>& eraseper);

  template <class T>
  void vec_erase_(std::vector<T>& in, const std::vector<cytnx_uint64>& eraseper);

}  // namespace cytnx

#endif  // CYTNX_UTILS_VEC_ERASE_H_
