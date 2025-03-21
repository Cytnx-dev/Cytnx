#ifndef CYTNX_UTILS_VEC_CLONE_H_
#define CYTNX_UTILS_VEC_CLONE_H_

#include <vector>
#include "Type.hpp"
namespace cytnx {

  template <class T>
  std::vector<T> vec_clone(const std::vector<T>& in_vec);

  template <class T>
  std::vector<T> vec_clone(const std::vector<T>& in_vec, const std::vector<cytnx_uint64>& locators);
}  // namespace cytnx

#endif  // CYTNX_UTILS_VEC_CLONE_H_
