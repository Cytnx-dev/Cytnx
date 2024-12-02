#ifndef CYTNX_UTILS_VEC_CONCATENATE_H_
#define CYTNX_UTILS_VEC_CONCATENATE_H_

#include <vector>
#include "Type.hpp"
namespace cytnx {

  template <class T>
  std::vector<T> vec_concatenate(const std::vector<T> &inL, const std::vector<T> &inR);

  template <class T>
  void vec_concatenate_(std::vector<T> &out, const std::vector<T> &inL, const std::vector<T> &inR);

}  // namespace cytnx

#endif  // CYTNX_UTILS_VEC_CONCATENATE_H_
