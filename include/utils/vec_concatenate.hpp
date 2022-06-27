#ifndef __H_vec_concatenate_
#define __H_vec_concatenate_

#include <vector>
#include "Type.hpp"
namespace cytnx {

  template <class T>
  std::vector<T> vec_concatenate(const std::vector<T> &inL, const std::vector<T> &inR);

  template <class T>
  void vec_concatenate_(std::vector<T> &out, const std::vector<T> &inL, const std::vector<T> &inR);

}  // namespace cytnx
#endif
