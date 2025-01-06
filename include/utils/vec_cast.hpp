#ifndef CYTNX_UTILS_VEC_CAST_H_
#define CYTNX_UTILS_VEC_CAST_H_

#include <vector>
#include "Type.hpp"
#include <initializer_list>
namespace cytnx {

  template <typename Tfrom, typename Tto>
  std::vector<Tto> vec_cast(const std::vector<Tfrom>& in) {
    std::vector<Tto> out(in.begin(), in.end());
    return out;
  }

  template <typename Tfrom, typename Tto>
  std::vector<Tto> vec_cast(std::initializer_list<Tfrom> in) {
    auto vin = in;
    std::vector<Tto> out(vin.begin(), vin.end());
    return out;
  }

}  // namespace cytnx

#endif  // CYTNX_UTILS_VEC_CAST_H_
