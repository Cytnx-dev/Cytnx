#include "utils/utils.hpp"
#include <vector>
#include <iostream>
#include <string>
namespace cytnx {

  // this is used by internal function for compare.
  //-------------
  bool _fx_compare_vec_inc(const std::vector<cytnx_int64> &v1, const std::vector<cytnx_int64> &v2) {
    std::pair<std::vector<cytnx_int64>, std::vector<cytnx_int64>> p{v1, v2};

    return p.first < p.second;
  }
  bool _fx_compare_vec_dec(const std::vector<cytnx_int64> &v1, const std::vector<cytnx_int64> &v2) {
    std::pair<std::vector<cytnx_int64>, std::vector<cytnx_int64>> p{v1, v2};

    return p.first > p.second;
  }

}  // namespace cytnx
