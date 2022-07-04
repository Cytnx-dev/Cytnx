#include "utils/vec2d_col_sort.hpp"
#include "utils/utils_internal_interface.hpp"
#include <algorithm>
#include <vector>
namespace cytnx {

  void vec2d_col_sort(std::vector<std::vector<cytnx_int64>> &v1) {
    std::sort(v1.begin(), v1.end(), utils_internal::_fx_compare_vec_inc);
  }

}  // namespace cytnx
