#include "utils/vec2d_col_sort.hpp"
#include "utils/utils.hpp"
#include <algorithm>
#include <vector>
namespace cytnx {

  void vec2d_col_sort(std::vector<std::vector<cytnx_int64>> &v1) {
    std::sort(v1.begin(), v1.end(), _fx_compare_vec_inc);
  }
  /*
  std::vector<cytnx_uint64> vec2d_col_sort(std::vector<std::vector<cytnx_int64> > &v1, const bool
  &return_order){

    if(return_order){
        vector<cytnx_uint64> out(v1.size());

        std::itoa(out.begin(),out.end(),0);

        std::stable_sort(out.begin(),out.end(),utils_internal::_fx_compare_vec_inc[v1&]);

        return out;

    }else{

        std::sort(v1.begin(), v1.end(), utils_internal::_fx_compare_vec_inc);

    }

  }
  */

}  // namespace cytnx
