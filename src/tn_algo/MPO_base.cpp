#include "tn_algo/MPO.hpp"

using namespace std;
namespace cytnx {
  namespace tn_algo {
    std::ostream &MPO_impl::Print(std::ostream &os) {
      cytnx_error_msg(
        true, "[ERROR] MPO_Base should not be called. Please initialize the MPO first.%s", "\n");
      return os;
    }

    UniTensor MPO_impl::get_op(const cytnx_uint64 &site_idx) {
      cytnx_error_msg(
        true, "[ERROR] MPO_Base should not be called. Please initialize the MPO first.%s", "\n");
      return UniTensor();
    }

  }  // namespace tn_algo

}  // namespace cytnx
