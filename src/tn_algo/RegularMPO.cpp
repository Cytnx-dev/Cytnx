#include "tn_algo/MPO.hpp"

using namespace std;
namespace cytnx {
  namespace tn_algo {
    std::ostream &RegularMPO::Print(std::ostream &os) {
      os << "[test][RegularMPO]" << endl;
      os << "MPO type:"
         << "Regular" << endl;
      os << "Number of Op: " << this->_TNs.size() << endl;

      return os;
    }

    UniTensor RegularMPO::get_op(const cytnx_uint64 &site_idx) {
      cytnx_error_msg(site_idx >= this->_TNs.size(), "[RegularMPO][ERROR] index out of bound%s",
                      "\n");
      return this->_TNs[site_idx];
    }

  }  // namespace tn_algo

}  // namespace cytnx
