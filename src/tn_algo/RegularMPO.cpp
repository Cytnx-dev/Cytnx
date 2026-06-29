#include "tn_algo/MPO.hpp"

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  namespace tn_algo {
    std::ostream &RegularMPO::Print(std::ostream &os) {
      os << "[test][RegularMPO]" << std::endl;
      os << "MPO type:"
         << "Regular" << std::endl;
      os << "Number of Op: " << this->_TNs.size() << std::endl;

      return os;
    }

    UniTensor RegularMPO::get_op(const cytnx_uint64 &site_idx) {
      cytnx_error_msg(site_idx >= this->_TNs.size(), "[RegularMPO][ERROR] index out of bound%s",
                      "\n");
      return this->_TNs[site_idx];
    }

  }  // namespace tn_algo

}  // namespace cytnx

#endif
