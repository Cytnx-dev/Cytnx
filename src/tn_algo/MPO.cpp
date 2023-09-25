#include "tn_algo/MPO.hpp"

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  namespace tn_algo {
    std::ostream& operator<<(std::ostream& os, const MPO& in) {
      in._impl->Print(os);
      return os;
    }

  }  // namespace tn_algo

}  // namespace cytnx
#endif
