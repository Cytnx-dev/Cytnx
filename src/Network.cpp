#include <typeinfo>
#include "Network.hpp"

#include "linalg.hpp"

using namespace std;

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  std::string NetworkType_class::getname(const int& nwrktype_id) {
    if (nwrktype_id == this->Regular) {
      return string("Regular");
    } else if (nwrktype_id == this->Fermion) {
      return string("Fermion");
    } else {
      cytnx_error_msg(true, "%s", "[ERROR] invalid nwrktype_id");
      return string("");
    }
    // extend more in here!!
  }

  NetworkType_class NtType;

  std::ostream& operator<<(std::ostream& os, const Network& in) {
    in._impl->PrintNet(os);
    return os;
  }

}  // namespace cytnx
#endif
