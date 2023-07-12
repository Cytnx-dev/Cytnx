#include "Gncon.hpp"

using namespace std;

// namespace cytnx {
//   UniTensor label_ncon(const std::vector<UniTensor> &tensors, const std::vector<std::string>
//   &tasks, const std::vector<std::string> &relabels) {
//     for(auto task : tasks){

//         cout << it <<endl;

//     }
// }  // namespace cytnx

namespace cytnx {
  std::string GnconType_class::getname(const int& nwrktype_id) {
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

  GnconType_class NtType;

  std::ostream& operator<<(std::ostream& os, const Gncon& in) {
    in._impl->PrintNet(os);
    return os;
  }

}  // namespace cytnx
