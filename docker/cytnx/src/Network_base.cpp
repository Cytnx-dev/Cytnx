#include <typeinfo>
#include "Network.hpp"
#include "utils/utils_internal_interface.hpp"
#include "linalg/linalg.hpp"

using namespace std;

namespace cytnx {

  void Network_base::Fromfile(const std::string &fname) {
    cytnx_error_msg(true, "[ERROR][Network][Fromfile] call from uninitialize network.%s", "\n");
  }

  void Network_base::PutUniTensor(const std::string &name, const UniTensor &utensor,
                                  const bool &is_clone) {
    cytnx_error_msg(true, "[ERROR][Network][PutUniTensor] call from uninitialize network.%s", "\n");
  }
  void Network_base::PutUniTensor(const cytnx_uint64 &idx, const UniTensor &utensor,
                                  const bool &is_clone) {
    cytnx_error_msg(true, "[ERROR][Network][PutUniTensor] call from uninitialize network.%s", "\n");
  }
  void Network_base::Clear() {
    cytnx_error_msg(true, "[ERROR][Network][Clear] call from uninitialize network.%s", "\n");
  }
  UniTensor Network_base::Launch() {
    cytnx_error_msg(true, "[ERROR][Network][Launch] call from uninitialize network.%s", "\n");
  }
  boost::intrusive_ptr<Network_base> Network_base::clone() {
    cytnx_error_msg(true, "[ERROR][Network][clone] call from uninitialize network. %s", "\n");
  }

}  // namespace cytnx
