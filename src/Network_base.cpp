#include <typeinfo>
#include "Network.hpp"
#include "utils/utils_internal_interface.hpp"
#include "linalg.hpp"

using namespace std;

namespace cytnx {
  void Network_base::Contract_plan(const std::vector<UniTensor> &utensors, const std::string &Tout,
                                   const std::vector<std::string> &alias,
                                   const std::string &contract_order) {
    cytnx_error_msg(true, "[ERROR][Network][Contract_plan] call from uninitialize network.%s",
                    "\n");
  }
  void Network_base::Fromfile(const std::string &fname) {
    cytnx_error_msg(true, "[ERROR][Network][Fromfile] call from uninitialize network.%s", "\n");
  }
  void Network_base::FromString(const std::vector<std::string> &fname) {
    cytnx_error_msg(true, "[ERROR][Network][FromString] call from uninitialize network.%s", "\n");
  }
  void Network_base::Savefile(const std::string &fname) {
    cytnx_error_msg(true, "[ERROR][Network][Savefile] call from uninitialize network.%s", "\n");
  }
  void Network_base::PutUniTensor(const std::string &name, const UniTensor &utensor) {
    cytnx_error_msg(true, "[ERROR][Network][PutUniTensor] call from uninitialize network.%s", "\n");
  }
  void Network_base::PutUniTensor(const cytnx_uint64 &idx, const UniTensor &utensor) {
    cytnx_error_msg(true, "[ERROR][Network][PutUniTensor] call from uninitialize network.%s", "\n");
  }
  void Network_base::PutUniTensors(const std::vector<string> &names,
                                   const std::vector<UniTensor> &utensors) {
    cytnx_error_msg(true, "[ERROR][Network][PutUniTensors] call from uninitialize network.%s",
                    "\n");
  }
  void Network_base::clear() {
    cytnx_error_msg(true, "[ERROR][Network][Clear] call from uninitialize network.%s", "\n");
  }
  std::string Network_base::getOptimalOrder() {
    cytnx_error_msg(true, "[ERROR][Network][getOptimalOrder] call from uninitialize network.%s",
                    "\n");
    return "";
  }
  UniTensor Network_base::Launch(const bool &optimal, const std::string &contract_order) {
    cytnx_error_msg(true, "[ERROR][Network][Launch] call from uninitialize network.%s", "\n");
    return UniTensor();
  }
  boost::intrusive_ptr<Network_base> Network_base::clone() {
    cytnx_error_msg(true, "[ERROR][Network][clone] call from uninitialize network. %s", "\n");
    return nullptr;
  }
  void Network_base::PrintNet(std::ostream &os) {
    os << "        [Empty Network]" << endl;
    os << "--- Please Load Network file ---\n";
    os << endl;
  }

}  // namespace cytnx
