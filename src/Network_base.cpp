#include <typeinfo>
#include "Network.hpp"

#include "linalg.hpp"

using namespace std;

#ifdef BACKEND_TORCH
#else

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
  void Network_base::RmUniTensor(const std::string &name) {
    cytnx_error_msg(true, "[ERROR][Network][RmUniTensor] call from uninitialize network.%s", "\n");
  }
  void Network_base::RmUniTensor(const cytnx_uint64 &idx) {
    cytnx_error_msg(true, "[ERROR][Network][RmUniTensor] call from uninitialize network.%s", "\n");
  }
  void Network_base::RmUniTensors(const std::vector<string> &names) {
    cytnx_error_msg(true, "[ERROR][Network][RmUniTensors] call from uninitialize network.%s", "\n");
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

  void Network_base::setOrder(const bool &optimal, const std::string &contract_order) {
    cytnx_error_msg(true, "[ERROR][Network][setOrder] call from uninitialize network.%s", "\n");
  }

  std::string Network_base::getOrder() {
    cytnx_error_msg(true, "[ERROR][Network][getOrder] call from uninitialize network.%s", "\n");
    return "";
  }

  UniTensor Network_base::Launch() {
    cytnx_error_msg(true, "[ERROR][Network][Launch] call from uninitialize network.%s", "\n");
    return UniTensor();
  }

  void Network_base::construct(const std::vector<std::string> &alias,
                               const std::vector<std::vector<std::string>> &labels,
                               const std::vector<std::string> &outlabel, const cytnx_int64 &outrk,
                               const std::string &order, const bool optim) {
    cytnx_error_msg(true, "[ERROR][Network][construct] call from uninitialize network.%s", "\n");
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

};  // namespace cytnx

#endif
