#include <typeinfo>
#include "Gncon.hpp"

#include "linalg.hpp"

using namespace std;

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  void Gncon_base::Contract_plan(const std::vector<UniTensor> &utensors, const std::string &Tout,
                                 const std::vector<std::string> &alias,
                                 const std::string &contract_order) {
    cytnx_error_msg(true, "[ERROR][Gncon][Contract_plan] call from uninitialize Gncon.%s", "\n");
  }
  void Gncon_base::Fromfile(const std::string &fname) {
    cytnx_error_msg(true, "[ERROR][Gncon][Fromfile] call from uninitialize Gncon.%s", "\n");
  }
  void Gncon_base::FromString(const std::vector<std::string> &fname) {
    cytnx_error_msg(true, "[ERROR][Gncon][FromString] call from uninitialize Gncon.%s", "\n");
  }
  void Gncon_base::Savefile(const std::string &fname) {
    cytnx_error_msg(true, "[ERROR][Gncon][Savefile] call from uninitialize Gncon.%s", "\n");
  }
  void Gncon_base::PutUniTensor(const std::string &name, const UniTensor &utensor) {
    cytnx_error_msg(true, "[ERROR][Gncon][PutUniTensor] call from uninitialize Gncon.%s", "\n");
  }
  void Gncon_base::PutUniTensor(const cytnx_uint64 &idx, const UniTensor &utensor) {
    cytnx_error_msg(true, "[ERROR][Gncon][PutUniTensor] call from uninitialize Gncon.%s", "\n");
  }
  void Gncon_base::PutUniTensors(const std::vector<string> &names,
                                 const std::vector<UniTensor> &utensors) {
    cytnx_error_msg(true, "[ERROR][Gncon][PutUniTensors] call from uninitialize Gncon.%s", "\n");
  }
  void Gncon_base::clear() {
    cytnx_error_msg(true, "[ERROR][Gncon][Clear] call from uninitialize Gncon.%s", "\n");
  }
  std::string Gncon_base::getOptimalOrder() {
    cytnx_error_msg(true, "[ERROR][Gncon][getOptimalOrder] call from uninitialize Gncon.%s", "\n");
    return "";
  }
  UniTensor Gncon_base::Launch(const bool &optimal, const std::string &contract_order) {
    cytnx_error_msg(true, "[ERROR][Gncon][Launch] call from uninitialize Gncon.%s", "\n");
    return UniTensor();
  }
  boost::intrusive_ptr<Gncon_base> Gncon_base::clone() {
    cytnx_error_msg(true, "[ERROR][Gncon][clone] call from uninitialize Gncon. %s", "\n");
    return nullptr;
  }
  void Gncon_base::PrintNet(std::ostream &os) {
    os << "        [Empty Gncon]" << endl;
    os << "--- Please Load Gncon file ---\n";
    os << endl;
  }

}  // namespace cytnx
#endif
