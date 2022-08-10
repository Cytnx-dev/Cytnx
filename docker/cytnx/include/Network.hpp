#ifndef _H_Network_
#define _H_Network_

#include "Type.hpp"
#include "Network.hpp"
#include "UniTensor.hpp"
#include "cytnx_error.hpp"
#include <initializer_list>
#include <vector>
#include <map>
#include <fstream>
#include "intrusive_ptr_base.hpp"
#include "utils/utils.hpp"
#include "contraction_tree.hpp"
namespace cytnx {

  /// @cond
  struct __ntwk {
    enum __nttype { Void = -1, Regular = 0, Fermion = 1 };
  };
  class NetworkType_class {
   public:
    enum : int { Void = -1, Regular = 0, Fermion = 1 };
    std::string getname(const int &nwrktype_id);
  };
  extern NetworkType_class NtType;
  /// @endcond

  /// @cond
  class Network_base : public intrusive_ptr_base<Network_base> {
   protected:
    int nwrktype_id;
    std::vector<UniTensor> tensors;
    std::vector<cytnx_int64> TOUT_labels;
    cytnx_uint64 TOUT_iBondNum;

    // bool ordered;

    // Contraction order.
    ContractionTree CtTree;
    std::vector<std::string> ORDER_tokens;

    // labels corr to the tn list.
    std::vector<std::vector<cytnx_int64>> label_arr;
    std::vector<cytnx_int64> iBondNums;

    // name of tn.
    std::vector<std::string> names;
    std::map<std::string, cytnx_uint64> name2pos;

   public:
    friend class FermionNetwork;
    friend class RegularNetwork;
    friend class Network;
    Network_base() : nwrktype_id(NtType.Void){};

    bool HasPutAllUniTensor() {
      for (cytnx_uint64 i = 0; i < this->tensors.size(); i++) {
        if (this->tensors[i].uten_type() == UTenType.Void) return false;
      }
      return true;
    }

    // void print_network() const;

    // void PreConstruct(bool force = true);

    // void PutTensor(cytnx_int64 idx, const UniTensor& UniT, bool force = true);

    // void PutTensor(const std::string  &name, const UniTensor &UniT, bool force = true);

    // UniTensor Launch(const std::string &Tname="");

    // std::string GetContractOrder() const;
    virtual void PutUniTensor(const std::string &name, const UniTensor &utensor,
                              const bool &is_clone);
    virtual void PutUniTensor(const cytnx_uint64 &idx, const UniTensor &utensor,
                              const bool &is_clone);
    virtual void Fromfile(const std::string &fname);
    virtual void Clear();
    virtual UniTensor Launch();
    virtual boost::intrusive_ptr<Network_base> clone();
    virtual ~Network_base(){};

  };  // Network_base

  class RegularNetwork : public Network_base {
   public:
    RegularNetwork() { this->nwrktype_id = NtType.Regular; };
    void Fromfile(const std::string &fname);
    void PutUniTensor(const std::string &name, const UniTensor &utensor,
                      const bool &is_clone = true);
    void PutUniTensor(const cytnx_uint64 &idx, const UniTensor &utensor,
                      const bool &is_clone = true);
    void Clear() {
      this->name2pos.clear();
      this->CtTree.clear();
      this->names.clear();
      this->iBondNums.clear();
      this->label_arr.clear();
      this->TOUT_labels.clear();
      this->TOUT_iBondNum = 0;
      this->ORDER_tokens.clear();
    }
    UniTensor Launch();
    boost::intrusive_ptr<Network_base> clone() {
      RegularNetwork *tmp = new RegularNetwork();
      tmp->name2pos = this->name2pos;
      tmp->CtTree = this->CtTree;
      tmp->names = this->names;
      tmp->iBondNums = this->iBondNums;
      tmp->label_arr = this->label_arr;
      tmp->TOUT_labels = this->TOUT_labels;
      tmp->TOUT_iBondNum = this->TOUT_iBondNum;
      tmp->ORDER_tokens = this->ORDER_tokens;
      boost::intrusive_ptr<Network_base> out(tmp);
      return out;
    }
    ~RegularNetwork(){};
  };

  class FermionNetwork : public Network_base {
   protected:
    // [Future] Swap gates.

   public:
    FermionNetwork() { this->nwrktype_id = NtType.Fermion; };
    void Fromfile(const std::string &fname){};
    void PutUniTensor(const std::string &name, const UniTensor &utensor,
                      const bool &is_clone = true){};
    void PutUniTensor(const cytnx_uint64 &idx, const UniTensor &utensor,
                      const bool &is_clone = true){};
    void Clear() {
      this->name2pos.clear();
      this->CtTree.clear();
      this->names.clear();
      this->iBondNums.clear();
      this->label_arr.clear();
      this->TOUT_labels.clear();
      this->TOUT_iBondNum = 0;
      this->ORDER_tokens.clear();
    }
    UniTensor Launch(){};
    boost::intrusive_ptr<Network_base> clone() {
      FermionNetwork *tmp = new FermionNetwork();
      tmp->name2pos = this->name2pos;
      tmp->CtTree = this->CtTree;
      tmp->names = this->names;
      tmp->iBondNums = this->iBondNums;
      tmp->label_arr = this->label_arr;
      tmp->TOUT_labels = this->TOUT_labels;
      tmp->TOUT_iBondNum = this->TOUT_iBondNum;
      tmp->ORDER_tokens = this->ORDER_tokens;
      boost::intrusive_ptr<Network_base> out(tmp);
      return out;
    }
    ~FermionNetwork(){};
  };

  ///@endcond

  /// @brief the Network object for easy build tensor network.
  // wrapper
  class Network {
   public:
    ///@cond
    boost::intrusive_ptr<Network_base> _impl;
    Network() : _impl(new Network_base()){};
    Network(const Network &rhs) { this->_impl = rhs._impl; }
    Network &operator=(const Network &rhs) {
      this->_impl = rhs._impl;
      return *this;
    }
    ///@endcond

    /**
    @brief Construct Network from network file.
    @param fname The network file path
    @param network_type The type of network.
           This can be [NtType.Regular] or [NtType.Fermion.].
           Currently, only Regular Network is support!


    ##note:
        1. each network file cannot have more than 1024 lines.

    */
    void Fromfile(const std::string &fname, const int &network_type = NtType.Regular) {
      if (network_type == NtType.Regular) {
        boost::intrusive_ptr<Network_base> tmp(new RegularNetwork());
        this->_impl = tmp;
      } else {
        cytnx_error_msg(true, "[Developing] currently only support regular type network.%s", "\n");
      }
      this->_impl->Fromfile(fname);
    }

    Network(const std::string &fname, const int &network_type = NtType.Regular) {
      this->Fromfile(fname, network_type);
    }

    void PutUniTensor(const std::string &name, const UniTensor &utensor,
                      const bool &is_clone = true) {
      this->_impl->PutUniTensor(name, utensor, is_clone);
    }
    void PutUniTensor(const cytnx_uint64 &idx, const UniTensor &utensor,
                      const bool &is_clone = true) {
      this->_impl->PutUniTensor(idx, utensor, is_clone);
    }
    void Launch() { this->_impl->Launch(); }
    void Clear() {
      boost::intrusive_ptr<Network_base> tmp(new Network_base());
      this->_impl = tmp;
    }

    Network clone() {
      Network out;
      out._impl = this->_impl->clone();
      return out;
    }
  };

  ///@cond
  // std::ostream& operator<<(std::ostream &os,const Network &bin);
  ///@endcond
}  // namespace cytnx

#endif
