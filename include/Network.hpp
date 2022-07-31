#ifndef _H_Network_
#define _H_Network_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include <initializer_list>
#include <vector>
#include <map>
#include <fstream>
#include "intrusive_ptr_base.hpp"
#include "utils/utils.hpp"
#include "UniTensor.hpp"
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
    // protected:
   public:
    int nwrktype_id;
    std::string filename;
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
    virtual void PutUniTensor(const std::string &name, const UniTensor &utensor);
    virtual void PutUniTensor(const cytnx_uint64 &idx, const UniTensor &utensor);
    virtual void PutUniTensors(const std::vector<std::string> &name,
                               const std::vector<UniTensor> &utensors);
    virtual void Contract_plan(const std::vector<UniTensor> &utensors, const std::string &Tout,
                               const std::vector<std::string> &alias,
                               const std::string &contract_order);

    virtual void Fromfile(const std::string &fname);
    virtual void FromString(const std::vector<std::string> &content);
    virtual void clear();
    virtual std::string getOptimalOrder();
    virtual UniTensor Launch(const bool &optimal = false, const std::string &contract_order = "");
    virtual void PrintNet(std::ostream &os);
    virtual boost::intrusive_ptr<Network_base> clone();
    virtual void Savefile(const std::string &fname);
    virtual ~Network_base(){};

  };  // Network_base

  class RegularNetwork : public Network_base {
   public:
    RegularNetwork() { this->nwrktype_id = NtType.Regular; };
    void Fromfile(const std::string &fname);
    void FromString(const std::vector<std::string> &contents);
    void PutUniTensor(const std::string &name, const UniTensor &utensor);
    void PutUniTensor(const cytnx_uint64 &idx, const UniTensor &utensor);
    void PutUniTensors(const std::vector<std::string> &name,
                       const std::vector<UniTensor> &utensors);
    void Contract_plan(const std::vector<UniTensor> &utensors, const std::string &Tout,
                       const std::vector<std::string> &alias = {},
                       const std::string &contract_order = "");
    void clear() {
      this->name2pos.clear();
      this->CtTree.clear();
      this->names.clear();
      this->iBondNums.clear();
      this->label_arr.clear();
      this->TOUT_labels.clear();
      this->TOUT_iBondNum = 0;
      this->ORDER_tokens.clear();
    }
    std::string getOptimalOrder();
    UniTensor Launch(const bool &optimal = false, const std::string &contract_order = "");
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
    void PrintNet(std::ostream &os);
    void Savefile(const std::string &fname);
    ~RegularNetwork(){};
  };

  // Under dev!!
  class FermionNetwork : public Network_base {
   protected:
    // [Future] Swap gates.

   public:
    FermionNetwork() { this->nwrktype_id = NtType.Fermion; };
    void Fromfile(const std::string &fname){};
    void FromString(const std::vector<std::string> &contents){};
    void PutUniTensor(const std::string &name, const UniTensor &utensor){};
    void PutUniTensor(const cytnx_uint64 &idx, const UniTensor &utensor){};
    void PutUniTensors(const std::vector<std::string> &name,
                       const std::vector<UniTensor> &utensors){};
    void Contract_plan(const std::vector<UniTensor> &utensors, const std::string &Tout,
                       const std::vector<std::string> &alias = {},
                       const std::string &contract_order = ""){};
    void clear() {
      this->name2pos.clear();
      this->CtTree.clear();
      this->names.clear();
      this->iBondNums.clear();
      this->label_arr.clear();
      this->TOUT_labels.clear();
      this->TOUT_iBondNum = 0;
      this->ORDER_tokens.clear();
    }
    UniTensor Launch(const bool &optimal = false) { return UniTensor(); };
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
    void PrintNet(std::ostream &os){};
    void Savefile(const std::string &fname){};
    ~FermionNetwork(){};
  };

  ///@endcond

  /* @brief the Network object for easy build tensor network.

       The Network is an object that allow one to create a complex network from a pre-defined
     Network file. By putting the Tensors into the Network, the user simply call “Network.Launch()”
     to get the out-come.
  */
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

    ##detail:
        Format of a network file:

        - each line defines a UniTensor, that takes the format '[name] : [Labels]'
        - the name can be any alphabets A-Z, a-z
        - There are two reserved name: 'TOUT' and 'ORDER' (all capital)
        - One can use 'TOUT' line to specify the output UniTensor's bond order using labels
        - The 'ORDER' line is used to specify the contraction order

        About [Labels]:

        - each label should seperate by a comma ","
        - one ';' is needed and used to seperate Rowrank and column rank

        About [ORDER]:

        - The contraction order, it can be specify using the standard mathmetical bracket ruls.
        - Without specify this line, the default contraction order will be from the first line to
    the last line


    ##example network file:
        \include example/Network/example.net

    ##example code for load the network file:
    ### c++ API:
    \include example/Network/Fromfile.cpp
    #### output>
    \verbinclude example/Network/Fromfile.cpp.out
    ### python API
    \include example/Network/Fromfile.py
    #### output>
    \verbinclude example/Network/Fromfile.py.out


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

    /**
    @brief Construct Network from a list of strings, where each string is the same as each line in
    network file
    @param contents The network file descriptions
    @param network_type The type of network.
           This can be [NtType.Regular] or [NtType.Fermion.].
           Currently, only Regular Network is support!


    ##note:
        1. contents cannot have more than 1024 lines/strings.

    ##detail:
        Format of each string follows the same policy as Fromfile.


    ##example code for load the network file:
    ### c++ API:
    \include example/Network/FromString.cpp
    #### output>
    \verbinclude example/Network/FromString.cpp.out
    ### python API
    \include example/Network/FromString.py
    #### output>
    \verbinclude example/Network/FromString.py.out

    */
    void FromString(const std::vector<std::string> &contents,
                    const int &network_type = NtType.Regular) {
      if (network_type == NtType.Regular) {
        boost::intrusive_ptr<Network_base> tmp(new RegularNetwork());
        this->_impl = tmp;
      } else {
        cytnx_error_msg(true, "[Developing] currently only support regular type network.%s", "\n");
      }
      this->_impl->FromString(contents);
    }
    // void Savefile(const std::string &fname);

    static Network Contract(const std::vector<UniTensor> &tensors, const std::string &Tout,
                            const std::vector<std::string> &alias = {},
                            const std::string &contract_order = "") {
      boost::intrusive_ptr<Network_base> tmp(new RegularNetwork());
      Network out;
      out._impl = tmp;
      out._impl->Contract_plan(tensors, Tout, alias, contract_order);
      return out;
    }

    Network(const std::string &fname, const int &network_type = NtType.Regular) {
      this->Fromfile(fname, network_type);
    }

    void PutUniTensor(const std::string &name, const UniTensor &utensor) {
      this->_impl->PutUniTensor(name, utensor);
    }
    void PutUniTensor(const cytnx_uint64 &idx, const UniTensor &utensor) {
      this->_impl->PutUniTensor(idx, utensor);
    }
    void PutUniTensors(const std::vector<std::string> &name,
                       const std::vector<UniTensor> &utensors) {
      this->_impl->PutUniTensors(name, utensors);
    }
    UniTensor Launch(const bool &optimal = false) { return this->_impl->Launch(optimal); }
    void clear() {
      // boost::intrusive_ptr<Network_base> tmp(new Network_base());
      this->_impl->clear();
    }

    Network clone() {
      Network out;
      out._impl = this->_impl->clone();
      return out;
    }
    void PrintNet() { this->_impl->PrintNet(std::cout); }

    void Savefile(const std::string &fname) { this->_impl->Savefile(fname); }
  };

  ///@cond
  std::ostream &operator<<(std::ostream &os, const Network &bin);
  ///@endcond
}  // namespace cytnx

#endif