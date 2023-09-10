#ifndef _H_Gncon_
#define _H_Gncon_

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

#ifdef BACKEND_TORCH
#else
namespace cytnx {
  /// @cond
  // struct __ntwk {
  //   enum __GNType { Void = -1, Regular = 0, Fermion = 1 };
  // };
  class GnconType_class {
   public:
    enum : int { Void = -1, Regular = 0, Fermion = 1 };
    std::string getname(const int &nwrktype_id);
  };
  extern GnconType_class GNType;
  /// @endcond

  /// @cond
  class Gncon_base : public intrusive_ptr_base<Gncon_base> {
    // protected:
   public:
    int nwrktype_id;
    std::string filename;
    // std::vector<std::vector<string>> tasks;  // ex: "A:a-B:b",... => [["A","a","B","b"],...]

    std::vector<UniTensor> tensors;
    std::vector<cytnx_int64> TOUT_labels;

    std::vector<std::vector<std::pair<std::string, std::string>>>
      table;  // table[i] =  i-th tensor's leg names to be contracted, and its target label.

    cytnx_uint64 TOUT_iBondNum;

    // bool ordered;

    // Contraction order.
    ContractionTree CtTree;
    std::vector<std::string> ORDER_tokens;

    // labels corr to the tn list.
    std::vector<std::vector<std::string>> label_arr;
    std::vector<cytnx_int64> iBondNums;

    // name of tn.
    std::vector<std::string> names;
    std::map<std::string, cytnx_uint64> name2pos;

    friend class FermionGncon;
    friend class RegularGncon;
    friend class Gncon;
    Gncon_base() : nwrktype_id(GNType.Void){};

    bool HasPutAllUniTensor() {
      for (cytnx_uint64 i = 0; i < this->tensors.size(); i++) {
        if (this->tensors[i].uten_type() == UTenType.Void) return false;
      }
      return true;
    }

    // void print_Gncon() const;

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
    virtual boost::intrusive_ptr<Gncon_base> clone();
    virtual void Savefile(const std::string &fname);
    virtual ~Gncon_base(){};

  };  // Gncon_base

  class RegularGncon : public Gncon_base {
   public:
    RegularGncon() { this->nwrktype_id = GNType.Regular; };
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
    boost::intrusive_ptr<Gncon_base> clone() {
      RegularGncon *tmp = new RegularGncon();
      tmp->name2pos = this->name2pos;
      tmp->CtTree = this->CtTree;
      tmp->names = this->names;
      tmp->iBondNums = this->iBondNums;
      tmp->label_arr = this->label_arr;
      tmp->TOUT_labels = this->TOUT_labels;
      tmp->TOUT_iBondNum = this->TOUT_iBondNum;
      tmp->ORDER_tokens = this->ORDER_tokens;
      boost::intrusive_ptr<Gncon_base> out(tmp);
      return out;
    }
    void PrintNet(std::ostream &os);
    void Savefile(const std::string &fname);
    ~RegularGncon(){};
  };

  // Under dev!!
  class FermionGncon : public Gncon_base {
   protected:
    // [Future] Swap gates.

   public:
    FermionGncon() { this->nwrktype_id = GNType.Fermion; };
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
    boost::intrusive_ptr<Gncon_base> clone() {
      FermionGncon *tmp = new FermionGncon();
      tmp->name2pos = this->name2pos;
      tmp->CtTree = this->CtTree;
      tmp->names = this->names;
      tmp->iBondNums = this->iBondNums;
      tmp->label_arr = this->label_arr;
      tmp->TOUT_labels = this->TOUT_labels;
      tmp->TOUT_iBondNum = this->TOUT_iBondNum;
      tmp->ORDER_tokens = this->ORDER_tokens;
      boost::intrusive_ptr<Gncon_base> out(tmp);
      return out;
    }
    void PrintNet(std::ostream &os){};
    void Savefile(const std::string &fname){};
    ~FermionGncon(){};
  };

  ///@endcond

  /* @brief the Gncon object for easy build tensor Gncon.

       The Gncon is an object that allow one to create a complex Gncon from a pre-defined
     Gncon file. By putting the Tensors into the Gncon, the user simply call “Gncon.Launch()”
     to get the out-come.
  */
  class Gncon {
   public:
    ///@cond
    boost::intrusive_ptr<Gncon_base> _impl;
    Gncon() : _impl(new Gncon_base()){};
    Gncon(const Gncon &rhs) { this->_impl = rhs._impl; }
    Gncon &operator=(const Gncon &rhs) {
      this->_impl = rhs._impl;
      return *this;
    }
    ///@endcond

    /**
    @brief Construct Gncon from Gncon file.
    @param fname The Gncon file path
    @param Gncon_type The type of Gncon.
           This can be [GNType.Regular] or [GNType.Fermion.].
           Currently, only Regular Gncon is support!


    ##note:
        1. each Gncon file cannot have more than 1024 lines.

    ##detail:
        Format of a Gncon file:

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


    ##example Gncon file:
        \include example/Gncon/example.net

    ##example code for load the Gncon file:
    ### c++ API:
    \include example/Gncon/Fromfile.cpp
    #### output>
    \verbinclude example/Gncon/Fromfile.cpp.out
    ### python API
    \include example/Gncon/Fromfile.py
    #### output>
    \verbinclude example/Gncon/Fromfile.py.out


    */
    void Fromfile(const std::string &fname, const int &Gncon_type = GNType.Regular) {
      if (Gncon_type == GNType.Regular) {
        boost::intrusive_ptr<Gncon_base> tmp(new RegularGncon());
        this->_impl = tmp;
      } else {
        cytnx_error_msg(true, "[Developing] currently only support regular type Gncon.%s", "\n");
      }
      this->_impl->Fromfile(fname);
    }

    /**
    @brief Construct Gncon from a list of strings, where each string is the same as each line in
    Gncon file
    @param contents The Gncon file descriptions
    @param Gncon_type The type of Gncon.
           This can be [GNType.Regular] or [GNType.Fermion.].
           Currently, only Regular Gncon is support!


    ##note:
        1. contents cannot have more than 1024 lines/strings.

    ##detail:
        Format of each string follows the same policy as Fromfile.


    ##example code for load the Gncon file:
    ### c++ API:
    \include example/Gncon/FromString.cpp
    #### output>
    \verbinclude example/Gncon/FromString.cpp.out
    ### python API
    \include example/Gncon/FromString.py
    #### output>
    \verbinclude example/Gncon/FromString.py.out

    */
    void FromString(const std::vector<std::string> &contents,
                    const int &Gncon_type = GNType.Regular) {
      if (Gncon_type == GNType.Regular) {
        boost::intrusive_ptr<Gncon_base> tmp(new RegularGncon());
        this->_impl = tmp;
      } else {
        cytnx_error_msg(true, "[Developing] currently only support regular type Gncon.%s", "\n");
      }
      this->_impl->FromString(contents);
    }
    // void Savefile(const std::string &fname);

    static Gncon Contract(const std::vector<UniTensor> &tensors, const std::string &Tout,
                          const std::vector<std::string> &alias = {},
                          const std::string &contract_order = "") {
      boost::intrusive_ptr<Gncon_base> tmp(new RegularGncon());
      Gncon out;
      out._impl = tmp;
      out._impl->Contract_plan(tensors, Tout, alias, contract_order);
      return out;
    }

    Gncon(const std::string &fname, const int &Gncon_type = GNType.Regular) {
      this->Fromfile(fname, Gncon_type);
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
    std::string getOptimalOrder(const int &Gncon_type = GNType.Regular) {
      if (Gncon_type == GNType.Regular) {
        return this->_impl->getOptimalOrder();
      } else {
        cytnx_error_msg(true, "[Developing] currently only support regular type Gncon.%s", "\n");
      }
    }
    UniTensor Launch(const bool &optimal, const std::string &contract_order = "",
                     const int &Gncon_type = GNType.Regular) {
      if (Gncon_type == GNType.Regular) {
        return this->_impl->Launch(optimal);
      } else {
        cytnx_error_msg(true, "[Developing] currently only support regular type Gncon.%s", "\n");
      }
    }
    void clear() {
      // boost::intrusive_ptr<Gncon_base> tmp(new Gncon_base());
      this->_impl->clear();
    }

    Gncon clone() {
      Gncon out;
      out._impl = this->_impl->clone();
      return out;
    }
    void PrintNet() { this->_impl->PrintNet(std::cout); }

    void Savefile(const std::string &fname) { this->_impl->Savefile(fname); }
  };

  ///@cond
  std::ostream &operator<<(std::ostream &os, const Gncon &bin);
  ///@endcond
}  // namespace cytnx

#endif  // BACKEND_TORCH

#endif
