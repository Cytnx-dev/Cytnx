#ifndef _H_Symmetry
#define _H_Symmetry
#include "Type.hpp"
#include "cytnx_error.hpp"
#include "intrusive_ptr_base.hpp"
#include <string>
#include <cstdio>
#include "utils/vec_clone.hpp"
namespace cytnx {

  ///@cond
  struct __sym {
    enum __stype { U = -1, Z = 0 };
  };

  class SymmetryType_class {
   public:
    enum : int {
      Void = -99,
      U = -1,
      Z = 0,
    };
    std::string getname(const int &stype);
  };
  extern SymmetryType_class SymType;
  ///@endcond

  ///@cond
  class Symmetry_base : public intrusive_ptr_base<Symmetry_base> {
   public:
    int stype_id;
    int n;
    Symmetry_base() : stype_id(SymType.Void){};
    Symmetry_base(const int &n) : stype_id(SymType.Void) { this->Init(n); };
    Symmetry_base(const Symmetry_base &rhs);
    Symmetry_base &operator=(const Symmetry_base &rhs);

    std::vector<cytnx_int64> combine_rule(const std::vector<cytnx_int64> &inL,
                                          const std::vector<cytnx_int64> &inR);
    cytnx_int64 combine_rule(const cytnx_int64 &inL, const cytnx_int64 &inR);

    virtual void Init(const int &n){};
    virtual boost::intrusive_ptr<Symmetry_base> clone(){};
    virtual bool check_qnum(
      const cytnx_int64 &in_qnum);  // check the passed in qnums satisfy the symmetry requirement.
    virtual bool check_qnums(const std::vector<cytnx_int64> &in_qnums);
    virtual void combine_rule_(std::vector<cytnx_int64> &out, const std::vector<cytnx_int64> &inL,
                               const std::vector<cytnx_int64> &inR);
    virtual void combine_rule_(cytnx_int64 &out, const cytnx_int64 &inL, const cytnx_int64 &inR);
    // virtual std::vector<cytnx_int64>& combine_rule(const std::vector<cytnx_int64> &inL, const
    // std::vector<cytnx_int64> &inR);
  };
  ///@endcond

  ///@cond
  class U1Symmetry : public Symmetry_base {
   public:
    U1Symmetry() { this->stype_id = SymType.U; };
    U1Symmetry(const int &n) { this->Init(n); };
    void Init(const int &n) {
      this->stype_id = SymType.U;
      this->n = n;
      if (n != 1) cytnx_error_msg(1, "%s", "[ERROR] U1Symmetry should set n = 1");
    }
    boost::intrusive_ptr<Symmetry_base> clone() {
      boost::intrusive_ptr<Symmetry_base> out(new U1Symmetry(this->n));
      return out;
    }
    bool check_qnum(const cytnx_int64 &in_qnum);
    bool check_qnums(const std::vector<cytnx_int64> &in_qnums);
    void combine_rule_(std::vector<cytnx_int64> &out, const std::vector<cytnx_int64> &inL,
                       const std::vector<cytnx_int64> &inR);
    void combine_rule_(cytnx_int64 &out, const cytnx_int64 &inL, const cytnx_int64 &inR);
  };
  ///@endcond

  ///@cond
  class ZnSymmetry : public Symmetry_base {
   public:
    ZnSymmetry() { this->stype_id = SymType.Z; };
    ZnSymmetry(const int &n) { this->Init(n); };
    void Init(const int &n) {
      this->stype_id = SymType.Z;
      this->n = n;
      if (n <= 1) cytnx_error_msg(1, "%s", "[ERROR] ZnSymmetry can only have n > 1");
    }
    boost::intrusive_ptr<Symmetry_base> clone() {
      boost::intrusive_ptr<Symmetry_base> out(new ZnSymmetry(this->n));
      return out;
    }
    bool check_qnum(const cytnx_int64 &in_qnum);
    bool check_qnums(const std::vector<cytnx_int64> &in_qnums);
    void combine_rule_(std::vector<cytnx_int64> &out, const std::vector<cytnx_int64> &inL,
                       const std::vector<cytnx_int64> &inR);
    void combine_rule_(cytnx_int64 &out, const cytnx_int64 &inL, const cytnx_int64 &inR);
  };
  ///@endcond

  //=====================================
  // this is API
  ///@brief the symmetry object
  class Symmetry {
   public:
    //[Note] these two are hide from user.
    ///@cond
    boost::intrusive_ptr<Symmetry_base> _impl;

    Symmetry(const int &stype = -1, const int &n = 0) : _impl(new Symmetry_base()) {
      this->Init(stype, n);
    };  // default is U1Symmetry

    void Init(const int &stype = -1, const int &n = 0) {
      if (stype == SymType.U) {
        boost::intrusive_ptr<Symmetry_base> tmp(new U1Symmetry(1));
        this->_impl = tmp;
      } else if (stype == SymType.Z) {
        boost::intrusive_ptr<Symmetry_base> tmp(new ZnSymmetry(n));
        this->_impl = tmp;
      } else {
        cytnx_error_msg(1, "%s", "[ERROR] invalid symmetry type.");
      }
    }
    Symmetry &operator=(const Symmetry &rhs) {
      this->_impl = rhs._impl;
      return *this;
    }
    Symmetry(const Symmetry &rhs) { this->_impl = rhs._impl; }
    ///@endcond

    //[genenrators]
    /**
    @brief create a U1 symmetry object

    ###valid qnum value range:

        \f$(-\infty , \infty)\f$

    ###combine rule:

        Q + Q

    ###description:
        create a new U1 symmetry object that serive as a generator.
        The symmetry object is a property of \link cytnx::Bond Bond \endlink. It is used to identify
    the symmetry of the quantum number set, as well as providing the combining rule for the quantum
    number when Bonds are combined.

    ## Example:
    ### c++ API:
    \include example/Symmetry/U1.cpp
    #### output>
    \verbinclude example/Symmetry/U1.cpp.out
    ### python API:
    \include example/Symmetry/U1.py
    #### output>
    \verbinclude example/Symmetry/U1.py.out

    */
    static Symmetry U1() { return Symmetry(SymType.U, 1); }

    /**
    @brief create a Zn descrete symmetry object with \f$n\in\mathbb{N}\f$

    ###valid qnum value range:

        \f$[0 , n)\f$

    ###combine rule:

        (Q + Q)%n

    ###description:
        create a new Zn descrete symmetry object with integer \f$ n \f$ that serive as a generator.
        The symmetry object is a property of \link cytnx::Bond Bond \endlink. It is used to identify
    the symmetry of the quantum number set, as well as providing the combining rule for the quantum
    number when Bonds are combined.


    ## Example:
    ### c++ API:
    \include example/Symmetry/Zn.cpp
    #### output>
    \verbinclude example/Symmetry/Zn.cpp.out
    ### python API:
    \include example/Symmetry/Zn.py
    #### output>
    \verbinclude example/Symmetry/Zn.py.out

    */
    static Symmetry Zn(const int &n) { return Symmetry(SymType.Z, n); }

    /**
    @brief return a clone instance of current Symmetry object.
    @return [Symmetry]

    ## Example:
    ### c++ API:
    \include example/Symmetry/clone.cpp
    #### output>
    \verbinclude example/Symmetry/clone.cpp.out
    ### python API:
    \include example/Symmetry/clone.py
    #### output>
    \verbinclude example/Symmetry/clone.py.out
    */
    Symmetry clone() const {
      Symmetry out;
      out._impl = this->_impl->clone();
      return out;
    }

    /**
    @brief return the symmetry type-id of current Symmetry object.
    @return [int]
        the symmetry type-id.

    */
    int stype() const { return this->_impl->stype_id; }

    /**
    @brief return the descrete n of current Symmetry object.
    @return [int]

    ###[Note]
        1. for U1, n=1 will be returned.
        2. for Zn, n is the descrete symmetry number. (ex: Z2, n=2)

    */
    int &n() const { return this->_impl->n; }

    /**
    @brief return the symmetry type name of current Symmetry object.
    @return [std::string]
        the symmetry type name.

    */
    std::string stype_str() const {
      return SymType.getname(this->_impl->stype_id) + std::to_string(this->_impl->n);
    }

    /**
    @brief check the quantum number is within the valid value range of current Symmetry.
    @param qnum a singule quantum number.
    @return [bool]

    */
    bool check_qnum(const cytnx_int64 &qnum) { return this->_impl->check_qnum(qnum); }

    /**
    @brief check all the quantum numbers are within the valid value range of current Symmetry.
    @param qnums the list of quantum numbers
    @return [bool]

    */
    bool check_qnums(const std::vector<cytnx_int64> &qnums) {
      return this->_impl->check_qnums(qnums);
    }

    /**
    @brief apply combine rule of current symmetry to two quantum number lists.
    @param inL the #1 quantum number list that is to be combined.
    @param inR the #2 quantum number list that is to be combined.
    @return the combined quantum numbers.

    */
    std::vector<cytnx_int64> combine_rule(const std::vector<cytnx_int64> &inL,
                                          const std::vector<cytnx_int64> &inR) {
      return this->_impl->combine_rule(inL, inR);
    }

    /**
    @brief apply combine rule of current symmetry to two quantum number lists, and store it into
    parameter \param out.
    @param out the output quantum number list.
    @param inL the #1 quantum number list that is to be combined.
    @param inR the #2 quantum number list that is to be combined.

    */
    void combine_rule_(std::vector<cytnx_int64> &out, const std::vector<cytnx_int64> &inL,
                       const std::vector<cytnx_int64> &inR) {
      this->_impl->combine_rule_(out, inL, inR);
    }

    /**
    @brief apply combine rule of current symmetry to two quantum numbers.
    @param inL the #1 quantum number.
    @param inR the #2 quantum number.
    @return the combined quantum number.

    */
    cytnx_int64 combine_rule(const cytnx_int64 &inL, const cytnx_int64 &inR) {
      return this->_impl->combine_rule(inL, inR);
    }

    /**
    @brief apply combine rule of current symmetry to two quantum numbers, and store the combined
    quntun number into parameter \param out.
    @param out the output quantum number.
    @param inL the #1 quantum number.
    @param inR the #2 quantum number.

    */
    void combine_rule_(cytnx_int64 &out, const cytnx_int64 &inL, const cytnx_int64 &inR) {
      this->_impl->combine_rule_(out, inL, inR);
    }

    bool operator==(const Symmetry &rhs) const;
    bool operator!=(const Symmetry &rhs) const;
  };

}  // namespace cytnx

#endif
