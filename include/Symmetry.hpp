#ifndef CYTNX_SYMMETRY_H_
#define CYTNX_SYMMETRY_H_

#include <fstream>
#include <ostream>
#include <string>
#include <vector>

#include "boost/smart_ptr/intrusive_ptr.hpp"

#include "cytnx_error.hpp"
#include "intrusive_ptr_base.hpp"
#include "Type.hpp"
#include "utils/dynamic_arg_resolver.hpp"

namespace cytnx {
  /**
   * @brief Symmetry type.
   * @details It is about the type of the Symmetry object
   *     The supported enumerations are as following:
   *
   *  enumeration  |  description
   * --------------|--------------------
   *  Void         |  -99, void type (that means not initialized)
   *  U            |  -1, U1 symmetry
   *  Z            |   0, Zn symmetry
   *  fPar         |  -2, fermionParity symmetry
   *  fNum         |  -3, fermionNumber symmetry
   *
   *  @see Symmetry::stype(), Symmetry::stype_str()
   */
  enum SymmetryType : int { Void = -99, U = -1, Z = 0, fPar = -2, fNum = -3 };

  /**
   * @brief fermionParity
   * @details the parity of fermionis
   * EVEN For bosons or an even number of fermions
   * ODD  For an even number of fermions
   */
  enum fermionParity : bool { EVEN = false, ODD = true };

  // helper class, has implicitly conversion to vector<int64>!
  class Qs {
   private:
    std::vector<cytnx_int64> tmpQs;

   public:
    template <class... Ts>
    Qs(const cytnx_int64 &e1, const Ts... elems) {
      this->tmpQs = dynamic_arg_int64_resolver(e1, elems...);
    }

    Qs(const std::vector<cytnx_int64> &qin) { this->tmpQs = qin; }

    // interprete as 2d vector directly implicitly convert!
    explicit operator std::vector<cytnx_int64>() const { return this->tmpQs; };

    std::pair<std::vector<cytnx_int64>, cytnx_uint64> operator>>(const cytnx_uint64 &dim) {
      return std::make_pair(this->tmpQs, dim);
    }
  };

  ///@cond
  class Symmetry_base : public intrusive_ptr_base<Symmetry_base> {
   public:
    int stype_id;
    int n;
    Symmetry_base() : stype_id(SymmetryType::Void){};
    Symmetry_base(const int &n) : stype_id(SymmetryType::Void) { this->Init(n); };
    Symmetry_base(const Symmetry_base &rhs);
    Symmetry_base &operator=(const Symmetry_base &rhs);

    std::vector<cytnx_int64> combine_rule(const std::vector<cytnx_int64> &inL,
                                          const std::vector<cytnx_int64> &inR);
    cytnx_int64 combine_rule(const cytnx_int64 &inL, const cytnx_int64 &inR,
                             const bool &is_reverse);

    cytnx_int64 reverse_rule(const cytnx_int64 &in);

    virtual void Init(const int &n){};
    virtual boost::intrusive_ptr<Symmetry_base> clone() { return nullptr; };
    virtual bool check_qnum(
      const cytnx_int64 &in_qnum);  // check the passed in qnums satisfy the symmetry requirement.
    virtual bool check_qnums(const std::vector<cytnx_int64> &in_qnums);
    virtual void combine_rule_(std::vector<cytnx_int64> &out, const std::vector<cytnx_int64> &inL,
                               const std::vector<cytnx_int64> &inR);
    virtual void combine_rule_(cytnx_int64 &out, const cytnx_int64 &inL, const cytnx_int64 &inR,
                               const bool &is_reverse);
    virtual void reverse_rule_(cytnx_int64 &out, const cytnx_int64 &in);
    virtual fermionParity get_fermion_parity(const cytnx_int64 &in_qnum) const;
    virtual bool is_fermionic() const { return false; };

    virtual void print_info() const;
    virtual std::string stype_str() const;
    // virtual std::vector<cytnx_int64>& combine_rule(const std::vector<cytnx_int64> &inL, const
    // std::vector<cytnx_int64> &inR);
  };
  ///@endcond

  ///@cond
  class U1Symmetry : public Symmetry_base {
   public:
    U1Symmetry() { this->stype_id = SymmetryType::U; };
    U1Symmetry(const int &n) { this->Init(n); };
    void Init(const int &n) {
      this->stype_id = SymmetryType::U;
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
    void combine_rule_(cytnx_int64 &out, const cytnx_int64 &inL, const cytnx_int64 &inR,
                       const bool &is_reverse);
    void reverse_rule_(cytnx_int64 &out, const cytnx_int64 &in);
    void print_info() const;
    std::string stype_str() const override { return "U1"; };
  };
  ///@endcond

  ///@cond
  class ZnSymmetry : public Symmetry_base {
   public:
    ZnSymmetry() { this->stype_id = SymmetryType::Z; };
    ZnSymmetry(const int &n) { this->Init(n); };
    void Init(const int &n) {
      this->stype_id = SymmetryType::Z;
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
    void combine_rule_(cytnx_int64 &out, const cytnx_int64 &inL, const cytnx_int64 &inR,
                       const bool &is_reverse);
    void reverse_rule_(cytnx_int64 &out, const cytnx_int64 &in);
    void print_info() const;
    std::string stype_str() const override { return "Z" + std::to_string(this->n); };
  };
  ///@endcond

  ///@cond
  class FermionParitySymmetry : public Symmetry_base {
   public:
    FermionParitySymmetry() {
      this->stype_id = SymmetryType::fPar;
      this->n = -2;
    };
    boost::intrusive_ptr<Symmetry_base> clone() {
      boost::intrusive_ptr<Symmetry_base> out(new FermionParitySymmetry());
      return out;
    }
    bool check_qnum(const cytnx_int64 &in_qnum);
    bool check_qnums(const std::vector<cytnx_int64> &in_qnums);
    void combine_rule_(std::vector<cytnx_int64> &out, const std::vector<cytnx_int64> &inL,
                       const std::vector<cytnx_int64> &inR);
    void combine_rule_(cytnx_int64 &out, const cytnx_int64 &inL, const cytnx_int64 &inR,
                       const bool &is_reverse);
    void reverse_rule_(cytnx_int64 &out, const cytnx_int64 &in);
    fermionParity get_fermion_parity(const cytnx_int64 &in_qnum) const override;
    bool is_fermionic() const override { return true; };
    void print_info() const;
    std::string stype_str() const override { return "fP"; }
  };
  ///@endcond

  ///@cond
  class FermionNumberSymmetry : public Symmetry_base {
   public:
    FermionNumberSymmetry() {
      this->stype_id = SymmetryType::fNum;
      this->n = -1;
    };
    boost::intrusive_ptr<Symmetry_base> clone() {
      boost::intrusive_ptr<Symmetry_base> out(new FermionNumberSymmetry());
      return out;
    }
    bool check_qnum(const cytnx_int64 &in_qnum);
    bool check_qnums(const std::vector<cytnx_int64> &in_qnums);
    void combine_rule_(std::vector<cytnx_int64> &out, const std::vector<cytnx_int64> &inL,
                       const std::vector<cytnx_int64> &inR);
    void combine_rule_(cytnx_int64 &out, const cytnx_int64 &inL, const cytnx_int64 &inR,
                       const bool &is_reverse);
    void reverse_rule_(cytnx_int64 &out, const cytnx_int64 &in);
    fermionParity get_fermion_parity(const cytnx_int64 &in_qnum) const override;
    bool is_fermionic() const override { return true; };
    void print_info() const;
    std::string stype_str() const override { return "f#"; }
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
      if (stype == SymmetryType::U) {
        boost::intrusive_ptr<Symmetry_base> tmp(new U1Symmetry(1));
        this->_impl = tmp;
      } else if (stype == SymmetryType::Z) {
        boost::intrusive_ptr<Symmetry_base> tmp(new ZnSymmetry(n));
        this->_impl = tmp;
      } else if (stype == SymmetryType::fPar) {
        boost::intrusive_ptr<Symmetry_base> tmp(new FermionParitySymmetry());
        this->_impl = tmp;
      } else if (stype == SymmetryType::fNum) {
        boost::intrusive_ptr<Symmetry_base> tmp(new FermionNumberSymmetry());
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
        create a new U1 symmetry object that serves as a generator.
        The symmetry object is a property of \link cytnx::Bond Bond \endlink. It is used to
    identify the symmetry of the quantum number set, as well as providing the combining rule for
    the quantum number when Bonds are combined.

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
    static Symmetry U1() { return Symmetry(SymmetryType::U, 1); }

    /**
    @brief create a Zn discrete symmetry object with \f$n\in\mathbb{N}\f$

    ###valid qnum value range:

        \f$[0 , n)\f$

    ###combine rule:

        (Q + Q)%n

    ###description:
        create a new Zn discrete symmetry object with integer \f$ n \f$ that serves as a
    generator. The symmetry object is a property of \link cytnx::Bond Bond \endlink. It is used
    to identify the symmetry of the quantum number set, as well as providing the combining rule
    for the quantum number when Bonds are combined.


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
    static Symmetry Zn(const int &n) { return Symmetry(SymmetryType::Z, n); }

    /**
    @brief create a fermionic parity symmetry object

    ###valid qnum value range:

        \f$[0 , 1]\f$
      where 0 corresponds to even parity and 1 to odd parity;

    ###combine rule:

        (Q + Q)%2

    ###description:
        create a new fermionic parity symmetry object that serves as a generator.
        The symmetry object is a property of \link cytnx::Bond Bond \endlink. It is used to
    identify the symmetry of the quantum number set, as well as providing the combining rule for
    the quantum number when Bonds are combined.

    */
    static Symmetry FermionParity() { return Symmetry(SymmetryType::fPar); }

    /**
    @brief create a fermionic occupation number symmetry object

    ###valid qnum value range:

        \f$(-\infty , \infty)\f$

    ###combine rule:

        Q + Q

    ###description:
        create a new fermionic occupation number object that serves as a generator. This is a U1
        symmetry with parity defined as EVEN for even occupation numbers and ODD for odd
    occupation numbers. The symmetry object is a property of \link cytnx::Bond Bond \endlink. It
    is used to identify the symmetry of the quantum number set, as well as providing the
    combining rule for the quantum number when Bonds are combined.

    */
    static Symmetry FermionNumber() { return Symmetry(SymmetryType::fNum); }

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
    @brief return the symmetry type-id of current Symmetry object, see cytnx::SymmetryType::
    @return [int]
        the symmetry type-id.

    */
    int stype() const { return this->_impl->stype_id; }

    /**
    @brief return the discrete n of current Symmetry object.
    @return [int]

    @note
        1. for U1, n=1 will be returned.
        2. for Zn, n is the discrete symmetry number. (ex: Z2, n=2)

    */
    int &n() const { return this->_impl->n; }

    /**
    @brief return the symmetry type name of current Symmetry object in string form, see
    cytnx::SymmetryType::
    @return [std::string]
        the symmetry type name.

    */
    std::string stype_str() const { return this->_impl->stype_str(); }

    /**
    @brief check the quantum number \p qnum is within the valid value range of current Symmetry.
    @param[in] qnum a singule quantum number.
    @return [bool]

    */
    bool check_qnum(const cytnx_int64 &qnum) { return this->_impl->check_qnum(qnum); }

    /**
    @brief check all the quantum numbers \qnums are within the valid value range of current
    Symmetry.
    @param[in] qnums the list of quantum numbers
    @return [bool]

    */
    bool check_qnums(const std::vector<cytnx_int64> &qnums) {
      return this->_impl->check_qnums(qnums);
    }

    /**
    @brief apply combine rule of current symmetry to two quantum number lists.
    @param[in] inL the #1 quantum number list that is to be combined.
    @param[in] inR the #2 quantum number list that is to be combined.
    @return the combined quantum numbers.
    */
    std::vector<cytnx_int64> combine_rule(const std::vector<cytnx_int64> &inL,
                                          const std::vector<cytnx_int64> &inR) {
      return this->_impl->combine_rule(inL, inR);
    }

    /**
    @brief apply combine rule of current symmetry to two quantum number lists, and store it into
    parameter \p out.
    @param[out] out the output quantum number list.
    @param[in] inL the #1 quantum number list that is to be combined.
    @param[in] inR the #2 quantum number list that is to be combined.
    */
    void combine_rule_(std::vector<cytnx_int64> &out, const std::vector<cytnx_int64> &inL,
                       const std::vector<cytnx_int64> &inR) {
      this->_impl->combine_rule_(out, inL, inR);
    }

    /**
    @brief apply combine rule of current symmetry to two quantum numbers.
    @param[in] inL the #1 quantum number.
    @param[in] inR the #2 quantum number.
    @return the combined quantum number.
    */
    cytnx_int64 combine_rule(const cytnx_int64 &inL, const cytnx_int64 &inR,
                             const bool &is_reverse = false) const {
      return this->_impl->combine_rule(inL, inR, is_reverse);
    }

    /**
    @brief apply combine rule of current symmetry to two quantum numbers, and store the combined
    quntun number into parameter \param out.
    @param[out] out the output quantum number.
    @param[in] inL the #1 quantum number.
    @param[in] inR the #2 quantum number.
    */
    void combine_rule_(cytnx_int64 &out, const cytnx_int64 &inL, const cytnx_int64 &inR,
                       const bool &is_reverse = false) {
      this->_impl->combine_rule_(out, inL, inR, is_reverse);
    }

    /**
    @brief Apply reverse rule of current symmetry to a given quantum number and store in
    parameter \p out.
    @details that means, \f$ o = -i \f$, where \f$ o \f$ is the output quantum number \p out,
    and \f$ i \f$ is the input quantum number \p in.
    @param[out] out the output quantum number.
    @param[in] in the input quantum number.
    */
    void reverse_rule_(cytnx_int64 &out, const cytnx_int64 &in) {
      this->_impl->reverse_rule_(out, in);
    }

    /**
    @brief Apply reverse rule of current symmetry to a given quantum number and return the
    result.
    @details that means, \f$ o = -i \f$, where \f$ o \f$ is the reverse quantum number,
    and \f$ i \f$ is the input quantum number \p in.
    @param[in] in the input quantum number.
    @return the reverse quantum number.
    */
    cytnx_int64 reverse_rule(const cytnx_int64 &in) const { return this->_impl->reverse_rule(in); }

    /**
    @brief fermionic parity for a given quantum number
    @param[in] in_qnum quantum number
    @return parity of the quantum number; false for bosonic degree of freedom, true for
    fermionic
    */
    fermionParity get_fermion_parity(const cytnx_int64 &in_qnum) const {
      return this->_impl->get_fermion_parity(in_qnum);
    }

    /**
    @brief check if the Symmetry is fermionic or not
    @return true if Symmetry is fermionic
    */
    bool is_fermionic() const { return this->_impl->is_fermionic(); }

    /**
     * @brief Save the current Symmetry object to a file.
     * @param[in] fname the file name.
     * @post the file extension will be automatically added as ".cysym".
     */
    void Save(const std::string &fname) const;

    /**
     * @brief Same as Save(const std::string &fname) const;
     */
    void Save(const char *fname) const;

    /**
     * @brief Load a Symmetry object from a file.
     * @param[in] fname the file name.
     * @pre the file extension must be ".cysym".
     * @return the loaded Symmetry object.
     */
    static Symmetry Load(const std::string &fname);

    /**
     * @brief Same as static Symmetry Load(const std::string &fname);
     */
    static Symmetry Load(const char *fname);

    /// @cond
    void _Save(std::fstream &f) const;
    void _Load(std::fstream &f);
    /// @endcond

    /**
     * @brief Print the information of current Symmetry object.
     */
    void print_info() const { this->_impl->print_info(); }

    /**
     * @brief the equality operator of the Symmetry object.
     */
    bool operator==(const Symmetry &rhs) const;

    /**
     * @brief the inequality operator of the Symmetry object.
     */
    bool operator!=(const Symmetry &rhs) const;
  };  // Symmetry

  /// @cond
  std::ostream &operator<<(std::ostream &os, const Symmetry &in);
  /// @endcond

};  // namespace cytnx
#endif  // CYTNX_SYMMETRY_H_
