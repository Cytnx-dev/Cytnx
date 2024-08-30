#ifndef _H_Bond_
#define _H_Bond_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include "Symmetry.hpp"
#include <initializer_list>
#include <vector>
#include <fstream>
#include <map>
#include <algorithm>
#include "intrusive_ptr_base.hpp"
#include "utils/vec_clone.hpp"

namespace cytnx {

  /**
   * @brief bond type
   * @details This is about the enumeration of the type of the object Bond.
   * 1. For the UniTensor which is non-symmetry, the corresponding bondType will be set as
   *   bondType.BD_REG defaultly. You also can set the bondType as bondType.BD_KET
   *   or bondType.BD_BRA if you want to give the UniTensor more physics meaning.
   * 2. For the UniTensor is symmetry, the corresponding bondType must be bondType.BD_KET
   *   or bondType.BD_BRA.
   * 3. Please note that if you want to do the contraction for
   *   symmetric UniTensor, you must make sure that the two bonds you want to contract
   *   are one bondType.BD_KET and the other bondType.BD_BRA. Namely, you cannot do the
   *   contraction if two bonds are both bondType.BD_KET or both bondType.BD_BRA.
   * @note currently using gBD_* to indicate this is bond with new qnum structure!
   */
  enum bondType : int {
    BD_KET = -1, /*!< -1, represent ket state in physics */
    BD_BRA = 1, /*!< 1, represent bra state in physics */
    BD_REG = 0, /*!< 0, only can be used in non-symmetry UniTensor */
    BD_NONE = 0, /*!< 0, same as BD_REG */
    BD_IN = -1, /*!< -1, same as BD_KET */
    BD_OUT = 1 /*!< 1, same as BD_BRA */
  };

  /// @cond
  class Bond_impl : public intrusive_ptr_base<Bond_impl> {
   private:
   public:
    friend class Bond;
    cytnx_uint64 _dim;
    bondType _type;
    std::vector<cytnx_uint64> _degs;  // this only works for Qnum
    /*
        [Note], _degs has size only when the Bond is defined with Qnum, deg !!
                Use this size to check if the bond is type-2 (new type)
    */
    std::vector<std::vector<cytnx_int64>> _qnums;  //(dim, # of sym)
    std::vector<Symmetry> _syms;

    Bond_impl() : _dim(0), _type(bondType::BD_REG){};

    void _rm_qnum(const cytnx_uint64 &q_index) {
      // this will not check, so check it before using this internal function!!
      this->_dim -= this->_degs[q_index];
      this->_degs.erase(this->_degs.begin() + q_index);
      this->_qnums.erase(this->_qnums.begin() + q_index);
    }

    void Init(const cytnx_uint64 &dim, const bondType &bd_type = bondType::BD_REG);

    // new added
    void Init(const bondType &bd_type, const std::vector<std::vector<cytnx_int64>> &in_qnums,
              const std::vector<cytnx_uint64> &degs, const std::vector<Symmetry> &in_syms = {});

    bondType type() const { return this->_type; };
    const std::vector<std::vector<cytnx_int64>> &qnums() const { return this->_qnums; }
    std::vector<std::vector<cytnx_int64>> &qnums() { return this->_qnums; }
    const cytnx_uint64 &dim() const { return this->_dim; }
    cytnx_uint32 Nsym() const { return this->_syms.size(); }
    const std::vector<Symmetry> &syms() const { return this->_syms; }
    std::vector<Symmetry> &syms() { return this->_syms; }

    // this is clone return.
    std::vector<std::vector<cytnx_int64>> qnums_clone() const { return this->_qnums; }
    std::vector<Symmetry> syms_clone() const { return vec_clone(this->_syms); }

    bool has_duplicate_qnums() const {
      if (this->_degs.size()) {
        auto tmp = this->_qnums;
        std::sort(tmp.begin(), tmp.end());
        return std::adjacent_find(tmp.begin(), tmp.end()) != tmp.end();
      } else {
        return false;
      }
    }

    void set_type(const bondType &new_bondType) {
      if ((this->_type != BD_REG)) {
        if (new_bondType == BD_REG) {
          cytnx_error_msg(this->_qnums.size(),
                          "[ERROR] cannot change type to BD_REG for a symmetry bond.%s", "\n");
        }
        if (std::abs(int(this->_type)) != std::abs(int(new_bondType))) {
          cytnx_error_msg(this->_qnums.size(),
                          "[ERROR] cannot exchange BDtype between BD_* <-> gBD_* .%s", "\n");
        }
      }

      this->_type = new_bondType;
    }

    void clear_type() {
      if (this->_type != BD_REG) {
        cytnx_error_msg(this->_qnums.size(), "[ERROR] cannot clear type for a symmetry bond.%s",
                        "\n");
      }
      this->_type = bondType::BD_REG;
    }

    boost::intrusive_ptr<Bond_impl> clone() const {
      boost::intrusive_ptr<Bond_impl> out(new Bond_impl());
      out->_dim = this->dim();
      out->_type = this->type();
      out->_qnums = this->qnums_clone();
      out->_syms = this->syms_clone();  // return a clone of vec!
      out->_degs = this->_degs;
      return out;
    }

    // [NOTE] for UniTensor iinternal, we might need to return the QNpool (unordered map for further
    // info on block arrangement!)
    void combineBond_(const boost::intrusive_ptr<Bond_impl> &bd_in, const bool &is_grp = true);

    boost::intrusive_ptr<Bond_impl> combineBond(const boost::intrusive_ptr<Bond_impl> &bd_in,
                                                const bool &is_grp = true) {
      boost::intrusive_ptr<Bond_impl> out = this->clone();
      out->combineBond_(bd_in, is_grp);
      return out;
    }

    // return a sorted qnums by removing all duplicates, sorted from large to small.
    std::vector<std::vector<cytnx_int64>> getUniqueQnums(std::vector<cytnx_uint64> &counts,
                                                         const bool &return_counts);
    // checked [KHW] ^^
    // return the degeneracy of the specify qnum set.
    cytnx_uint64 getDegeneracy(const std::vector<cytnx_int64> &qnum, const bool &return_indices,
                               std::vector<cytnx_uint64> &indices);

    // return the effective qnums when Bra-Ket mismatch.
    std::vector<std::vector<cytnx_int64>> calc_reverse_qnums();

    std::vector<cytnx_uint64> &getDegeneracies() { return this->_degs; };
    const std::vector<cytnx_uint64> &getDegeneracies() const { return this->_degs; };

    std::vector<cytnx_uint64> group_duplicates_();

    boost::intrusive_ptr<Bond_impl> group_duplicates(std::vector<cytnx_uint64> &mapper) const {
      boost::intrusive_ptr<Bond_impl> out = this->clone();
      mapper = out->group_duplicates_();
      return out;
    }

    void force_combineBond_(const boost::intrusive_ptr<Bond_impl> &bd_in, const bool &is_grp);

  };  // Bond_impl
  ///@endcond

  /**
   * @brief the object contains auxiliary properties for each Tensor rank (bond)
   * @details The Bond object is used to construct the bond of the UniTensor.
   *     1. For non-symmetric UniTensor (regular UniTensor, that means
   *       cytnx::UTenType.Dense, see cytnx::UTenType), the bond type need will be set as
   *       bondType.BD_REG defaultly. And you can set the bond type as bondType.BD_KET or
   *       bondType.BD_BRA if you want to describe the it as ket or bra basis.
   *       For non-symmetric case, you cannot input the quantum numbers and Symmetry object.
   *     2. For symmteric UniTensor (cytnx::UTenType.Block, see cytnx::UTenType), the
   *       bond type need to be set as bondType.BD_KET or bondType.BD_BRA depend on
   *       what physical system you describe. And you should input the quantum numbers
   *       and Symmetry objects.
   */
  class Bond {
   public:
    ///@cond
    boost::intrusive_ptr<Bond_impl> _impl;
    Bond() : _impl(new Bond_impl()){};
    Bond(const Bond &rhs) { this->_impl = rhs._impl; }
    Bond &operator=(const Bond &rhs) {
      this->_impl = rhs._impl;
      return *this;
    }
    ///@endcond

    /**
     * @brief The constructor of the Bond object.
     * @details This function will call \ref
 *  Init(const cytnx_uint64 &dim, const bondType &bd_type)
     *  	 "Init" to do initialization.
     * @param[in] dim the dimenstion of the Bond
     * @param[in] bd_type the type (see \ref bondType) of the bond

 *
     * @see
 *  Init(const cytnx_uint64 &dim, const bondType &bd_type)
     */
    Bond(const cytnx_uint64 &dim, const bondType &bd_type = bondType::BD_REG)
        : _impl(new Bond_impl()) {
      this->_impl->Init(dim, bd_type);
    }

    /**
     * @brief The constructor of the Bond object.
     * @details This function will call \ref
     *  Init(const bondType &bd_type,
     *       const std::vector<std::vector<cytnx_int64>> &in_qnums,
     *       const std::vector<cytnx_uint64> &degs,
     *  	 const std::vector<Symmetry> &in_syms)
     *  	 "Init" to do initialization.
     *
     * @param[in] bd_type the type (see \ref bondType) of the bond
     * @param[in] in_qnums input the quantum numbers of the bond (for symmetry case)
     * @param[in] degs the degrees of freedom of the bond.
     * @param[in] in_syms input the symmetries of the bond (for symmetry case)
     * @attention This function is can only use for symmetry case.
     * @see
     *  Init(const bondType &bd_type,
     *       const std::vector<std::vector<cytnx_int64>> &in_qnums,
     *       const std::vector<cytnx_uint64> &degs,
     *  	 const std::vector<Symmetry> &in_syms)
     */
    Bond(const bondType &bd_type, const std::vector<std::vector<cytnx_int64>> &in_qnums,
         const std::vector<cytnx_uint64> &degs, const std::vector<Symmetry> &in_syms = {})
        : _impl(new Bond_impl()) {
      this->_impl->Init(bd_type, in_qnums, degs, in_syms);
    }

    /**
     * @see
     *   Bond(const bondType &bd_type,
     *        const std::vector<std::vector<cytnx_int64>> &in_qnums,
     *        const std::vector<cytnx_uint64> &degs,
     *        const std::vector<Symmetry> &in_syms)
     */
    Bond(const bondType &bd_type, const std::initializer_list<std::vector<cytnx_int64>> &in_qnums,
         const std::vector<cytnx_uint64> &degs, const std::vector<Symmetry> &in_syms = {})
        : _impl(new Bond_impl()) {
      this->_impl->Init(bd_type, in_qnums, degs, in_syms);
    }

    // this is needed for python binding!
    /**
     * @see
     *   Bond(const bondType &bd_type,
     *        const std::vector<std::vector<cytnx_int64>> &in_qnums,
     *        const std::vector<cytnx_uint64> &degs,
     *        const std::vector<Symmetry> &in_syms)
     */
    Bond(const bondType &bd_type, const std::vector<cytnx::Qs> &in_qnums,
         const std::vector<cytnx_uint64> &degs, const std::vector<Symmetry> &in_syms = {})
        : _impl(new Bond_impl()) {
      vec2d<cytnx_int64> qnums(in_qnums.begin(), in_qnums.end());
      this->_impl->Init(bd_type, qnums, degs, in_syms);
    }

    /**
     * @see
     *   Bond(const bondType &bd_type,
     *        const std::vector<std::vector<cytnx_int64>> &in_qnums,
     *        const std::vector<cytnx_uint64> &degs,
     *        const std::vector<Symmetry> &in_syms)
     */
    Bond(const bondType &bd_type,
         const std::vector<std::pair<std::vector<cytnx_int64>, cytnx_uint64>> &in_qnums_dims,
         const std::vector<Symmetry> &in_syms = {})
        : _impl(new Bond_impl()) {
      this->Init(bd_type, in_qnums_dims, in_syms);
    }

    /**
    @brief init a bond object
    @param[in] dim the dimension of the bond (rank)
    @param[in] bd_type the tag of the bond, it can be BD_BRA, BD_KET as physical tagged; or BD_REG
    as regular bond (rank)


    details:
        1. each bond can be tagged with BD_BRA or BD_KET that represent the bond is defined in Bra
    space or Ket space.

        @pre
            1. \p dim cannot be 0.
      2. The bond can be tagged with bondType.BD_BRA or
                  bondType.BD_KET, or bondType.BD_REG depending on the usage.


    ## Example:
    ### c++ API:
    \include example/Bond/Init.cpp
    #### output>
    \verbinclude example/Bond/Init.cpp.out
    ### python API:
    \include example/Bond/Init.py
    #### output>
    \verbinclude example/Bond/Init.py.out
    */
    void Init(const cytnx_uint64 &dim, const bondType &bd_type = bondType::BD_REG) {
      this->_impl->Init(dim, bd_type);
    }

    /**
    @brief init a bond object
        @details This function is initialization for symmetry bond case.
        1. each bond can be tagged with BD_BRA or BD_KET that represent the bond is
                    defined in Bra space or Ket space.
        2. the bond can have arbitrary multiple symmetries, with the type of each
                    symmetry associate to the qnums are provided with the in_syms.

    @param[in] bd_type the tag of the bond, it can be bondType.BD_BRA, bondType.BD_KET as
            physical tagged and cannot be bondType.BD_REG (regular bond).
    @param[in] in_qnums the quantum number(s) of the bond. it should be a 2d vector with
            shape (# of unique qnum labels, # of symmetry).
    @param[in] degs the degeneracy correspond to each qunatum number sets specified
            in the qnums, the size should match the # of rows of passed-in qnums.
    @param[in] in_syms a vector of symmetry objects of the bond, the size should
            match the # of cols of passed-in qnums. [Note] if qnums are provided, the
                default symmetry type is \link cytnx::Symmetry::U1 Symmetry::U1 \endlink

        @pre
            1. The size of \p degs need to same as the size of \p in_qnums.
                2. \p in_qnums and \p degs cannot be empty.
                3. All sub-vector in \p in_qnums MUST have the same size.
                4. If \p in_syms is provided, the size of \p in_syms MUST same as the size of
                  the sub-vector in \p in_qnums.
                5. \p bd_type cannot be bondType.BD_REG.
                6. The quantum numbers in \p in_qnums and \p in_syms need to be consistent.
                  For example, you cannot set the quntum number 2 in Z(2) symmetry.

        @note
            1. If quantum number(s) are provided (which means the bond is with symmetry)
                  then the bond MUST be tagged with either bondType.BD_BRA or bondType.BD_KET.
        2. If the bond is non-symmetry, then it can be tagged with bondType.BD_BRA
                  or bondType.BD_KET, or bondType.BD_REG depending on the usage.
        3. The "bond dimension" is the sum over all numbers specified in degs.
    */
    void Init(const bondType &bd_type, const std::vector<std::vector<cytnx_int64>> &in_qnums,
              const std::vector<cytnx_uint64> &degs, const std::vector<Symmetry> &in_syms = {}) {
      this->_impl->Init(bd_type, in_qnums, degs, in_syms);
    }

    /**
         @see
     Init(const bondType &bd_type, const std::vector<std::vector<cytnx_int64>> &in_qnums,
          const std::vector<cytnx_uint64> &degs, const std::vector<Symmetry> &in_syms)
         */
    void Init(const bondType &bd_type,
              const std::vector<std::pair<std::vector<cytnx_int64>, cytnx_uint64>> &in_qnums_dims,
              const std::vector<Symmetry> &in_syms = {}) {
      vec2d<cytnx_int64> qnums(in_qnums_dims.size());
      std::vector<cytnx_uint64> degs(in_qnums_dims.size());
      for (int i = 0; i < in_qnums_dims.size(); i++) {
        qnums[i] = in_qnums_dims[i].first;
        degs[i] = in_qnums_dims[i].second;
      }
      this->_impl->Init(bd_type, qnums, degs, in_syms);
    }

    /**
    @brief return the current bond type (see cytnx::bondType).
    @return cytnx::bondType
    */
    bondType type() const { return this->_impl->type(); };

    //@{
    /**
    @brief return the current quantum number set(s) by reference
    @return [2d vector] with shape: (dim, # of Symmetry)
        @note Compare to qnums_clone(), this function return reference.
    */
    const std::vector<std::vector<cytnx_int64>> &qnums() const { return this->_impl->qnums(); };
    /**
        @see qnums() const
    */
    std::vector<std::vector<cytnx_int64>> &qnums() { return this->_impl->qnums(); };
    //@}

    /**
    @brief return the clone (deep copy) of the current quantum number set(s)
    @return [2d vector] with shape: (dim, # of Symmetry)
        @note Compare to qnums(), this function return the clone (deep copy).
    */
    std::vector<std::vector<cytnx_int64>> qnums_clone() const {
      return this->_impl->qnums_clone();
    };

    /**
    @brief return the dimension of the bond
    @return [cytnx_uint64]
    */
    cytnx_uint64 dim() const { return this->_impl->dim(); };

    /**
    @brief return the number of the symmetries
    @return [cytnx_uint32]
    */
    cytnx_uint32 Nsym() const { return this->_impl->syms().size(); };

    //@{
    /**
    @brief return the vector of symmetry objects by reference.
        @note Compare to syms_clone() const, this function return by reference.
    @return [vector of Symmetry]
    */
    const std::vector<Symmetry> &syms() const { return this->_impl->syms(); };
    /**
        @see syms() const
    */
    std::vector<Symmetry> &syms() { return this->_impl->syms(); };
    //@}

    /**
    @brief return copy of the vector of symmetry objects.
        @note Compare to syms() const, this function return the clone (deep copy).
    @return [vector of Symmetry]
    */
    std::vector<Symmetry> syms_clone() const { return this->_impl->syms_clone(); };

    /**
    @brief change the tag-type of the instance Bond
    @param[in] new_bondType the new tag-type, it can be bondType.BD_BRA,
            boncType.BD_KET or bondType.BD_REG. See cytnx::bondType.
        @attention You cannot change the symmetry bond (bondType.BD_BRA or bondType.BD_KET
            to regular type (bondType.BD_REG) except the size of the quantum number is 0.
    */
    Bond &set_type(const bondType &new_bondType) {
      this->_impl->set_type(new_bondType);
      return *this;
    }

    /**
    @brief create a new instance of Bond with type changed to the new tag-type.
    @param[in]  new_bondType the new tag-type, it can be bondType.BD_BRA,
            bondType.BD_KET or bondType.BD_REG. See cytnx::bondType.
        @note This is equivalent to Bond.clone().set_type()
        @attention You cannot change the symmetry bond (bondType.BD_BRA or bondType.BD_KET
            to regular type (bondType.BD_REG) except the size of the quantum number is 0.
        @see clone(), set_type(const bondType & new_bondType)
    */
    Bond retype(const bondType &new_bondType) {
      auto out = this->clone();
      out.set_type(new_bondType);
      return out;
    }

    /**
    @brief create a new instance of Bond with type changed in between
            bondType.BD_BRA / bondType.BD_KET.
    */
    Bond redirect() const {
      auto out = this->clone();
      out.set_type(bondType(int(out.type()) * -1));
      return out;
    }

    /**
    @brief Change the bond type between bondType.BD_BRA and bondType.BD_KET
            in the Bond.
    */
    Bond &redirect_() {
      this->set_type(bondType(int(this->type()) * -1));
      return *this;
    }

    /**
    @brief change the tag-type to the default value bondType.BD_REG.
        @pre The size of quantum number should be 0. Namely, you cannot clear the
            symmetric bond.
    */
    void clear_type() { this->_impl->clear_type(); }

    /**
    @brief return a copy of the instance Bond
    @return [Bond] a new instance of Bond that have the same contents

    ## Example:
    ### c++ API:
    \include example/Bond/clone.cpp
    #### output>
    \verbinclude example/Bond/clone.cpp.out
    ### python API:
    \include example/Bond/clone.py
    #### output>
    \verbinclude example/Bond/clone.py.out
    */
    Bond clone() const {
      Bond out;
      out._impl = this->_impl->clone();
      return out;
    }

    /**
    @brief Combine the input bond with self, inplacely.
    @param[in] bd_in the bond that to be combined with self.
          @param[in] is_grp this parameter is only used when the bond is
      symmetric bond (bondType.BD_BRA or bondType.BD_KET).
      If is_grp is true, the basis with duplicated quantum number will be
      grouped together as a single basis. See
      group_duplicates(std::vector<cytnx_uint64> &mapper) const.
          @pre
            1. The type of two bonds (see cytnx::bondType) need to be same.
            2. The Symmetry of two bonds should be same.
          @note Compare to \n
      combineBond(const Bond &bd_in, const bool &is_grp) const, \n
            this function in inplace function.
          @see combineBond(const Bond &bd_in, const bool &is_grp)const

    ## Example:
    ### c++ API:
    \include example/Bond/combineBondinplace.cpp
    #### output>
    \verbinclude example/Bond/combineBondinplace.cpp.out
    ### python API:
    \include example/Bond/combineBondinplace.py
    #### output>
    \verbinclude example/Bond/combineBondinplace.py.out
    */
    void combineBond_(const Bond &bd_in, const bool &is_grp = true) {
      this->_impl->combineBond_(bd_in._impl, is_grp);
    }

    /**
    @brief combine the input bond with self, and return a new combined Bond instance.
    @param[in] bd_in the bond that to be combined.
          @param[in] is_grp this parameter is only used when the bond is
      symmetric bond (bondType.BD_BRA or bondType.BD_KET).
      If is_grp is true, the basis with duplicated quantum number will be
      grouped together as a single basis. See
      group_duplicates(std::vector<cytnx_uint64> &mapper) const.
    @return [Bond] a new combined bond instance.
          @pre
            1. The type of two bonds (see cytnx::bondType) need to be same.
            2. The Symmetry of two bonds should be same.
          @note Compare to \n
      combineBond_(const Bond &bd_in, const bool &is_grp), \n
            this function will create a new Bond object.
          @see combineBond_(const Bond &bd_in, const bool &is_grp)

    ## Example:
    ### c++ API:
    \include example/Bond/combineBond.cpp
    #### output>
    \verbinclude example/Bond/combineBond.cpp.out
    ### python API:
    \include example/Bond/combineBond.py
    #### output>
    \verbinclude example/Bond/combineBond.py.out
    */
    Bond combineBond(const Bond &bd_in, const bool &is_grp = true) const {
      Bond out;
      out._impl = this->_impl->combineBond(bd_in._impl, is_grp);
      return out;
    }

    /**
    @brief combine multiple input bonds with self, and return a new combined Bond instance.
    @param[in] bds the bonds that to be combined with self.
          @param[in] is_grp this parameter is only used when the bond is
      symmetric bond (bondType.BD_BRA or bondType.BD_KET).
      If is_grp is true, the basis with duplicated quantum number will be
      grouped together as a single basis. See
      group_duplicates(std::vector<cytnx_uint64> &mapper) const.
    @return [Bond] a new combined bond instance.
          @pre
            1. The type of all bonds (see cytnx::bondType) need to be same.
            2. The Symmetry of all bonds should be same.
          @note Compare to \n
      combineBond_(const std::vector<Bond> &bds, const bool &is_grp),\n
            this function will create a new Bond object.
    @see combineBond_(const std::vector<Bond> &bds, const bool &is_grp)

    ## Example:
    ### c++ API:
    \include example/Bond/combineBond.cpp
    #### output>
    \verbinclude example/Bond/combineBond.cpp.out
    ### python API:
    \include example/Bond/combineBond.py
    #### output>
    \verbinclude example/Bond/combineBond.py.out
    */
    Bond combineBond(const std::vector<Bond> &bds, const bool &is_grp = true) {
      Bond out = this->clone();
      for (cytnx_uint64 i = 0; i < bds.size(); i++) {
        out.combineBond_(bds[i], is_grp);
      }
      return out;
    }

    /**
    @brief combine multiple input bonds with self, inplacely
    @param[in] bds the bonds that to be combined with self.
          @param[in] is_grp this parameter is only used when the bond is
      symmetric bond (bondType.BD_BRA or bondType.BD_KET).
      If is_grp is true, the basis with duplicated quantum number will be
      grouped together as a single basis. See
      group_duplicates(std::vector<cytnx_uint64> &mapper) const.
        @pre
          1. The type of all bonds (see cytnx::bondType) need to be same.
          2. The Symmetry of all bonds should be same.
        @note Compare to \n
      combineBond(const std::vector<Bond> &bds, const bool &is_grp),\n
          this function will create a new Bond object.
    @see combineBond(const std::vector<Bond> &bds, const bool &is_grp)

    ## Example:
    ### c++ API:
    \include example/Bond/combineBond_.cpp
    #### output>
    \verbinclude example/Bond/combineBond_.cpp.out
    ### python API:
    \include example/Bond/combineBond_.py
    #### output>
    \verbinclude example/Bond/combineBond_.py.out
    */
    void combineBond_(const std::vector<Bond> &bds, const bool &is_grp = true) {
      for (cytnx_uint64 i = 0; i < bds.size(); i++) {
        this->combineBond_(bds[i], is_grp);
      }
    }

    /**
    @deprecated
    @brief combine multiple input bonds with self, and return a new combined Bond instance.
    @param[in] bds the bonds that to be combined with self.
          @param[in] is_grp this parameter is only used when the bond is
      symmetric bond (bondType.BD_BRA or bondType.BD_KET).
      If is_grp is true, the basis with duplicated quantum number will be
      grouped together as a single basis. See
      group_duplicates(std::vector<cytnx_uint64> &mapper) const.
    @return [Bond] a new combined bond instance.
          @pre
            1. The type of all bonds (see cytnx::bondType) need to be same.
            2. The Symmetry of all bonds should be same.
          @note Compare to \n
      combineBonds_(const std::vector<Bond> &bds, const bool &is_grp),\n
            this function will create a new Bond object.
    @see combineBonds_(const std::vector<Bond> &bds, const bool &is_grp)

    ## Example:
    ### c++ API:
    \include example/Bond/combineBonds.cpp
    #### output>
    \verbinclude example/Bond/combineBonds.cpp.out
    ### python API:
    \include example/Bond/combineBonds.py
    #### output>
    \verbinclude example/Bond/combineBonds.py.out
    */
    Bond combineBonds(const std::vector<Bond> &bds, const bool &is_grp = true) {
      Bond out = this->clone();
      for (cytnx_uint64 i = 0; i < bds.size(); i++) {
        out.combineBond_(bds[i], is_grp);
      }
      return out;
    }

    /**
    @deprecated
    @brief combine multiple input bonds with self, inplacely
    @param[in] bds the bonds that to be combined with self.
          @param[in] is_grp this parameter is only used when the bond is
      symmetric bond (bondType.BD_BRA or bondType.BD_KET).
      If is_grp is true, the basis with duplicated quantum number will be
      grouped together as a single basis. See
      group_duplicates(std::vector<cytnx_uint64> &mapper) const.
        @pre
          1. The type of all bonds (see cytnx::bondType) need to be same.
          2. The Symmetry of all bonds should be same.
        @note Compare to \n
      combineBonds(const std::vector<Bond> &bds, const bool &is_grp),\n
          this function will create a new Bond object.
    @see combineBonds(const std::vector<Bond> &bds, const bool &is_grp)

    ## Example:
    ### c++ API:
    \include example/Bond/combineBonds_.cpp
    #### output>
    \verbinclude example/Bond/combineBonds_.cpp.out
    ### python API:
    \include example/Bond/combineBonds_.py
    #### output>
    \verbinclude example/Bond/combineBonds_.py.out
    */
    void combineBonds_(const std::vector<Bond> &bds, const bool &is_grp = true) {
      for (cytnx_uint64 i = 0; i < bds.size(); i++) {
        this->combineBond_(bds[i], is_grp);
      }
    }

    /**
    @brief return a sorted qnum sets by removing all the duplicate qnum sets.
    @param[out] counts output the counts of the unique quantum numbers.
        @pre The bond cannot be regular type (namely, bondType.BD_REG)
    @return std::vector<std::vector<cytnx_int64>> unique_qnums
        @see getUniqueQnums()
    */
    std::vector<std::vector<cytnx_int64>> getUniqueQnums(std::vector<cytnx_uint64> &counts) {
      return this->_impl->getUniqueQnums(counts, true);
    }

    /**
    @brief return a sorted qnum sets by removing all the duplicate qnum sets.
        @pre The bond cannot be regular type (namely, bondType.BD_REG)
    @return std::vector<std::vector<cytnx_int64>> unique_qnums
        @see getUniqueQnums(std::vector<cytnx_uint64> &counts)
    */
    std::vector<std::vector<cytnx_int64>> getUniqueQnums() {
      std::vector<cytnx_uint64> tmp;
      return this->_impl->getUniqueQnums(tmp, false);
    }

    /**
    @brief return the degeneracy of specify qnum set.
        @param[in] qnum input the quantum number you want to get the degeneracy.
        @pre the input \p qnum should match the number of the Symmetry. (see Nsym())
    @return cytnx_uint64 degeneracy

        @note
        if the bond has no symmetries, return 0.
        if the degeneracy queried does not exist, return 0, and the indicies is empty
        @see
         getDegeneracy(const std::vector<cytnx_int64> &qnum,
                       std::vector<cytnx_uint64> &indices) const.
    */
    cytnx_uint64 getDegeneracy(const std::vector<cytnx_int64> &qnum) const {
      std::vector<cytnx_uint64> tmp;
      return this->_impl->getDegeneracy(qnum, false, tmp);
    }

    /**
    @brief return the degeneracy of specify qnum set.
        @param[in] qnum input the quantum number you want to get the degeneracy.
        @param[out] indices output the indices location of the quantum number \p qnum.
        @pre the input \p qnum should match the number of the Symmetry. (see Nsym())
    @return cytnx_uint64 degeneracy

        @note
        If the bond has no symmetries, return 0.
        If the degeneracy queried does not exist, return 0, and the indicies is empty.
        @see
         getDegeneracy(const std::vector<cytnx_int64> &qnum) const.
    */
    cytnx_uint64 getDegeneracy(const std::vector<cytnx_int64> &qnum,
                               std::vector<cytnx_uint64> &indices) const {
      indices.clear();
      return this->_impl->getDegeneracy(qnum, true, indices);
    }

    /**
    @brief return all degeneracies.
    @return std::vector<cytnx_uint64> degeneracy
        @see getDegeneracies() const,
         getDegeneracy(const std::vector<cytnx_int64> &qnum) const,
         getDegeneracy(const std::vector<cytnx_int64> &qnum,
                       std::vector<cytnx_uint64> &indices) const.
    */
    std::vector<cytnx_uint64> &getDegeneracies() { return this->_impl->getDegeneracies(); }

    /**
        @see getDegeneracies()
    */
    const std::vector<cytnx_uint64> &getDegeneracies() const {
      return this->_impl->getDegeneracies();
    }

    /**
    @brief Group the duplicated quantum number, inplacely.
        @details This function will group the duplicated quantum number and return the
            mapper, where mapper is about the new index from old index via\n
        new_index = return<cytnx_uint64>[old_index].
        @pre The Bond need to be symmetric type (namely, bondType should be
            bondType.BD_BRA or bondType.BD_DET, see \ref bondType.)
        @note
            1. Compare to the function
              group_duplicates(std::vector<cytnx_uint64> &mapper) const,
                  this function is inplace function.
                2. This function will sort ascending of the quantum number.
        @see group_duplicates(std::vector<cytnx_uint64> &mapper) const,
            has_duplicate_qnums() const.
    @return std::vector<cytnx_uint64> mapper
    */
    std::vector<cytnx_uint64> group_duplicates_() { return this->_impl->group_duplicates_(); }

    /**
    @brief Group the duplicated quantum number and return the new instance
            of the Bond ojbect.
        @details This function will group the duplicated quantum number and return
            the new instance of the Bond object. It will also the \p mapper, where
                \p mapper is about the new index from old index via\n
        new_index = return<cytnx_uint64>[old_index].
        @param[out] mapper the new index from old index via\n
        new_index = return<cytnx_uint64>[old_index].
        @pre The Bond need to be symmetric type (namely, bondType should be
            bondType.BD_BRA or bondType.BD_DET, see \ref bondType.)
        @note
            1. Compare to the function
              group_duplicates_(), this function will create the new instance of
                  Bond object.
                2. This function will sort ascending of the quantum number.
        @see group_duplicates_(std::vector<cytnx_uint64> &mapper),
            has_duplicate_qnums() const.
    @return Bond
    */
    Bond group_duplicates(std::vector<cytnx_uint64> &mapper) const {
      Bond out;
      out._impl = this->_impl->group_duplicates(mapper);
      return out;
    }

    /**
    @brief Check whether there is duplicated quantum numbers in the Bond.
        @details This function will check whether there is any duplicated quantum number
            is the Bond. If yes, return ture. Otherwise, return false.
        @note For the regular bond (bondType.BD_REG), it will always return false.
        @see
            group_duplicates_(std::vector<cytnx_uint64> &mapper)
            group_duplicates(std::vector<cytnx_uint64> &mapper) const
    @return bool
    */
    bool has_duplicate_qnums() const { return this->_impl->has_duplicate_qnums(); }

    /**
    @brief Calculate the reverse of the quantum numbers.
        @details This function will calculate the reverse of the qunatum numbers by the
            reverse rule of the Symmetry. See Symmetry.reverse_rule().
        @note You may use this function if the Bra-Ket mismatch.
        @see Symmetry.reverse_rule()
    @return std::vector<std::vector<cytnx_int64>>
    */
    std::vector<std::vector<cytnx_int64>> calc_reverse_qnums() {
      return this->_impl->calc_reverse_qnums();
    }

    /**
    @brief Save the Bond object to the file.
    @details Save the Bond object to the file. The file extension will be automatically
      added as ".cybd".
          @param[in] fname the file name of the Bond object (exclude the file extension).
    @see Load(const std::string &fname)
    */
    void Save(const std::string &fname) const;

    /**
        @see Save(const std::string &fname) const;
    */
    void Save(const char *fname) const;

    /**
    @brief Load the Bond object from the file.
          @param[in] fname the file name of the Bond object.
          @pre The file need to be the file of Bond object, which is saved by the
      function Bond::Save(const std::string &fname) const.
    */
    static cytnx::Bond Load(const std::string &fname);

    /**
    @see Load(const std::string &fname)
    */
    static cytnx::Bond Load(const char *fname);

    /// @cond
    void _Save(std::fstream &f) const;
    void _Load(std::fstream &f);
    /// @endcond

    /**
    @brief The comparison operator 'equal to'.
    @details The comparison operators to compare two Bonds. If two Bond object are
          same, return true. Otherwise, return false. This equal to operator will
    compare all the "value" of the Bond object. Even the Bond object are different
    object (different address), but they are same "value", it will return true.
          @see operator!=(const Bond &rhs) const
    */
    bool operator==(const Bond &rhs) const;

    /**
    @brief The comparison operator 'not equal to'.
    @details The comparison operators to compare two Bonds. More precisely, it is
    the opposite result of the operator==(const Bond &rhs) const.
          @see operator==(const Bond &rhs) const
    */
    bool operator!=(const Bond &rhs) const;

    /**
    @brief The multiplication operator of the Bond object.
    @details The multiplication operator of the Bond means that Combine two Bond.
            So this operator is same as
                \ref combineBond(const Bond &bd_in, const bool &is_grp) const "combineBond".
            The following code are same result:
        @code
            // bd1, bd2 and bd3 are Bond objects.
                bd3 = bd1.combineBond(bd2);
        @endcode
        @code
            // bd1, bd2 and bd3 are Bond objects.
                bd3 = bd1 * bd2;
        @endcode
        @see operator*=(const Bond &rhs),
        combineBond(const Bond &bd_in, const bool &is_grp) const
    */
    Bond operator*(const Bond &rhs) const { return this->combineBond(rhs); }

    /**
    @brief The multiplication assignment operator of the Bond object.
    @details The multiplication assignment operator of the Bond means that Combine
            two Bond inplacely, So this operator is same as
                \ref combineBond_ "combineBond_".
            The following code are same result:
        @code
            // bd1 and bd2 are Bond objects.
                bd1.combineBond_(bd2);
        @endcode
        @code
            // bd1 and bd2 are Bond objects.
                bd1 *= bd2;
        @endcode
        @see operator*(const Bond &rhs) const,
                combineBond_(const Bond &bd_in, const bool &is_grp)
    */
    Bond &operator*=(const Bond &rhs) {
      this->combineBond_(rhs);
      return *this;
    }
  };

  ///@cond
  std::ostream &operator<<(std::ostream &os, const Bond &bin);
  ///@endcond
}  // namespace cytnx

#endif
