#ifndef _H_Bond_
#define _H_Bond_

#include "Type.hpp"
#include "Symmetry.hpp"
#include "cytnx_error.hpp"
#include <initializer_list>
#include <vector>
#include <fstream>
#include <map>
#include <algorithm>
#include "intrusive_ptr_base.hpp"
#include "utils/vec_clone.hpp"
namespace cytnx {

  /// currently using gBD_* to indicate this is bond with new qnum structure!
  enum bondType : int { BD_KET = -1, BD_BRA = 1, BD_REG = 0, BD_NONE = 0, BD_IN = -1, BD_OUT = 1 };
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

    void _rm_qnum(const cytnx_uint64 &q_index){
        // this will not check, so check it before using this internal function!!
        this->_dim -= this->_degs[q_index];
        this->_degs.erase(this->_degs.begin() + q_index);
        this->_qnums.erase(this->_qnums.begin() + q_index);
    }

    void Init(const cytnx_uint64 &dim, const bondType &bd_type = bondType::BD_REG,
              const std::vector<std::vector<cytnx_int64>> &in_qnums = {},
              const std::vector<Symmetry> &in_syms = {});

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

    bool has_duplicate_qnums() const{
        if(this->_degs.size()){
            auto tmp = this->_qnums;
            std::sort(tmp.begin(),tmp.end());
            return std::adjacent_find(tmp.begin(), tmp.end()) != tmp.end();
        }else{
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


    std::vector<cytnx_uint64>& getDegeneracies(){ return this->_degs;};
    const std::vector<cytnx_uint64>& getDegeneracies() const{return this->_degs;};

    std::vector<cytnx_uint64> group_duplicates_();

    boost::intrusive_ptr<Bond_impl> group_duplicates(std::vector<cytnx_uint64> &mapper) const{
        boost::intrusive_ptr<Bond_impl> out = this->clone();
        mapper = out->group_duplicates_();
        return out;
    }

    void force_combineBond_(const boost::intrusive_ptr<Bond_impl> &bd_in, const bool &is_grp);



  };  // Bond_impl
  ///@endcond

  /// @brief the object contains auxiliary properties for each Tensor rank (bond)
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

    Bond(const cytnx_uint64 &dim, const bondType &bd_type = bondType::BD_REG,
         const std::vector<std::vector<cytnx_int64>> &in_qnums = {},
         const std::vector<Symmetry> &in_syms = {})
        : _impl(new Bond_impl()) {
      this->_impl->Init(dim, bd_type, in_qnums, in_syms);
    }

    Bond(const bondType &bd_type, const std::vector<std::vector<cytnx_int64>> &in_qnums,
         const std::vector<cytnx_uint64> &degs, const std::vector<Symmetry> &in_syms = {})
        : _impl(new Bond_impl()) {
      this->_impl->Init(bd_type, in_qnums, degs, in_syms);
    }


    Bond(const bondType &bd_type, const std::initializer_list<std::vector<cytnx_int64>> &in_qnums,
         const std::vector<cytnx_uint64> &degs, const std::vector<Symmetry> &in_syms = {})
        : _impl(new Bond_impl()) {
      this->_impl->Init(bd_type, in_qnums, degs, in_syms);
    }

    // this is needed for python binding!
    Bond(const bondType &bd_type, const std::vector<cytnx::Qs> &in_qnums,
         const std::vector<cytnx_uint64> &degs, const std::vector<Symmetry> &in_syms = {})
        : _impl(new Bond_impl()) {
      vec2d<cytnx_int64> qnums(in_qnums.begin(),in_qnums.end()); 
      this->_impl->Init(bd_type, qnums, degs, in_syms);
    }

    

    Bond(const bondType &bd_type, const std::vector< std::pair<std::vector<cytnx_int64>, cytnx_uint64> > &in_qnums_dims,
         const std::vector<Symmetry> &in_syms = {})
        : _impl(new Bond_impl()) {
      this->Init(bd_type, in_qnums_dims, in_syms);
    }


    /**
    @brief init a bond object
    @param dim the dimension of the bond (rank)
    @param bd_type the tag of the bond, it can be BD_BRA, BD_KET as physical tagged; or BD_REG as
    regular bond (rank)
    @param in_qnums the quantum number(s) of the bond. it should be a 2d vector with shape (# of
    symmetry, dim)
    @param in_syms the symmetry object of the bond. [Note] if qnums are provided, the default
    symmetry type is \link cytnx::Symmetry::U1 Symmetry::U1 \endlink

    description:
        1. each bond can be tagged with BD_BRA or BD_KET that represent the bond is defined in Bra
    space or Ket space.
        2. the bond can have arbitrary multiple symmetries, with the type of each symmetry associate
    to the qnums are provided with the in_syms.

    [Note]
        1. if quantum number(s) are provided (which means the bond is with symmetry) then the bond
    MUST be tagged with either BD_BRA or BD_KET
        2. if the bond is non-symmetry, then it can be tagged with BD_BRA or BD_KET, or BD_REG
    depending on the usage.

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
    void Init(const cytnx_uint64 &dim, const bondType &bd_type = bondType::BD_REG,
              const std::vector<std::vector<cytnx_int64>> &in_qnums = {},
              const std::vector<Symmetry> &in_syms = {}) {
      this->_impl->Init(dim, bd_type, in_qnums, in_syms);
    }

    /**
    @brief init a bond object
    @param bd_type the tag of the bond, it can be BD_BRA, BD_KET as physical tagged and cannot be
    BD_REG (regular bond).
    @param in_qnums the quantum number(s) of the bond. it should be a 2d vector with shape (# of
    unique qnum labels, # of symmetry).
    @param degs the degeneracy correspond to each qunatum number sets specified in the qnums, the
    size should match the # of rows of passed-in qnums.
    @param in_syms a vector of symmetry objects of the bond, the size should match the # of cols of
    passed-in qnums. [Note] if qnums are provided, the default symmetry type is \link
    cytnx::Symmetry::U1 Symmetry::U1 \endlink

    description:
        1. each bond can be tagged with BD_BRA or BD_KET that represent the bond is defined in Bra
    space or Ket space.
        2. the bond can have arbitrary multiple symmetries, with the type of each symmetry associate
    to the qnums are provided with the in_syms.

    [Note]
        1. if quantum number(s) are provided (which means the bond is with symmetry) then the bond
    MUST be tagged with either BD_BRA or BD_KET
        2. if the bond is non-symmetry, then it can be tagged with BD_BRA or BD_KET, or BD_REG
    depending on the usage.
        3. the "bond dimension" is the sum over all numbers specified in degs.
    */
    void Init(const bondType &bd_type, const std::vector<std::vector<cytnx_int64>> &in_qnums,
              const std::vector<cytnx_uint64> &degs, const std::vector<Symmetry> &in_syms = {}) {
      this->_impl->Init(bd_type, in_qnums, degs, in_syms);
    }

    void Init(const bondType &bd_type, const std::vector< std::pair<std::vector<cytnx_int64>, cytnx_uint64> > &in_qnums_dims,
         const std::vector<Symmetry> &in_syms = {}){
      
      vec2d<cytnx_int64> qnums(in_qnums_dims.size());
      std::vector<cytnx_uint64> degs(in_qnums_dims.size());
      for(int i=0;i<in_qnums_dims.size();i++){
        qnums[i] = in_qnums_dims[i].first;
        degs[i] = in_qnums_dims[i].second;
      }
            
      this->_impl->Init(bd_type, qnums, degs, in_syms);

    }




    /**
    @brief return the current tag type
    @return [bondType] can be BD_BRA, BD_KET or BD_REG

    */
    bondType type() const { return this->_impl->type(); };

    //@{
    /**
    @brief return the current quantum number set(s) by reference
    @return [2d vector] with shape: (dim, # of Symmetry)


    */
    const std::vector<std::vector<cytnx_int64>> &qnums() const { return this->_impl->qnums(); };
    std::vector<std::vector<cytnx_int64>> &qnums() { return this->_impl->qnums(); };
    //@}

    /**
    @brief return copy of the current quantum number set(s)
    @return [2d vector] with shape: (dim, # of Symmetry)

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
    @brief return the number of symmetries
    @return [cytnx_uint32]

    */
    cytnx_uint32 Nsym() const { return this->_impl->syms().size(); };

    //@{
    /**
    @brief return the vector of symmetry objects by reference.
    @return [vector of Symmetry]

    */
    const std::vector<Symmetry> &syms() const { return this->_impl->syms(); };
    std::vector<Symmetry> &syms() { return this->_impl->syms(); };
    //@}

    /**
    @brief return copy of the vector of symmetry objects.
    @return [vector of Symmetry]

    */
    std::vector<Symmetry> syms_clone() const { return this->_impl->syms_clone(); };

    /**
    @brief change the tag-type of the instance Bond
    @param new_bondType the new tag-type, it can be BD_BRA,BD_KET or BD_REG

    */
    Bond &set_type(const bondType &new_bondType) {
      this->_impl->set_type(new_bondType);
      return *this;
    }

    /**
    @brief create a new instance of Bond with type changed to the new tag-type:
    @param  new_bondType the new tag-type, it can be BD_BRA,BD_KET or BD_REG

    [Note]
        This is equivalent to Bond.clone().set_type()

    */
    Bond retype(const bondType &new_bondType) {
      auto out = this->clone();
      out.set_type(new_bondType);
      return out;
    }

    /**
    @brief create a new instance of Bond with type changed in btwn BRA / KET:

    */
    Bond redirect() const {
      auto out = this->clone();
      out.set_type(bondType(int(out.type()) * -1));
      return out;
    }
    
    /**
    @brief create a new instance of Bond with type changed in btwn BRA / KET:

    */
    Bond& redirect_(){
        this->set_type(bondType(int(this->type()) * -1));
        return *this;
    }


    /**
    @brief change the tag-type to the default value BD_REG

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
    @brief combine the input bond with self, inplacely
    @param bd_in the bond that to be combined with self.

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
    @param bd_in the bond that to be combined.
    @return [Bond] a new combined bond instance.

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
    @param bds the bonds that to be combined with self.
    @return [Bond] a new combined bond instance.

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
    @brief combine multiple input bonds with self, inplacely
    @param bds the bonds that to be combined with self.

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
    @return unique_qnums

    */
    std::vector<std::vector<cytnx_int64>> getUniqueQnums(std::vector<cytnx_uint64> &counts) {
      return this->_impl->getUniqueQnums(counts, true);
    }
    std::vector<std::vector<cytnx_int64>> getUniqueQnums() {
      std::vector<cytnx_uint64> tmp;
      return this->_impl->getUniqueQnums(tmp, false);
    }

    /**
    @brief return the degeneracy of specify qnum set.
    @return degeneracy

    ## [Note]
        if the bond has no symmetries, return 0.
        if the degeneracy queried does not exist, return 0, and the indicies is empty

    */
    cytnx_uint64 getDegeneracy(const std::vector<cytnx_int64> &qnum) const {
      std::vector<cytnx_uint64> tmp;
      return this->_impl->getDegeneracy(qnum, false, tmp);
    }
    cytnx_uint64 getDegeneracy(const std::vector<cytnx_int64> &qnum,
                               std::vector<cytnx_uint64> &indices) const {
      indices.clear();
      return this->_impl->getDegeneracy(qnum, true, indices);
    }

    std::vector<cytnx_uint64> & getDegeneracies(){
        return this->_impl->getDegeneracies();
    }
    const std::vector<cytnx_uint64> & getDegeneracies() const{
        return this->_impl->getDegeneracies();
    }
        
    // the map returns the new index from old index via
    // new_index = return<cytnx_uint64>[old_index]
    std::vector<cytnx_uint64> group_duplicates_(){
        return this->_impl->group_duplicates_();
    }

    Bond group_duplicates(std::vector<cytnx_uint64> &mapper) const{
        Bond out;
        out._impl = this->_impl->group_duplicates(mapper);
        return out;
    }

    bool has_duplicate_qnums() const {
        return this->_impl->has_duplicate_qnums();
    }

    std::vector<std::vector<cytnx_int64>> calc_reverse_qnums() {
      return this->_impl->calc_reverse_qnums();
    }

    void Save(const std::string &fname) const;
    void Save(const char *fname) const;
    static cytnx::Bond Load(const std::string &fname);
    static cytnx::Bond Load(const char *fname);

    /// @cond
    void _Save(std::fstream &f) const;
    void _Load(std::fstream &f);
    /// @endcond

    bool operator==(const Bond &rhs) const;
    bool operator!=(const Bond &rhs) const;

    Bond operator*(const Bond &rhs) const { return this->combineBond(rhs); }

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
