#ifndef _H_UniTensor_
#define _H_UniTensor_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include "Device.hpp"
#include "Tensor.hpp"
#include "utils/utils.hpp"
#include "intrusive_ptr_base.hpp"
#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <initializer_list>
#include <fstream>
#include <algorithm>
#include "Symmetry.hpp"
#include "Bond.hpp"
#include "Generator.hpp"
#include <random>

#ifdef BACKEND_TORCH
#else
  #include "backend/Scalar.hpp"

// namespace cytnx{
namespace cytnx {
  namespace random {
    extern std::random_device __static_random_device;
  }

  using namespace cytnx;
  /// @cond
  class UniTensorType_class {
   public:
    enum : int {
      Void = -99,
      Dense = 0,
      Sparse = 1,
      Block = 2,
    };
    std::string getname(const int &ut_type);
  };
  /// @endcond
  /**
   * @brief UniTensor type.
   * @details It is about the type of the UniTensor.\n
   *     The supported enumerations are as following:
   *
   *  enumeration  |  description
   * --------------|--------------------
   *  Void         |  -1, void UniTensor
   *  Dense        |  0, dense UniTensor
   *  Sparse       |  1, sparse UniTensor (deprecated)
   *  Block        |  2, block UniTensor
   *
   *  @warning the type \em Sparse is deprecated. Use \em Block instead.
   *  @see UniTensor::uten_type(), UniTensor::uten_type_str()
   */

  extern UniTensorType_class UTenType;

  /// @cond
  // class DenseUniTensor;
  // class SparseUniTensor;
  class UniTensor_base : public intrusive_ptr_base<UniTensor_base> {
   public:
    int uten_type_id;  // the unitensor type id.
    bool _is_braket_form;
    bool _is_tag;
    bool _is_diag;
    cytnx_int64 _rowrank;
    std::string _name;
    std::vector<std::string> _labels;
    std::vector<Bond> _bonds;

    bool _update_braket() {
      if (_bonds.size() == 0) return false;

      if (this->_bonds[0].type() != bondType::BD_REG) {
        // check:
        for (unsigned int i = 0; i < this->_bonds.size(); i++) {
          if (i < this->_rowrank) {
            if (this->_bonds[i].type() != bondType::BD_KET) return false;
          } else {
            if (this->_bonds[i].type() != bondType::BD_BRA) return false;
          }
        }
        return true;
      } else {
        return false;
      }
    }

    friend class UniTensor;  // allow wrapper to access the private elems
    friend class DenseUniTensor;
    // friend class SparseUniTensor;
    friend class BlockUniTensor;

    UniTensor_base()
        : _is_tag(false),
          _name(std::string("")),
          _is_braket_form(false),
          _rowrank(0),
          _is_diag(false),
          uten_type_id(UTenType.Void){};

    // copy&assignment constr., use intrusive_ptr's !!
    UniTensor_base(const UniTensor_base &rhs);
    UniTensor_base &operator=(UniTensor_base &rhs);

    cytnx_uint64 rowrank() const { return this->_rowrank; }
    bool is_diag() const { return this->_is_diag; }
    const bool &is_braket_form() const { return this->_is_braket_form; }
    const bool &is_tag() const { return this->_is_tag; }
    const std::vector<std::string> &labels() const { return this->_labels; }
    /**
     * @brief Get the index of an desired label string
     *
     * @param label Label you want to find
     * @return The index of the label. If not found, return -1
     */
    cytnx_int64 get_index(std::string label) const {
      std::vector<std::string> labels = this->_labels;
      for (cytnx_uint64 i = 0; i < labels.size(); i++) {
        if (labels[i] == label) return i;
      }
      return -1;
    }
    const std::vector<Bond> &bonds() const { return this->_bonds; }
    std::vector<Bond> &bonds() { return this->_bonds; }

    Bond &bond_(const cytnx_uint64 &idx) {
      cytnx_error_msg(idx >= this->_bonds.size(), "[ERROR][bond] index %d out of bound, total %d\n",
                      idx, this->_bonds.size());
      return this->_bonds[idx];
    }

    Bond &bond_(const std::string &label) {
      auto res = std::find(this->_labels.begin(), this->_labels.end(), label);
      cytnx_error_msg(res == this->_labels.end(), "[ERROR] label %s not exists.\n", label.c_str());
      cytnx_uint64 idx = std::distance(this->_labels.begin(), res);

      return this->bond_(idx);
    }

    const std::string &name() const { return this->_name; }
    cytnx_uint64 rank() const { return this->_labels.size(); }
    void set_name(const std::string &in) { this->_name = in; }

    /**
     * @brief Set the label object
     * @details Replace the old label by new label.
     * @param[in] oldlabel The old label you want to replace.
     * @param[in] new_lable The label you want to replace with.
     * @pre
     * 1. \p oldlabel should be exist in this UniTensor.
     * 2. The new label \p new_label cannot set as others exit labels (cannot be duplicated.)
     * @see set_label(const cytnx_int64 &inx, const std::string &new_label)
     */
    void set_label(const std::string &oldlabel, const std::string &new_label) {
      cytnx_int64 idx;
      auto res = std::find(this->_labels.begin(), this->_labels.end(), oldlabel);
      cytnx_error_msg(res == this->_labels.end(), "[ERROR] label %s not exists.\n",
                      oldlabel.c_str());
      idx = std::distance(this->_labels.begin(), res);

      cytnx_error_msg(idx >= this->_labels.size(), "[ERROR] index exceed the rank of UniTensor%s",
                      "\n");
      // check in:
      bool is_dup = false;
      for (cytnx_uint64 i = 0; i < this->_labels.size(); i++) {
        if (i == idx) continue;
        if (new_label == this->_labels[i]) {
          is_dup = true;
          break;
        }
      }
      cytnx_error_msg(is_dup, "[ERROR] alreay has a label that is the same as the input label%s",
                      "\n");
      this->_labels[idx] = new_label;
    }
    void set_label(const cytnx_int64 &inx, const std::string &new_label) {
      cytnx_error_msg(inx < 0, "[ERROR] index is negative%s", "\n");
      cytnx_error_msg(inx >= this->_labels.size(), "[ERROR] index exceed the rank of UniTensor%s",
                      "\n");
      // check in:
      bool is_dup = false;
      for (cytnx_uint64 i = 0; i < this->_labels.size(); i++) {
        if (i == inx) continue;
        if (new_label == this->_labels[i]) {
          is_dup = true;
          break;
        }
      }
      cytnx_error_msg(is_dup, "[ERROR] alreay has a label that is the same as the input label%s",
                      "\n");
      this->_labels[inx] = new_label;
    }

    void set_labels(const std::vector<std::string> &new_labels);
    void relabel_(const std::vector<std::string> &new_labels);  // implemented
    void relabels_(const std::vector<std::string> &new_labels);  // implemented
    void relabel_(const std::vector<std::string> &old_labels,
                  const std::vector<std::string> &new_labels);  // implemented
    void relabels_(const std::vector<std::string> &old_labels,
                   const std::vector<std::string> &new_labels);  // implemented
    void relabel_(const std::string &old_label, const std::string &new_label) {
      this->set_label(old_label, new_label);
    }
    void relabel_(const cytnx_int64 &inx, const std::string &new_label) {
      this->set_label(inx, new_label);
    }

    int uten_type() { return this->uten_type_id; }
    std::string uten_type_str() { return UTenType.getname(this->uten_type_id); }

    /// VIRTUAL FUNCTIONS:

    // string labels!
    virtual void Init(const std::vector<Bond> &bonds,
                      const std::vector<std::string> &in_labels = {},
                      const cytnx_int64 &rowrank = -1, const unsigned int &dtype = Type.Double,
                      const int &device = Device.cpu, const bool &is_diag = false,
                      const bool &no_alloc = false, const std::string &name = "");

    virtual void Init_by_Tensor(const Tensor &in, const bool &is_diag = false,
                                const cytnx_int64 &rowrank = -1, const std::string &name = "");
    virtual std::vector<cytnx_uint64> shape() const;
    virtual bool is_blockform() const;
    virtual bool is_contiguous() const;
    virtual void to_(const int &device);
    virtual boost::intrusive_ptr<UniTensor_base> to(const int &device);
    virtual boost::intrusive_ptr<UniTensor_base> clone() const;
    virtual unsigned int dtype() const;
    virtual int device() const;
    virtual std::string dtype_str() const;
    virtual std::string device_str() const;
    virtual void set_rowrank_(const cytnx_uint64 &new_rowrank);
    virtual boost::intrusive_ptr<UniTensor_base> set_rowrank(const cytnx_uint64 &new_rowrank) const;

    virtual boost::intrusive_ptr<UniTensor_base> permute(const std::vector<cytnx_int64> &mapper,
                                                         const cytnx_int64 &rowrank = -1);
    virtual boost::intrusive_ptr<UniTensor_base> permute(const std::vector<std::string> &mapper,
                                                         const cytnx_int64 &rowrank = -1);
    // virtual boost::intrusive_ptr<UniTensor_base> permute(const std::vector<cytnx_int64> &mapper,
    //                                                      const cytnx_int64 &rowrank = -1);

    virtual void permute_(const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank = -1);
    virtual void permute_(const std::vector<std::string> &mapper, const cytnx_int64 &rowrank = -1);

    // virtual void permute_(const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank =
    // -1);
    virtual boost::intrusive_ptr<UniTensor_base> contiguous_();
    virtual boost::intrusive_ptr<UniTensor_base> contiguous();
    virtual void print_diagram(const bool &bond_info = false);
    virtual void print_blocks(const bool &full_info = true) const;
    virtual void print_block(const cytnx_int64 &idx, const bool &full_info = true) const;

    virtual boost::intrusive_ptr<UniTensor_base> astype(const unsigned int &dtype) const;

    virtual cytnx_uint64 Nblocks() const { return 0; };
    virtual Tensor get_block(const cytnx_uint64 &idx = 0) const;  // return a copy of block
    virtual Tensor get_block(const std::vector<cytnx_int64> &qnum,
                             const bool &force) const;  // return a copy of block

    virtual const Tensor &get_block_(const cytnx_uint64 &idx = 0)
      const;  // return a share view of block, this only work for non-symm tensor.
    virtual const Tensor &get_block_(const std::vector<cytnx_int64> &qnum,
                                     const bool &force) const;  // return a copy of block
    virtual Tensor &get_block_(const cytnx_uint64 &idx = 0);  // return a share view of block, this
                                                              // only work for non-symm tensor.
    virtual Tensor &get_block_(const std::vector<cytnx_int64> &qnum,
                               const bool &force);  // return a copy of block
    virtual bool same_data(const boost::intrusive_ptr<UniTensor_base> &rhs) const;

    virtual std::vector<Tensor> get_blocks() const;
    virtual const std::vector<Tensor> &get_blocks_(const bool &) const;
    virtual std::vector<Tensor> &get_blocks_(const bool &);

    virtual void put_block(const Tensor &in, const cytnx_uint64 &idx = 0);
    virtual void put_block_(Tensor &in, const cytnx_uint64 &idx = 0);
    virtual void put_block(const Tensor &in, const std::vector<cytnx_int64> &qnum,
                           const bool &force);
    virtual void put_block_(Tensor &in, const std::vector<cytnx_int64> &qnum, const bool &force);

    // this will only work on non-symm tensor (DenseUniTensor)
    virtual boost::intrusive_ptr<UniTensor_base> get(const std::vector<Accessor> &accessors);

    // this will only work on non-symm tensor (DenseUniTensor)
    virtual void set(const std::vector<Accessor> &accessors, const Tensor &rhs);

    virtual void reshape_(const std::vector<cytnx_int64> &new_shape,
                          const cytnx_uint64 &rowrank = 0);
    virtual boost::intrusive_ptr<UniTensor_base> reshape(const std::vector<cytnx_int64> &new_shape,
                                                         const cytnx_uint64 &rowrank = 0);
    virtual boost::intrusive_ptr<UniTensor_base> to_dense();
    virtual void to_dense_();
    virtual void combineBond(const std::vector<std::string> &indicators, const bool &force = false);
    virtual void combineBonds(const std::vector<cytnx_int64> &indicators, const bool &force,
                              const bool &by_label);
    virtual void combineBonds(const std::vector<std::string> &indicators,
                              const bool &force = false);
    virtual void combineBonds(const std::vector<cytnx_int64> &indicators,
                              const bool &force = false);
    virtual boost::intrusive_ptr<UniTensor_base> contract(
      const boost::intrusive_ptr<UniTensor_base> &rhs, const bool &mv_elem_self = false,
      const bool &mv_elem_rhs = false);
    virtual std::vector<Bond> getTotalQnums(const bool &physical = false);
    virtual std::vector<std::vector<cytnx_int64>> get_blocks_qnums() const;
    virtual void Trace_(const std::string &a, const std::string &b);
    virtual void Trace_(const cytnx_int64 &a, const cytnx_int64 &b);

    virtual boost::intrusive_ptr<UniTensor_base> Trace(const std::string &a, const std::string &b);
    virtual boost::intrusive_ptr<UniTensor_base> Trace(const cytnx_int64 &a, const cytnx_int64 &b);

    virtual boost::intrusive_ptr<UniTensor_base> relabel(
      const std::vector<std::string> &new_labels);
    virtual boost::intrusive_ptr<UniTensor_base> relabels(
      const std::vector<std::string> &new_labels);

    virtual boost::intrusive_ptr<UniTensor_base> relabel(
      const std::vector<std::string> &old_labels, const std::vector<std::string> &new_labels);
    virtual boost::intrusive_ptr<UniTensor_base> relabels(
      const std::vector<std::string> &old_labels, const std::vector<std::string> &new_labels);

    virtual boost::intrusive_ptr<UniTensor_base> relabel(const std::string &old_label,
                                                         const std::string &new_label);

    virtual boost::intrusive_ptr<UniTensor_base> relabel(const cytnx_int64 &inx,
                                                         const std::string &new_label);

    virtual std::vector<Symmetry> syms() const;

    // arithmetic
    virtual void Add_(const boost::intrusive_ptr<UniTensor_base> &rhs);
    virtual void Add_(const Scalar &rhs);

    virtual void Mul_(const boost::intrusive_ptr<UniTensor_base> &rhs);
    virtual void Mul_(const Scalar &rhs);

    virtual void Sub_(const boost::intrusive_ptr<UniTensor_base> &rhs);
    virtual void Sub_(const Scalar &rhs);
    virtual void lSub_(const Scalar &lhs);

    virtual void Div_(const boost::intrusive_ptr<UniTensor_base> &rhs);
    virtual void Div_(const Scalar &rhs);
    virtual void lDiv_(const Scalar &lhs);

    virtual Tensor Norm() const;
    virtual boost::intrusive_ptr<UniTensor_base> normalize();
    virtual void normalize_();

    virtual boost::intrusive_ptr<UniTensor_base> Conj();
    virtual void Conj_();

    virtual boost::intrusive_ptr<UniTensor_base> Transpose();
    virtual void Transpose_();

    virtual boost::intrusive_ptr<UniTensor_base> Dagger();
    virtual void Dagger_();

    virtual void tag();

    virtual void truncate_(const std::string &bond_idx, const cytnx_uint64 &dim);
    virtual void truncate_(const cytnx_int64 &bond_idx, const cytnx_uint64 &dim);

    virtual bool elem_exists(const std::vector<cytnx_uint64> &locator) const;

    // this a workaround, as virtual function cannot template.
    virtual Scalar::Sproxy at_for_sparse(const std::vector<cytnx_uint64> &locator);
    virtual const Scalar::Sproxy at_for_sparse(const std::vector<cytnx_uint64> &locator) const;

    virtual cytnx_complex128 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                            const cytnx_complex128 &aux);
    virtual cytnx_complex64 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                           const cytnx_complex64 &aux);
    virtual cytnx_double &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                        const cytnx_double &aux);
    virtual cytnx_float &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                       const cytnx_float &aux);
    virtual cytnx_uint64 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                        const cytnx_uint64 &aux);
    virtual cytnx_int64 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                       const cytnx_int64 &aux);
    virtual cytnx_uint32 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                        const cytnx_uint32 &aux);
    virtual cytnx_int32 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                       const cytnx_int32 &aux);
    virtual cytnx_uint16 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                        const cytnx_uint16 &aux);
    virtual cytnx_int16 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                       const cytnx_int16 &aux);

    virtual const cytnx_complex128 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                  const cytnx_complex128 &aux) const;
    virtual const cytnx_complex64 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                 const cytnx_complex64 &aux) const;
    virtual const cytnx_double &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                              const cytnx_double &aux) const;
    virtual const cytnx_float &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                             const cytnx_float &aux) const;
    virtual const cytnx_uint64 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                              const cytnx_uint64 &aux) const;
    virtual const cytnx_int64 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                             const cytnx_int64 &aux) const;
    virtual const cytnx_uint32 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                              const cytnx_uint32 &aux) const;
    virtual const cytnx_int32 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                             const cytnx_int32 &aux) const;
    virtual const cytnx_uint16 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                              const cytnx_uint16 &aux) const;
    virtual const cytnx_int16 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                             const cytnx_int16 &aux) const;

    virtual void from_(const boost::intrusive_ptr<UniTensor_base> &rhs, const bool &force);

    virtual void group_basis_();
    virtual const std::vector<cytnx_uint64> &get_qindices(const cytnx_uint64 &bidx) const;
    virtual std::vector<cytnx_uint64> &get_qindices(const cytnx_uint64 &bidx);
    virtual const vec2d<cytnx_uint64> &get_itoi() const;
    virtual vec2d<cytnx_uint64> &get_itoi();

    virtual void _save_dispatch(std::fstream &f) const;
    virtual void _load_dispatch(std::fstream &f);

    virtual ~UniTensor_base(){};
  };
  /// @endcond

  //======================================================================
  /// @cond
  class DenseUniTensor : public UniTensor_base {
   protected:
   public:
    Tensor _block;
    std::vector<Tensor> _interface_block;  // this is serves as interface for get_blocks_();
    DenseUniTensor *clone_meta() const {
      DenseUniTensor *tmp = new DenseUniTensor();
      tmp->_bonds = vec_clone(this->_bonds);
      tmp->_labels = this->_labels;
      tmp->_is_braket_form = this->_is_braket_form;
      tmp->_rowrank = this->_rowrank;
      tmp->_is_diag = this->_is_diag;
      tmp->_name = this->_name;
      tmp->_is_tag = this->_is_tag;
      return tmp;
    }
    //------------------------------------------

    DenseUniTensor() { this->uten_type_id = UTenType.Dense; };
    friend class UniTensor;  // allow wrapper to access the private elems
    // virtual functions

    // void Init(const std::vector<Bond> &bonds, const std::vector<cytnx_int64> &in_labels = {},
    //           const cytnx_int64 &rowrank = -1, const unsigned int &dtype = Type.Double,
    //           const int &device = Device.cpu, const bool &is_diag = false,
    //           const bool &no_alloc = false);

    void Init(const std::vector<Bond> &bonds, const std::vector<std::string> &in_labels = {},
              const cytnx_int64 &rowrank = -1, const unsigned int &dtype = Type.Double,
              const int &device = Device.cpu, const bool &is_diag = false,
              const bool &no_alloc = false, const std::string &name = "");
    // this only work for non-symm tensor
    void Init_by_Tensor(const Tensor &in_tensor, const bool &is_diag = false,
                        const cytnx_int64 &rowrank = -1, const std::string &name = "");
    std::vector<cytnx_uint64> shape() const {
      if (this->_is_diag) {
        std::vector<cytnx_uint64> shape = this->_block.shape();
        shape.push_back(shape[0]);
        return shape;
      } else {
        return this->_block.shape();
      }
    }
    bool is_blockform() const { return false; }
    void to_(const int &device) { this->_block.to_(device); }
    boost::intrusive_ptr<UniTensor_base> to(const int &device) {
      if (this->device() == device) {
        std::vector<Tensor> _interface_block;  // this is serves as interface for get_blocks_();
        return this;
      } else {
        boost::intrusive_ptr<UniTensor_base> out = this->clone();
        out->to_(device);
        return out;
      }
    }
    void set_rowrank_(const cytnx_uint64 &new_rowrank) {
      cytnx_error_msg(new_rowrank > this->_labels.size(),
                      "[ERROR] rowrank cannot exceed the rank of UniTensor.%s", "\n");
      if (this->is_diag()) {
        cytnx_error_msg(new_rowrank != 1, "[ERROR] rowrank should be [==1] when is_diag =true!.%s",
                        "\n");
      }

      this->_rowrank = new_rowrank;
    }

    boost::intrusive_ptr<UniTensor_base> set_rowrank(const cytnx_uint64 &new_rowrank) const {
      DenseUniTensor *out_raw = this->clone_meta();
      out_raw->_block = this->_block;
      out_raw->set_rowrank_(new_rowrank);
      boost::intrusive_ptr<UniTensor_base> out(out_raw);
      return out;
    }

    boost::intrusive_ptr<UniTensor_base> clone() const {
      DenseUniTensor *tmp = this->clone_meta();
      tmp->_block = this->_block.clone();
      boost::intrusive_ptr<UniTensor_base> out(tmp);
      return out;
    };
    bool is_contiguous() const { return this->_block.is_contiguous(); }
    unsigned int dtype() const { return this->_block.dtype(); }
    int device() const { return this->_block.device(); }
    std::string dtype_str() const { return Type.getname(this->_block.dtype()); }
    std::string device_str() const { return Device.getname(this->_block.device()); }
    /**
     * @brief
     *
     *
     *
     * @param mapper
     * @param rowrank
     * @return boost::intrusive_ptr<UniTensor_base>
     */
    boost::intrusive_ptr<UniTensor_base> permute(const std::vector<cytnx_int64> &mapper,
                                                 const cytnx_int64 &rowrank = -1);
    boost::intrusive_ptr<UniTensor_base> permute(const std::vector<std::string> &mapper,
                                                 const cytnx_int64 &rowrank = -1);

    /**
     * @brief
     *
     *
     *
     * @param mapper
     * @param rowrank
     */
    void permute_(const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank = -1);
    void permute_(const std::vector<std::string> &mapper, const cytnx_int64 &rowrank = -1);

    boost::intrusive_ptr<UniTensor_base> relabel(const std::vector<std::string> &new_labels);
    boost::intrusive_ptr<UniTensor_base> relabels(const std::vector<std::string> &new_labels);

    boost::intrusive_ptr<UniTensor_base> relabel(const std::vector<std::string> &old_labels,
                                                 const std::vector<std::string> &new_labels);
    boost::intrusive_ptr<UniTensor_base> relabels(const std::vector<std::string> &old_labels,
                                                  const std::vector<std::string> &new_labels);

    /**
     * @brief
     *
     *
     *
     * @param inx
     * @param new_label
     *
     * @return boost::intrusive_ptr<UniTensor_base>
     */
    boost::intrusive_ptr<UniTensor_base> relabel(const std::string &old_label,
                                                 const std::string &new_label);
    boost::intrusive_ptr<UniTensor_base> relabel(const cytnx_int64 &inx,
                                                 const std::string &new_label);

    boost::intrusive_ptr<UniTensor_base> astype(const unsigned int &dtype) const {
      DenseUniTensor *tmp = this->clone_meta();
      tmp->_block = this->_block.astype(dtype);
      boost::intrusive_ptr<UniTensor_base> out(tmp);
      return tmp;
    }

    std::vector<Symmetry> syms() const {
      cytnx_error_msg(true, "[ERROR][DenseUniTensor] dense unitensor does not have symmetry.%s",
                      "\n");
      return std::vector<Symmetry>();
    }

    boost::intrusive_ptr<UniTensor_base> contiguous_() {
      this->_block.contiguous_();
      return boost::intrusive_ptr<UniTensor_base>(this);
    }
    boost::intrusive_ptr<UniTensor_base> contiguous() {
      // if contiguous then return self!
      if (this->is_contiguous()) {
        boost::intrusive_ptr<UniTensor_base> out(this);
        return out;
      } else {
        DenseUniTensor *tmp = this->clone_meta();
        tmp->_block = this->_block.contiguous();
        boost::intrusive_ptr<UniTensor_base> out(tmp);
        return out;
      }
    }
    void print_diagram(const bool &bond_info = false);
    void print_blocks(const bool &full_info = true) const;
    void print_block(const cytnx_int64 &idx, const bool &full_info = true) const;
    Tensor get_block(const cytnx_uint64 &idx = 0) const { return this->_block.clone(); }

    Tensor get_block(const std::vector<cytnx_int64> &qnum, const bool &force) const {
      cytnx_error_msg(
        true, "[ERROR][DenseUniTensor] try to get_block() using qnum on a non-symmetry UniTensor%s",
        "\n");
      return Tensor();
    }
    // return a share view of block, this only work for non-symm tensor.
    const Tensor &get_block_(const std::vector<cytnx_int64> &qnum, const bool &force) const {
      cytnx_error_msg(
        true,
        "[ERROR][DenseUniTensor] try to get_block_() using qnum on a non-symmetry UniTensor%s",
        "\n");
      return this->_block;
    }
    Tensor &get_block_(const std::vector<cytnx_int64> &qnum, const bool &force) {
      cytnx_error_msg(
        true,
        "[ERROR][DenseUniTensor] try to get_block_() using qnum on a non-symmetry UniTensor%s",
        "\n");
      return this->_block;
    }

    // return a share view of block, this only work for non-symm tensor.
    Tensor &get_block_(const cytnx_uint64 &idx = 0) { return this->_block; }
    // return a share view of block, this only work for non-symm tensor.
    const Tensor &get_block_(const cytnx_uint64 &idx = 0) const { return this->_block; }

    cytnx_uint64 Nblocks() const { return 1; };
    std::vector<Tensor> get_blocks() const {
      std::vector<Tensor> out;
      cytnx_error_msg(
        true, "[ERROR][DenseUniTensor] cannot use get_blocks(), use get_block() instead!%s", "\n");
      return out;  // this will not share memory!!
    }
    const std::vector<Tensor> &get_blocks_(const bool &silent = false) const {
      cytnx_error_msg(
        true, "[ERROR][DenseUniTensor] cannot use get_blocks_(), use get_block_() instead!%s",
        "\n");
      return this->_interface_block;  // this will not share memory!!
    }
    std::vector<Tensor> &get_blocks_(const bool &silent = false) {
      cytnx_error_msg(
        true, "[ERROR][DenseUniTensor] cannot use get_blocks_(), use get_block_() instead!%s",
        "\n");
      return this->_interface_block;  // this will not share memory!!
    }

    void put_block(const Tensor &in, const cytnx_uint64 &idx = 0) {
      // We don't check the dtype for DenseUniTensor, since it'll be more convinent to change
      // DenseUniTensor's dtype

      // cytnx_error_msg(in.dtype() != this->dtype(),
      //                 "[ERROR][DenseUniTensor][put_block] The input tensor dtype does not
      //                 match.%s",
      //                 "\n");
      cytnx_error_msg(in.device() != this->device(),
                      "[ERROR][DenseUniTensor][put_block] The input tensor device does not "
                      "match.%s",
                      "\n");
      // We shouldn't check the contiguous
      // cytnx_error_msg(!in.contiguous());
      if (this->is_diag()) {
        cytnx_error_msg(
          in.shape() != this->_block.shape(),
          "[ERROR][DenseUniTensor] put_block, the input tensor shape does not match.%s", "\n");
        this->_block = in.clone();
      } else {
        cytnx_error_msg(
          in.shape() != this->shape(),
          "[ERROR][DenseUniTensor] put_block, the input tensor shape does not match.%s", "\n");
        this->_block = in.clone();
      }
    }
    // share view of the block
    void put_block_(Tensor &in, const cytnx_uint64 &idx = 0) {
      // We don't check the dtype for DenseUniTensor, since it'll be more convinent to change
      // DenseUniTensor's dtype

      // cytnx_error_msg(in.dtype() != this->dtype(),
      //                 "[ERROR][DenseUniTensor][put_block] The input tensor dtype does not
      //                 match.%s",
      //                 "\n");
      cytnx_error_msg(in.device() != this->device(),
                      "[ERROR][DenseUniTensor][put_block] The input tensor device does not "
                      "match.%s",
                      "\n");
      // We shouldn't check the contiguous
      // cytnx_error_msg(!in.contiguous());
      if (this->is_diag()) {
        cytnx_error_msg(
          in.shape() != this->_block.shape(),
          "[ERROR][DenseUniTensor] put_block, the input tensor shape does not match.%s", "\n");
        this->_block = in;
      } else {
        cytnx_error_msg(
          in.shape() != this->shape(),
          "[ERROR][DenseUniTensor] put_block, the input tensor shape does not match.%s", "\n");
        this->_block = in;
      }
    }

    void put_block(const Tensor &in, const std::vector<cytnx_int64> &qnum, const bool &force) {
      cytnx_error_msg(
        true, "[ERROR][DenseUniTensor] try to put_block using qnum on a non-symmetry UniTensor%s",
        "\n");
    }
    void put_block_(Tensor &in, const std::vector<cytnx_int64> &qnum, const bool &force) {
      cytnx_error_msg(
        true, "[ERROR][DenseUniTensor] try to put_block using qnum on a non-symmetry UniTensor%s",
        "\n");
    }
    // this will only work on non-symm tensor (DenseUniTensor)
    boost::intrusive_ptr<UniTensor_base> get(const std::vector<Accessor> &accessors) {
      boost::intrusive_ptr<UniTensor_base> out(new DenseUniTensor());
      out->Init_by_Tensor(this->_block.get(accessors), false, 0);  // wrapping around.
      return out;
    }
    // this will only work on non-symm tensor (DenseUniTensor)
    void set(const std::vector<Accessor> &accessors, const Tensor &rhs) {
      this->_block.set(accessors, rhs);
    }

    void reshape_(const std::vector<cytnx_int64> &new_shape, const cytnx_uint64 &rowrank = 0);
    boost::intrusive_ptr<UniTensor_base> reshape(const std::vector<cytnx_int64> &new_shape,
                                                 const cytnx_uint64 &rowrank = 0);
    boost::intrusive_ptr<UniTensor_base> to_dense();
    void to_dense_();
    /**
     * @brief
     *
     * @deprecated
     *
     * @param indicators
     * @param permute_back
     * @param by_label
     */
    void combineBond(const std::vector<std::string> &indicators, const bool &force = true);
    void combineBonds(const std::vector<cytnx_int64> &indicators, const bool &force,
                      const bool &by_label);
    void combineBonds(const std::vector<std::string> &indicators, const bool &force = true);
    void combineBonds(const std::vector<cytnx_int64> &indicators, const bool &force = true);
    boost::intrusive_ptr<UniTensor_base> contract(const boost::intrusive_ptr<UniTensor_base> &rhs,
                                                  const bool &mv_elem_self = false,
                                                  const bool &mv_elem_rhs = false);
    std::vector<Bond> getTotalQnums(const bool &physical = false) {
      cytnx_error_msg(true, "[ERROR][DenseUniTensor] %s",
                      "getTotalQnums can only operate on UniTensor with symmetry.\n");
      return std::vector<Bond>();
    }

    std::vector<std::vector<cytnx_int64>> get_blocks_qnums() const {
      cytnx_error_msg(true, "[ERROR][DenseUniTensor] %s",
                      "get_blocks_qnums can only operate on UniTensor with symmetry.\n");
      return std::vector<std::vector<cytnx_int64>>();
    }

    bool same_data(const boost::intrusive_ptr<UniTensor_base> &rhs) const {
      if (rhs->uten_type() != UTenType.Dense) return false;

      return this->get_block_().same_data(rhs->get_block_());
    }

    ~DenseUniTensor(){};

    // arithmetic
    void Add_(const boost::intrusive_ptr<UniTensor_base> &rhs);
    void Add_(const Scalar &rhs);

    void Mul_(const boost::intrusive_ptr<UniTensor_base> &rhs);
    void Mul_(const Scalar &rhs);

    void Sub_(const boost::intrusive_ptr<UniTensor_base> &rhs);
    void Sub_(const Scalar &rhs);
    void lSub_(const Scalar &lhs);

    void Div_(const boost::intrusive_ptr<UniTensor_base> &rhs);
    void Div_(const Scalar &rhs);
    void lDiv_(const Scalar &lhs);

    void Conj_() { this->_block.Conj_(); };

    boost::intrusive_ptr<UniTensor_base> Conj() {
      boost::intrusive_ptr<UniTensor_base> out = this->clone();
      out->Conj_();
      return out;
    }

    boost::intrusive_ptr<UniTensor_base> Transpose() {
      boost::intrusive_ptr<UniTensor_base> out = this->clone();
      out->Transpose_();
      return out;
    }
    void Transpose_();

    boost::intrusive_ptr<UniTensor_base> normalize() {
      boost::intrusive_ptr<UniTensor_base> out = this->clone();
      out->normalize_();
      return out;
    }
    void normalize_();

    boost::intrusive_ptr<UniTensor_base> Dagger() {
      boost::intrusive_ptr<UniTensor_base> out = this->Conj();
      out->Transpose_();
      return out;
    }
    void Dagger_() {
      this->Conj_();
      this->Transpose_();
    }
    /**
     * @brief
     *
     *
     *
     * @param a
     * @param b
     *
     */
    void Trace_(const cytnx_int64 &a, const cytnx_int64 &b);
    void Trace_(const std::string &a, const std::string &b);
    boost::intrusive_ptr<UniTensor_base> Trace(const std::string &a, const std::string &b) {
      boost::intrusive_ptr<UniTensor_base> out = this->clone();
      out->Trace_(a, b);
      return out;
    }
    boost::intrusive_ptr<UniTensor_base> Trace(const cytnx_int64 &a, const cytnx_int64 &b) {
      boost::intrusive_ptr<UniTensor_base> out = this->clone();
      out->Trace_(a, b);
      return out;
    }

    Tensor Norm() const;

    const Scalar::Sproxy at_for_sparse(const std::vector<cytnx_uint64> &locator) const {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      return Scalar::Sproxy();
    }
    const cytnx_complex128 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                          const cytnx_complex128 &aux) const {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      return cytnx_complex128(0, 0);
    }
    const cytnx_complex64 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                         const cytnx_complex64 &aux) const {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      return cytnx_complex64(0, 0);
    }
    const cytnx_double &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                      const cytnx_double &aux) const {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      return 0;
    }
    const cytnx_float &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                     const cytnx_float &aux) const {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      return 0;
    }
    const cytnx_uint64 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                      const cytnx_uint64 &aux) const {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      return 0;
    }
    const cytnx_int64 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                     const cytnx_int64 &aux) const {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      return 0;
    }
    const cytnx_uint32 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                      const cytnx_uint32 &aux) const {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      return 0;
    }
    const cytnx_int32 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                     const cytnx_int32 &aux) const {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      return 0;
    }
    const cytnx_uint16 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                      const cytnx_uint16 &aux) const {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      return 0;
    }
    const cytnx_int16 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                     const cytnx_int16 &aux) const {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      return 0;
    }

    Scalar::Sproxy at_for_sparse(const std::vector<cytnx_uint64> &locator) {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      return Scalar::Sproxy();
    }
    cytnx_complex128 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                    const cytnx_complex128 &aux) {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      return *(cytnx_complex128 *)nullptr;
    }
    cytnx_complex64 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                   const cytnx_complex64 &aux) {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      return *(cytnx_complex64 *)nullptr;
    }
    cytnx_double &at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_double &aux) {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      return *(cytnx_double *)nullptr;
    }
    cytnx_float &at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_float &aux) {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      return *(cytnx_float *)nullptr;
    }
    cytnx_uint64 &at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint64 &aux) {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      return *(cytnx_uint64 *)nullptr;
    }
    cytnx_int64 &at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int64 &aux) {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      return *(cytnx_int64 *)nullptr;
    }
    cytnx_uint32 &at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint32 &aux) {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      return *(cytnx_uint32 *)nullptr;
    }
    cytnx_int32 &at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int32 &aux) {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      return *(cytnx_int32 *)nullptr;
    }
    cytnx_uint16 &at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint16 &aux) {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      return *(cytnx_uint16 *)nullptr;
    }
    cytnx_int16 &at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int16 &aux) {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      return *(cytnx_int16 *)nullptr;
    }

    bool elem_exists(const std::vector<cytnx_uint64> &locator) const {
      cytnx_error_msg(
        true, "[ERROR][DenseUniTensor] elem_exists can only be used on UniTensor with Symmetry.%s",
        "\n");
    }
    void tag() {
      if (!this->is_tag()) {
        for (int i = 0; i < this->_rowrank; i++) {
          this->_bonds[i].set_type(BD_KET);
        }
        for (int i = this->_rowrank; i < this->_bonds.size(); i++) {
          this->_bonds[i].set_type(BD_BRA);
        }
        this->_is_tag = true;
        this->_is_braket_form = this->_update_braket();
      }
    }
    /**
     * @brief
     *
     *
     *
     * @param bond_idx / bond_label
     * @param dim
     *
     */
    void truncate_(const cytnx_int64 &bond_idx, const cytnx_uint64 &dim);
    void truncate_(const std::string &bond_idx, const cytnx_uint64 &dim);

    void from_(const boost::intrusive_ptr<UniTensor_base> &rhs, const bool &force);

    void group_basis_() {
      cytnx_warning_msg(true, "[WARNING] group basis will not have any effect on DensUniTensor.%s",
                        "\n");
    }

    void _save_dispatch(std::fstream &f) const;
    void _load_dispatch(std::fstream &f);

    const std::vector<cytnx_uint64> &get_qindices(const cytnx_uint64 &bidx) const {
      cytnx_error_msg(true, "[ERROR] get_qindices can only be unsed on UniTensor with Symmetry.%s",
                      "\n");
    }
    std::vector<cytnx_uint64> &get_qindices(const cytnx_uint64 &bidx) {
      cytnx_error_msg(true, "[ERROR] get_qindices can only be unsed on UniTensor with Symmetry.%s",
                      "\n");
    }

    const vec2d<cytnx_uint64> &get_itoi() const {
      cytnx_error_msg(true, "[ERROR] get_itoi can only be unsed on UniTensor with Symmetry.%s",
                      "\n");
    }
    vec2d<cytnx_uint64> &get_itoi() {
      cytnx_error_msg(true, "[ERROR] get_itoi can only be unsed on UniTensor with Symmetry.%s",
                      "\n");
    }

    // end virtual function
  };
  /// @endcond

  //======================================================================
  /// @cond
  class BlockUniTensor : public UniTensor_base {
   protected:
   public:
    std::vector<std::vector<cytnx_uint64>> _inner_to_outer_idx;
    std::vector<Tensor> _blocks;
    Tensor NullRefTensor;  // this returns when access block is not exists!

    // given an index list [loc], get qnums from this->_bonds[loc] and return the combined qnums
    // calculated from Symm object! this assume 1. symmetry are the same for each bond!
    //             2. total_qns are feeded with size len(symmetry)
    void _fx_get_total_fluxs(std::vector<cytnx_uint64> &loc, const std::vector<Symmetry> &syms,
                             std::vector<cytnx_int64> &total_qns) {
      memset(&total_qns[0], 0, sizeof(cytnx_int64) * total_qns.size());

      for (cytnx_int32 i = 0; i < syms.size(); i++) {
        if (this->_bonds[0].type() == BD_BRA)
          total_qns[i] = syms[0].reverse_rule(this->_bonds[0]._impl->_qnums[loc[0]][i]);
        else
          total_qns[i] = this->_bonds[0]._impl->_qnums[loc[0]][i];

        for (auto j = 1; j < loc.size(); j++) {
          if (this->_bonds[j].type() == BD_BRA)
            total_qns[i] = syms[i].combine_rule(
              total_qns[i], syms[i].reverse_rule(this->_bonds[j]._impl->_qnums[loc[j]][i]));
          else {
            total_qns[i] =
              syms[i].combine_rule(total_qns[i], this->_bonds[j]._impl->_qnums[loc[j]][i]);
          }
        }
      }
    }

    void _fx_locate_elem(cytnx_int64 &bidx, std::vector<cytnx_uint64> &loc_in_T,
                         const std::vector<cytnx_uint64> &locator) const;

    // internal function, grouping all duplicate qnums in all bonds
    void _fx_group_duplicates(const std::vector<cytnx_uint64> &dup_bond_idxs,
                              const std::vector<std::vector<cytnx_uint64>> &idx_mappers);

    void set_meta(BlockUniTensor *tmp, const bool &inner, const bool &outer) const {
      // outer meta
      if (outer) {
        tmp->_bonds = vec_clone(this->_bonds);
        tmp->_labels = this->_labels;
        tmp->_is_braket_form = this->_is_braket_form;
        tmp->_rowrank = this->_rowrank;
        tmp->_name = this->_name;
      }

      tmp->_is_diag = this->_is_diag;

      // inner meta
      if (inner) {
        tmp->_inner_to_outer_idx = this->_inner_to_outer_idx;
      }
    }

    BlockUniTensor *clone_meta(const bool &inner, const bool &outer) const {
      BlockUniTensor *tmp = new BlockUniTensor();
      this->set_meta(tmp, inner, outer);
      return tmp;
    };

    friend class UniTensor;
    BlockUniTensor() {
      this->uten_type_id = UTenType.Block;
      this->_is_tag = true;
    }

    // virtual functions:
    // void Init(const std::vector<Bond> &bonds, const std::vector<cytnx_int64> &in_labels = {},
    //           const cytnx_int64 &rowrank = -1, const unsigned int &dtype = Type.Double,
    //           const int &device = Device.cpu, const bool &is_diag = false,
    //           const bool &no_alloc = false);

    void Init(const std::vector<Bond> &bonds, const std::vector<std::string> &in_labels = {},
              const cytnx_int64 &rowrank = -1, const unsigned int &dtype = Type.Double,
              const int &device = Device.cpu, const bool &is_diag = false,
              const bool &no_alloc = false, const std::string &name = "");

    void Init_by_Tensor(const Tensor &in_tensor, const bool &is_diag = false,
                        const cytnx_int64 &rowrank = -1, const std::string &name = "") {
      cytnx_error_msg(
        true, "[ERROR][BlockUniTensor] cannot use Init_by_tensor() on a BlockUniTensor.%s", "\n");
    }

    std::vector<cytnx_uint64> shape() const {
      std::vector<cytnx_uint64> out(this->_bonds.size());
      for (cytnx_uint64 i = 0; i < out.size(); i++) {
        out[i] = this->_bonds[i].dim();
      }
      return out;
    }

    bool is_blockform() const { return true; }
    bool is_contiguous() const {
      bool out = true;
      for (int i = 0; i < this->_blocks.size(); i++) {
        out &= this->_blocks[i].is_contiguous();
      }
      return out;
    };

    cytnx_uint64 Nblocks() const { return this->_blocks.size(); };

    void to_(const int &device) {
      for (cytnx_uint64 i = 0; i < this->_blocks.size(); i++) {
        this->_blocks[i].to_(device);
      }
    };

    boost::intrusive_ptr<UniTensor_base> to(const int &device) {
      if (this->device() == device) {
        return this;
      } else {
        boost::intrusive_ptr<UniTensor_base> out = this->clone();
        out->to_(device);
        return out;
      }
    };

    boost::intrusive_ptr<UniTensor_base> clone() const {
      BlockUniTensor *tmp = this->clone_meta(true, true);
      tmp->_blocks = vec_clone(this->_blocks);
      boost::intrusive_ptr<UniTensor_base> out(tmp);
      return out;
    };

    unsigned int dtype() const {
  #ifdef UNI_DEBUG
      cytnx_error_msg(this->_blocks.size() == 0, "[ERROR][internal] empty blocks for blockform.%s",
                      "\n");
  #endif
      return this->_blocks[0].dtype();
    };
    int device() const {
  #ifdef UNI_DEBUG
      cytnx_error_msg(this->_blocks.size() == 0, "[ERROR][internal] empty blocks for blockform.%s",
                      "\n");
  #endif
      return this->_blocks[0].device();
    };
    std::string dtype_str() const {
  #ifdef UNI_DEBUG
      cytnx_error_msg(this->_blocks.size() == 0, "[ERROR][internal] empty blocks for blockform.%s",
                      "\n");
  #endif
      return this->_blocks[0].dtype_str();
    };
    std::string device_str() const {
  #ifdef UNI_DEBUG
      cytnx_error_msg(this->_blocks.size() == 0, "[ERROR][internal] empty blocks for blockform.%s",
                      "\n");
  #endif
      return this->_blocks[0].device_str();
    };

    Tensor get_block(const cytnx_uint64 &idx = 0) const {
      cytnx_error_msg(idx >= this->_blocks.size(), "[ERROR][BlockUniTensor] index out of range%s",
                      "\n");
      return this->_blocks[idx].clone();
    };

    // this one for Block will return the indicies!!
    Tensor get_block(const std::vector<cytnx_int64> &indices, const bool &force_return) const {
      cytnx_error_msg(indices.size() != this->rank(),
                      "[ERROR][get_block][BlockUniTensor] len(indices) must be the same as the "
                      "Tensor rank (number of legs).%s",
                      "\n");

      std::vector<cytnx_uint64> inds(indices.begin(), indices.end());

      // find if the indices specify exists!
      cytnx_int64 b = -1;
      for (cytnx_uint64 i = 0; i < this->_inner_to_outer_idx.size(); i++) {
        if (inds == this->_inner_to_outer_idx[i]) {
          b = i;
          break;
        }
      }

      if (b < 0) {
        if (force_return) {
          return NullRefTensor;
        } else {
          cytnx_error_msg(true,
                          "[ERROR][get_block][BlockUniTensor] no avaliable block exists, "
                          "force_return=false, so "
                          "error throws. \n    If you want to return an empty block without "
                          "error when block is "
                          "not avaliable, set force_return=True.%s",
                          "\n");
        }
      } else {
        return this->_blocks[b].clone();
      }
    }

    const Tensor &get_block_(const cytnx_uint64 &idx = 0) const {
      cytnx_error_msg(idx >= this->_blocks.size(), "[ERROR][BlockUniTensor] index out of range%s",
                      "\n");
      return this->_blocks[idx];
    };

    Tensor &get_block_(const cytnx_uint64 &idx = 0) {
      cytnx_error_msg(idx >= this->_blocks.size(), "[ERROR][BlockUniTensor] index out of range%s",
                      "\n");
      return this->_blocks[idx];
    };

    const Tensor &get_block_(const std::vector<cytnx_int64> &indices,
                             const bool &force_return) const {
      cytnx_error_msg(indices.size() != this->rank(),
                      "[ERROR][get_block][BlockUniTensor] len(indices) must be the same as the "
                      "Tensor rank (number of legs).%s",
                      "\n");

      std::vector<cytnx_uint64> inds(indices.begin(), indices.end());

      // find if the indices specify exists!
      cytnx_int64 b = -1;
      for (cytnx_uint64 i = 0; i < this->_inner_to_outer_idx.size(); i++) {
        if (inds == this->_inner_to_outer_idx[i]) {
          b = i;
          break;
        }
      }

      if (b < 0) {
        if (force_return) {
          return this->NullRefTensor;
        } else {
          cytnx_error_msg(true,
                          "[ERROR][get_block][BlockUniTensor] no avaliable block exists, "
                          "force_return=false, so "
                          "error throws. \n    If you want to return an empty block without "
                          "error when block is "
                          "not avaliable, set force_return=True.%s",
                          "\n");
        }
      } else {
        return this->_blocks[b];
      }
    }

    Tensor &get_block_(const std::vector<cytnx_int64> &indices, const bool &force_return) {
      cytnx_error_msg(indices.size() != this->rank(),
                      "[ERROR][get_block][BlockUniTensor] len(indices) must be the same as the "
                      "Tensor rank (number of legs).%s",
                      "\n");

      std::vector<cytnx_uint64> inds(indices.begin(), indices.end());

      // find if the indices specify exists!
      cytnx_int64 b = -1;
      for (cytnx_uint64 i = 0; i < this->_inner_to_outer_idx.size(); i++) {
        if (inds == this->_inner_to_outer_idx[i]) {
          b = i;
          break;
        }
      }

      if (b < 0) {
        if (force_return) {
          return this->NullRefTensor;
        } else {
          cytnx_error_msg(true,
                          "[ERROR][get_block][BlockUniTensor] no avaliable block exists, "
                          "force_return=false, so "
                          "error throws. \n    If you want to return an empty block without "
                          "error when block is "
                          "not avaliable, set force_return=True.%s",
                          "\n");
        }
      } else {
        return this->_blocks[b];
      }
    }

    std::vector<Tensor> get_blocks() const { return vec_clone(this->_blocks); }
    const std::vector<Tensor> &get_blocks_(const bool &) const { return this->_blocks; }
    std::vector<Tensor> &get_blocks_(const bool &) { return this->_blocks; }

    bool same_data(const boost::intrusive_ptr<UniTensor_base> &rhs) const {
      if (rhs->uten_type() != UTenType.Block) return false;
      if (rhs->get_blocks_(1).size() != this->get_blocks_(1).size()) return false;

      for (int i = 0; i < rhs->get_blocks_(1).size(); i++)
        if (this->get_blocks_(1)[i].same_data(rhs->get_blocks_(1)[i]) == false) return false;

      return true;
    }

    void set_rowrank_(const cytnx_uint64 &new_rowrank) {
      cytnx_error_msg(new_rowrank > this->rank(),
                      "[ERROR][BlockUniTensor] rowrank should be [>=0] and [<=UniTensor.rank].%s",
                      "\n");
      if (this->is_diag()) {
        cytnx_error_msg(new_rowrank != 1,
                        "[ERROR][BlockUniTensor] rowrank should be [==1] when is_diag =true!.%s",
                        "\n");
      }
      this->_rowrank = new_rowrank;
      this->_is_braket_form = this->_update_braket();
    }

    boost::intrusive_ptr<UniTensor_base> set_rowrank(const cytnx_uint64 &new_rowrank) const {
      BlockUniTensor *tmp = this->clone_meta(true, true);
      tmp->_blocks = this->_blocks;
      tmp->set_rowrank_(new_rowrank);
      boost::intrusive_ptr<UniTensor_base> out(tmp);
      return out;
    }

    boost::intrusive_ptr<UniTensor_base> permute(const std::vector<cytnx_int64> &mapper,
                                                 const cytnx_int64 &rowrank = -1);
    boost::intrusive_ptr<UniTensor_base> permute(const std::vector<std::string> &mapper,
                                                 const cytnx_int64 &rowrank = -1);

    void permute_(const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank = -1);
    void permute_(const std::vector<std::string> &mapper, const cytnx_int64 &rowrank = -1);

    boost::intrusive_ptr<UniTensor_base> contiguous_() {
      for (unsigned int b = 0; b < this->_blocks.size(); b++) this->_blocks[b].contiguous_();
      return boost::intrusive_ptr<UniTensor_base>(this);
    }

    boost::intrusive_ptr<UniTensor_base> contiguous();

    void print_diagram(const bool &bond_info = false);
    void print_blocks(const bool &full_info = true) const;
    void print_block(const cytnx_int64 &idx, const bool &full_info = true) const;

    boost::intrusive_ptr<UniTensor_base> contract(const boost::intrusive_ptr<UniTensor_base> &rhs,
                                                  const bool &mv_elem_self = false,
                                                  const bool &mv_elem_rhs = false);

    boost::intrusive_ptr<UniTensor_base> relabel(const std::vector<std::string> &new_labels);
    boost::intrusive_ptr<UniTensor_base> relabels(const std::vector<std::string> &new_labels);

    boost::intrusive_ptr<UniTensor_base> relabel(const std::vector<std::string> &old_labels,
                                                 const std::vector<std::string> &new_labels);
    boost::intrusive_ptr<UniTensor_base> relabels(const std::vector<std::string> &old_labels,
                                                  const std::vector<std::string> &new_labels);

    boost::intrusive_ptr<UniTensor_base> relabel(const std::string &old_label,
                                                 const std::string &new_label);
    boost::intrusive_ptr<UniTensor_base> relabel(const cytnx_int64 &inx,
                                                 const std::string &new_label);

    std::vector<Symmetry> syms() const;

    void reshape_(const std::vector<cytnx_int64> &new_shape, const cytnx_uint64 &rowrank = 0) {
      cytnx_error_msg(true, "[ERROR] cannot reshape a UniTensor with symmetry.%s", "\n");
    }
    boost::intrusive_ptr<UniTensor_base> reshape(const std::vector<cytnx_int64> &new_shape,
                                                 const cytnx_uint64 &rowrank = 0) {
      cytnx_error_msg(true, "[ERROR] cannot reshape a UniTensor with symmetry.%s", "\n");
      return nullptr;
    }

    boost::intrusive_ptr<UniTensor_base> astype(const unsigned int &dtype) const {
      BlockUniTensor *tmp = this->clone_meta(true, true);
      tmp->_blocks.resize(this->_blocks.size());
      for (cytnx_int64 blk = 0; blk < this->_blocks.size(); blk++) {
        tmp->_blocks[blk] = this->_blocks[blk].astype(dtype);
      }
      boost::intrusive_ptr<UniTensor_base> out(tmp);
      return out;
    };

    // this will only work on non-symm tensor (DenseUniTensor)
    boost::intrusive_ptr<UniTensor_base> get(const std::vector<Accessor> &accessors) {
      cytnx_error_msg(
        true,
        "[ERROR][BlockUniTensor][get] cannot use get on a UniTensor with "
        "Symmetry.\n suggestion: try get_block/get_block_/get_blocks/get_blocks_ first.%s",
        "\n");
      return nullptr;
    }

    // this will only work on non-symm tensor (DenseUniTensor)
    void set(const std::vector<Accessor> &accessors, const Tensor &rhs) {
      cytnx_error_msg(
        true,
        "[ERROR][BlockUniTensor][get] cannot use get on a UniTensor with "
        "Symmetry.\n suggestion: try get_block/get_block_/get_blocks/get_blocks_ first.%s",
        "\n");
    }

    void put_block(const Tensor &in, const cytnx_uint64 &idx = 0) {
      cytnx_error_msg(in.dtype() != this->dtype(),
                      "[ERROR][DenseUniTensor][put_block] The input tensor dtype does not match.%s",
                      "\n");
      cytnx_error_msg(in.device() != this->device(),
                      "[ERROR][DenseUniTensor][put_block] The input tensor device does not "
                      "match.%s",
                      "\n");
      // We shouldn't check the contiguous
      // cytnx_error_msg(!in.contiguous());
      cytnx_error_msg(idx >= this->_blocks.size(), "[ERROR][BlockUniTensor] index out of range%s",
                      "\n");
      cytnx_error_msg(in.shape() != this->_blocks[idx].shape(),
                      "[ERROR][BlockUniTensor] the shape of input tensor does not match the shape "
                      "of block @ idx=%d\n",
                      idx);

      this->_blocks[idx] = in.clone();
    }
    void put_block_(Tensor &in, const cytnx_uint64 &idx = 0) {
      cytnx_error_msg(in.dtype() != this->dtype(),
                      "[ERROR][DenseUniTensor][put_block] The input tensor dtype does not match.%s",
                      "\n");
      cytnx_error_msg(in.device() != this->device(),
                      "[ERROR][DenseUniTensor][put_block] The input tensor device does not "
                      "match.%s",
                      "\n");
      // We shouldn't check the contiguous
      // cytnx_error_msg(!in.contiguous());
      cytnx_error_msg(idx >= this->_blocks.size(), "[ERROR][BlockUniTensor] index out of range%s",
                      "\n");
      cytnx_error_msg(in.shape() != this->_blocks[idx].shape(),
                      "[ERROR][BlockUniTensor] the shape of input tensor does not match the shape "
                      "of block @ idx=%d\n",
                      idx);

      this->_blocks[idx] = in;
    }
    void put_block(const Tensor &in, const std::vector<cytnx_int64> &indices, const bool &check) {
      cytnx_error_msg(in.dtype() != this->dtype(),
                      "[ERROR][DenseUniTensor][put_block] The input tensor dtype does not match.%s",
                      "\n");
      cytnx_error_msg(in.device() != this->device(),
                      "[ERROR][DenseUniTensor][put_block] The input tensor device does not "
                      "match.%s",
                      "\n");
      // We shouldn't check the contiguous
      // cytnx_error_msg(!in.contiguous());
      cytnx_error_msg(indices.size() != this->rank(),
                      "[ERROR][put_block][BlockUniTensor] len(indices) must be the same as the "
                      "Tensor rank (number of legs).%s",
                      "\n");

      std::vector<cytnx_uint64> inds(indices.begin(), indices.end());

      // find if the indices specify exists!
      cytnx_int64 b = -1;
      for (cytnx_uint64 i = 0; i < this->_inner_to_outer_idx.size(); i++) {
        if (inds == this->_inner_to_outer_idx[i]) {
          b = i;
          break;
        }
      }

      if (b < 0) {
        if (check) {
          cytnx_error_msg(true,
                          "[ERROR][put_block][BlockUniTensor] no avaliable block exists, "
                          "check=true, so error throws. \n    If you want without error when block "
                          "is not avaliable, set check=false.%s",
                          "\n");
        }
      } else {
        cytnx_error_msg(
          in.shape() != this->_blocks[b].shape(),
          "[ERROR][BlockUniTensor] the shape of input tensor does not match the shape "
          "of block @ idx=%d\n",
          b);

        this->_blocks[b] = in.clone();
      }
    }
    void put_block_(Tensor &in, const std::vector<cytnx_int64> &indices, const bool &check) {
      cytnx_error_msg(in.dtype() != this->dtype(),
                      "[ERROR][DenseUniTensor][put_block] The input tensor dtype does not match.%s",
                      "\n");
      cytnx_error_msg(in.device() != this->device(),
                      "[ERROR][DenseUniTensor][put_block] The input tensor device does not "
                      "match.%s",
                      "\n");
      // We shouldn't check the contiguous
      // cytnx_error_msg(!in.contiguous());
      cytnx_error_msg(indices.size() != this->rank(),
                      "[ERROR][put_block][BlockUniTensor] len(indices) must be the same as the "
                      "Tensor rank (number of legs).%s",
                      "\n");

      std::vector<cytnx_uint64> inds(indices.begin(), indices.end());

      // find if the indices specify exists!
      cytnx_int64 b = -1;
      for (cytnx_uint64 i = 0; i < this->_inner_to_outer_idx.size(); i++) {
        if (inds == this->_inner_to_outer_idx[i]) {
          b = i;
          break;
        }
      }

      if (b < 0) {
        if (check) {
          cytnx_error_msg(true,
                          "[ERROR][put_block][BlockUniTensor] no avaliable block exists, "
                          "check=true, so error throws. \n    If you want without error when block "
                          "is not avaliable, set check=false.%s",
                          "\n");
        }
      } else {
        cytnx_error_msg(
          in.shape() != this->_blocks[b].shape(),
          "[ERROR][BlockUniTensor] the shape of input tensor does not match the shape "
          "of block @ idx=%d\n",
          b);
        this->_blocks[b] = in;
      }
    }

    void tag() {
      // no-use!
    }

    boost::intrusive_ptr<UniTensor_base> Conj() {
      boost::intrusive_ptr<UniTensor_base> out = this->clone();
      out->Conj_();
      return out;
    }

    void Conj_() {
      for (int i = 0; i < this->_blocks.size(); i++) {
        this->_blocks[i].Conj_();
      }
    };

    void Transpose_();
    boost::intrusive_ptr<UniTensor_base> Transpose() {
      boost::intrusive_ptr<UniTensor_base> out = this->clone();
      out->Transpose_();
      return out;
    }

    void normalize_();
    boost::intrusive_ptr<UniTensor_base> normalize() {
      boost::intrusive_ptr<UniTensor_base> out = this->clone();
      out->normalize_();
      return out;
    }

    boost::intrusive_ptr<UniTensor_base> Dagger() {
      boost::intrusive_ptr<UniTensor_base> out = this->Conj();
      out->Transpose_();
      return out;
    }
    void Dagger_() {
      this->Conj_();
      this->Transpose_();
    }

    void Trace_(const std::string &a, const std::string &b);
    void Trace_(const cytnx_int64 &a, const cytnx_int64 &b);

    boost::intrusive_ptr<UniTensor_base> Trace(const std::string &a, const std::string &b) {
      boost::intrusive_ptr<UniTensor_base> out = this->clone();
      out->Trace_(a, b);
      if (out->rank() == 0) {
        DenseUniTensor *tmp = new DenseUniTensor();
        tmp->_block = ((BlockUniTensor *)out.get())->_blocks[0];
        out = boost::intrusive_ptr<UniTensor_base>(tmp);
      }
      return out;
    }
    boost::intrusive_ptr<UniTensor_base> Trace(const cytnx_int64 &a, const cytnx_int64 &b) {
      boost::intrusive_ptr<UniTensor_base> out = this->clone();
      out->Trace_(a, b);
      if (out->rank() == 0) {
        DenseUniTensor *tmp = new DenseUniTensor();
        tmp->_block = ((BlockUniTensor *)out.get())->_blocks[0];
        out = boost::intrusive_ptr<UniTensor_base>(tmp);
      }
      return out;
    }

    Tensor Norm() const;

    bool elem_exists(const std::vector<cytnx_uint64> &locator) const;

    const Scalar::Sproxy at_for_sparse(const std::vector<cytnx_uint64> &locator) const;
    const cytnx_complex128 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                          const cytnx_complex128 &aux) const;
    const cytnx_complex64 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                         const cytnx_complex64 &aux) const;
    const cytnx_double &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                      const cytnx_double &aux) const;
    const cytnx_float &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                     const cytnx_float &aux) const;
    const cytnx_uint64 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                      const cytnx_uint64 &aux) const;
    const cytnx_int64 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                     const cytnx_int64 &aux) const;
    const cytnx_uint32 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                      const cytnx_uint32 &aux) const;
    const cytnx_int32 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                     const cytnx_int32 &aux) const;
    const cytnx_uint16 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                      const cytnx_uint16 &aux) const;
    const cytnx_int16 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                     const cytnx_int16 &aux) const;

    Scalar::Sproxy at_for_sparse(const std::vector<cytnx_uint64> &locator);
    cytnx_complex128 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                    const cytnx_complex128 &aux);
    cytnx_complex64 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                   const cytnx_complex64 &aux);
    cytnx_double &at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_double &aux);
    cytnx_float &at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_float &aux);
    cytnx_uint64 &at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint64 &aux);
    cytnx_int64 &at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int64 &aux);
    cytnx_uint32 &at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint32 &aux);
    cytnx_int32 &at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int32 &aux);
    cytnx_uint16 &at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint16 &aux);
    cytnx_int16 &at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int16 &aux);

    void _save_dispatch(std::fstream &f) const;
    void _load_dispatch(std::fstream &f);

    // this will remove the [q_index]-th qnum at [bond_idx]-th Bond!
    void truncate_(const std::string &bond_idx, const cytnx_uint64 &q_index);
    void truncate_(const cytnx_int64 &bond_idx, const cytnx_uint64 &q_index);

    void Add_(const boost::intrusive_ptr<UniTensor_base> &rhs);
    void Add_(const Scalar &rhs) {
      cytnx_error_msg(
        true,
        "[ERROR] cannot perform elementwise arithmetic '+' btwn Scalar and BlockUniTensor.\n %s "
        "\n",
        "This operation will destroy block structure. [Suggest] using get/set_block(s) to do "
        "operation on the block(s).");
    }

    void Mul_(const boost::intrusive_ptr<UniTensor_base> &rhs);
    void Mul_(const Scalar &rhs);

    void Sub_(const boost::intrusive_ptr<UniTensor_base> &rhs);
    void Sub_(const Scalar &rhs) {
      cytnx_error_msg(
        true,
        "[ERROR] cannot perform elementwise arithmetic '+' btwn Scalar and BlockUniTensor.\n %s "
        "\n",
        "This operation will destroy block structure. [Suggest] using get/set_block(s) to do "
        "operation on the block(s).");
    }
    void lSub_(const Scalar &lhs) {
      cytnx_error_msg(
        true,
        "[ERROR] cannot perform elementwise arithmetic '+' btwn Scalar and BlockUniTensor.\n %s "
        "\n",
        "This operation will destroy block structure. [Suggest] using get/set_block(s) to do "
        "operation on the block(s).");
    }

    void Div_(const boost::intrusive_ptr<UniTensor_base> &rhs) {
      cytnx_error_msg(
        true,
        "[ERROR] cannot perform elementwise arithmetic '+' btwn Scalar and BlockUniTensor.\n %s "
        "\n",
        "This operation will destroy block structure. [Suggest] using get/set_block(s) to do "
        "operation on the block(s).");
    }
    void Div_(const Scalar &rhs);
    void lDiv_(const Scalar &lhs) {
      cytnx_error_msg(
        true,
        "[ERROR] cannot perform elementwise arithmetic '+' btwn Scalar and BlockUniTensor.\n %s "
        "\n",
        "This operation will destroy block structure. [Suggest] using get/set_block(s) to do "
        "operation on the block(s).");
    }
    void from_(const boost::intrusive_ptr<UniTensor_base> &rhs, const bool &force);

    void group_basis_();

    void combineBond(const std::vector<std::string> &indicators, const bool &force = false);
    void combineBonds(const std::vector<cytnx_int64> &indicators, const bool &force = false);
    void combineBonds(const std::vector<cytnx_int64> &indicators, const bool &force,
                      const bool &by_label);
    void combineBonds(const std::vector<std::string> &indicators, const bool &force = false);

    const std::vector<cytnx_uint64> &get_qindices(const cytnx_uint64 &bidx) const {
      cytnx_error_msg(
        bidx >= this->Nblocks(),
        "[ERROR][BlockUniTensor] bidx out of bound! only %d blocks in current UTen.\n",
        this->Nblocks());
      return this->_inner_to_outer_idx[bidx];
    }
    std::vector<cytnx_uint64> &get_qindices(const cytnx_uint64 &bidx) {
      cytnx_error_msg(
        bidx >= this->Nblocks(),
        "[ERROR][BlockUniTensor] bidx out of bound! only %d blocks in current UTen.\n",
        this->Nblocks());
      return this->_inner_to_outer_idx[bidx];
    }

    const vec2d<cytnx_uint64> &get_itoi() const { return this->_inner_to_outer_idx; }
    vec2d<cytnx_uint64> &get_itoi() { return this->_inner_to_outer_idx; }
  };
  /// @endcond
  //======================================================================

  /// @cond
  class UniTensor_options {
   public:
    bool _is_diag;
    int _dtype;
    int _device;
    int _rowrank;

    UniTensor_options() {
      this->_is_diag = false;
      this->_dtype = Type.Double;
      this->_device = Device.cpu;
      this->_rowrank = -1;
    }

    UniTensor_options(const UniTensor_options &rhs) {
      this->_is_diag = rhs._is_diag;
      this->_dtype = rhs._dtype;
      this->_device = rhs._device;
      this->_rowrank = rhs._rowrank;
    }

    UniTensor_options &operator=(const UniTensor_options &rhs) {
      this->_is_diag = rhs._is_diag;
      this->_dtype = rhs._dtype;
      this->_device = rhs._device;
      this->_rowrank = rhs._rowrank;
      return *this;
    }

    UniTensor_options &is_diag(const bool &in) {
      this->_is_diag = in;
      return *this;
    }
    UniTensor_options &dtype(const int &in) {
      this->_dtype = in;
      return *this;
    }
    UniTensor_options &device(const int &in) {
      this->_device = in;
      return *this;
    }
    UniTensor_options &rowrank(const int &in) {
      this->_rowrank = in;
      return *this;
    }
  };
  /// @endcond

  ///@brief An Enhanced tensor specifically designed for physical Tensor network simulation
  class UniTensor {
   public:
    ///@cond
    boost::intrusive_ptr<UniTensor_base> _impl;
    UniTensor() : _impl(new UniTensor_base()){};
    UniTensor(const UniTensor &rhs) { this->_impl = rhs._impl; }
    UniTensor &operator=(const UniTensor &rhs) {
      this->_impl = rhs._impl;
      return *this;
    }
    ///@endcond

    //@{
    /**
    @brief Construct a UniTensor from a cytnx::Tensor.
    @param[in] in_tensor a cytnx::Tensor
    @param[in] is_diag Whether the input Tensor \p in_tensor is a diagonal Tensor.
                     This will requires that the input of \p in_tensor to be 1D.
    @param[in] rowrank the rowrank of the outcome UniTensor

    @note
        1. The constructed UniTensor will have same rank as the input Tensor, with default labels,
    and a shared view (shared instance) of interal block as the input Tensor.
        2. The constructed UniTensor is always untagged.
    @attention The internal block of UniTensor is a referece of input Tensor. That is, they
    share the same memory. All the change afterward on UniTensor block will change in input Tensor
    as well. Use Tensor.clone() if a shared view is not the case.

    ## Example:
    ### c++ API:
    \include example/UniTensor/fromTensor.cpp
    #### output>
    \verbinclude example/UniTensor/fromTensor.cpp.out
    ### python API:
    \include example/UniTensor/fromTensor.py
    #### output>
    \verbinclude example/UniTensor/fromTensor.py.out

    */
    explicit UniTensor(const Tensor &in_tensor, const bool &is_diag = false,
                       const cytnx_int64 &rowrank = -1,
                       const std::vector<std::string> &in_labels = {}, const std::string &name = "")
        : _impl(new UniTensor_base()) {
      this->Init(in_tensor, is_diag, rowrank, in_labels, name);
    }
    /**
    @brief Initialize a UniTensor with cytnx::Tensor.
    @param[in] in_tensor a cytnx::Tensor
    @param[in] is_diag Whether the input Tensor \p in_tensor is a diagonal Tensor.
                     This will requires that the input of \p in_tensor to be 1D.
    @param[in] rowrank the rowrank of the outcome UniTensor.
    @param[in] name user specified name of the UniTensor.

    @note
        1. The constructed UniTensor will have same rank as the input Tensor, with default labels,
    and a shared view (shared instance) of interal block as the input Tensor.
        2. The constructed UniTensor is always untagged.
    @attention The internal block of UniTensor is a referece of input Tensor. That is, they
    share the same memory. All the change afterward on UniTensor block will change in input Tensor
    as well. Use Tensor.clone() if a shared view is not the case.
        @see UniTensor(const Tensor &, const bool &, const cytnx_int64 &)
    */
    void Init(const Tensor &in_tensor, const bool &is_diag = false, const cytnx_int64 &rowrank = -1,
              const std::vector<std::string> &in_labels = {}, const std::string &name = "") {
      // std::cout << "[entry!]" << std::endl;
      boost::intrusive_ptr<UniTensor_base> out(new DenseUniTensor());
      out->Init_by_Tensor(in_tensor, is_diag, rowrank, name);
      this->_impl = out;
      if (in_labels.size() != 0) this->set_labels(in_labels);
    }
    //@}

    //@{
    /**
    @brief Construct a UniTensor.
    @param[in] bonds the bond list. Each bond will be deep copy( not a shared view of bond object
    with input bond)
    @param[in] in_labels the labels for each rank(bond)
    @param[in] rowrank the rank of physical row space.
    @param[in] dtype the data type of the UniTensor. It can be any type defined in cytnx::Type.
    @param[in] device the device that the UniTensor is put on. It can be any device defined in
    cytnx::Device.
    @param[in] is_diag if the constructed UniTensor is a diagonal UniTensor.
        This can only be assigned true when the UniTensor is square and rank-2 UniTensor.
        The UniTensor must have one in-bond and one out-bond.
    @pre
        1. the bonds cannot contain simutaneously untagged bond(s) and tagged bond(s)
        2. If the bonds are with symmetry (qnums), the symmetry types should be the same across
    all the bonds.
    */
    UniTensor(const std::vector<Bond> &bonds, const std::vector<std::string> &in_labels = {},
              const cytnx_int64 &rowrank = -1, const unsigned int &dtype = Type.Double,
              const int &device = Device.cpu, const bool &is_diag = false,
              const std::string &name = "")
        : _impl(new UniTensor_base()) {
  #ifdef UNI_DEBUG
      cytnx_warning_msg(
        true,
        "[DEBUG] message: entry for UniTensor(const std::vector<Bond> &bonds, const "
        "std::vector<std::string> &in_labels={}, const cytnx_int64 &rowrank=-1, const unsigned "
        "int "
        "&dtype=Type.Double, const int &device = Device.cpu, const bool &is_diag=false)%s",
        "\n");
  #endif
      this->Init(bonds, in_labels, rowrank, dtype, device, is_diag, name);
    }

    /// @cond
    /* [developing]
    void Init(const std::vector<Bond> &bonds, const std::vector<std::string> &in_labels = {},
              const UniTensor_options &UToptions = UniTensor_options(), const std::string &name =
    ""){ this->Init(bonds,in_labels, UToptions._rowrank, UToptions._dtype  , UToptions._device ,
                                    UToptions._is_diag,
                                    name);
    }
    */
    /// @endcond

    /**
    @brief Initialize the UniTensor with the given arguments.
        @details This is the initial function of the UniTensor. If you want to initialize
           your UniTensor after declaring just a 'void' UniTensor. You can use this
       function to  initialize it.
    @param[in] bonds the bond list. Each bond will be deep copy( not a shared view of
           bond object with input bond)
    @param[in] in_labels the labels for each rank(bond)
    @param[in] rowrank the rank of physical row space
    @param[in] dtype the dtype of the UniTensor. It can be any type defined in cytnx::Type.
    @param[in] device the device that the UniTensor is put on. It can be any device defined in
    cytnx::Device.
    @param[in] is_diag if the constructed UniTensor is a diagonal UniTensor. This can
           only be assigned true when the UniTensor is square, untagged and rank-2
           UniTensor.
    @param[in] name user specified name of the UniTensor.
        @pre Please ensure that all of the Bond in \p bonds should be all symmetric or
          non-symmetric. You cannot mix them.
        @see
    UniTensor(const std::vector<Bond> &bonds, const std::vector<std::string> &in_labels,
              const cytnx_int64 &rowrank, const unsigned int &dtype, const int &device,
                          const bool &is_diag)
    */
    void Init(const std::vector<Bond> &bonds, const std::vector<std::string> &in_labels = {},
              const cytnx_int64 &rowrank = -1, const unsigned int &dtype = Type.Double,
              const int &device = Device.cpu, const bool &is_diag = false,
              const std::string &name = "") {
      // checking type:
      bool is_sym = false;
      int sym_fver = -1;

      for (cytnx_uint64 i = 0; i < bonds.size(); i++) {
        // check
        if (bonds[i].syms().size() != 0) {
          is_sym = true;
          if (sym_fver == -1)
            sym_fver = bonds[i]._impl->_degs.size();
          else {
            // std::cout << sym_fver << " " <<
            // bonds[i]._impl->_degs.size() << std::endl;
            cytnx_error_msg((bool(sym_fver) ^ bool(bonds[i]._impl->_degs.size())),
                            "[ERROR] All the Bond when init a UniTensor with symmetric must be in "
                            "the same format!%s",
                            "\n");
          }
        } else
          cytnx_error_msg(
            is_sym, "[ERROR] cannot have bonds with mixing of symmetry and non-symmetry.%s", "\n");
      }

      // dynamical dispatch:
      if (is_sym) {
  #ifdef UNI_DEBUG
        cytnx_warning_msg(true, "[DEBUG] message: entry dispatch: UniTensor: symmetric%s", "\n");
  #endif
        // cytnx_warning_msg(true,"[warning, still developing, some functions will display
        // \"[Developing]\"][SparseUniTensor]%s","\n");
        if (sym_fver == 0) {
          // boost::intrusive_ptr<UniTensor_base> out(new SparseUniTensor());
          // this->_impl = out;
          cytnx_error_msg(true,
                          "[ERROR] internal error! [legacy Sparse entry] the Bond is symmetry but "
                          "the version is not properly determined!%s",
                          "\n")
        } else if (sym_fver == -1) {
          cytnx_error_msg(true,
                          "[ERROR] internal error! the Bond is symmetry but the version is not "
                          "properly determined!%s",
                          "\n");
        } else {
          boost::intrusive_ptr<UniTensor_base> out(new BlockUniTensor());
          this->_impl = out;
        }
      } else {
        boost::intrusive_ptr<UniTensor_base> out(new DenseUniTensor());
        this->_impl = out;
      }
      this->_impl->Init(bonds, in_labels, rowrank, dtype, device, is_diag, false, name);
    }

    /**
    @brief Set the name of the UniTensor.
        @details You can use this function to give a name for the UniTensor.
    @param[in] in Input the name you want to set for the UniTensor. It should be a string.
        @return UniTensor
    */
    UniTensor &set_name(const std::string &in) {
      this->_impl->set_name(in);
      return *this;
    }

    /**
    @brief Set a new label for bond at the assigned index.
    @param[in] idx the index of the bond.
    @param[in] new_label the new label that is assign to the bond.
    @note
        1. the new assign label cannot be the same as the label of any other bonds in the
    UniTensor. ( cannot have duplicate labels )
        2. Compare to relabel(const cytnx_int64 &idx, const std::string &new_label) const,
        this function set the new label and return self.
    */
    UniTensor &set_label(const cytnx_int64 &idx, const std::string &new_label) {
      this->_impl->set_label(idx, new_label);
      return *this;
    }

    /**
          @see set_label(const cytnx_int64 &idx, const std::string &new_label)
    */
    UniTensor &set_label(const cytnx_int64 &idx, const char *new_label) {
      this->_impl->set_label(idx, std::string(new_label));
      return *this;
    }

    /**
    @brief set a new label for bond to replace one of the current label.
    @param[in] old_label the current label of the bond.
    @param[in] new_label the new label that is assign to the bond.
    @note
        1. the new assign label cannot be the same as the label of any other bonds in the
    UniTensor. ( cannot have duplicate labels )
        2. Compare to relabel(const std::string &old_label, const std::string &new_label) const,
        this function set the new label and return self.
    */
    UniTensor &set_label(const std::string &old_label, const std::string &new_label) {
      this->_impl->set_label(old_label, new_label);
      return *this;
    }

    /**
          @see set_label(const std::string &old_label, const std::string &new_label)
    */
    UniTensor &set_label(const char *old_label, const std::string &new_label) {
      this->_impl->set_label(std::string(old_label), new_label);
      return *this;
    }

    /**
          @see set_label(const std::string &old_label, const std::string &new_label)
    */
    UniTensor &set_label(const std::string &old_label, const char *new_label) {
      this->_impl->set_label(old_label, std::string(new_label));
      return *this;
    }

    /**
          @see set_label(const std::string &old_label, const std::string &new_label)
    */
    UniTensor &set_label(const char *old_label, const char *new_label) {
      this->_impl->set_label(std::string(old_label), std::string(new_label));
      return *this;
    }

    /*
    UniTensor& change_label(const cytnx_int64 &old_label, const cytnx_int64 &new_label){
        this->_impl->change_label(old_label,new_label);
        return *this;
    }
    */

    /**
    @brief Set new labels for all the bonds.
    @param[in] new_labels the new labels for each bond.
    @note
        1. the new assign label cannot be the same as the label of any other bonds in the
    UniTensor. ( cannot have duplicate labels )
        2. Compare to relabels(const std::vector<std::string> &new_labels) const, this
        function set the new label and return self.
    */
    UniTensor &set_labels(const std::vector<std::string> &new_labels) {
      this->_impl->set_labels(new_labels);
      return *this;
    }

    /**
          @see
    set_labels(const std::vector<std::string> &new_labels)
         */
    UniTensor &set_labels(const std::initializer_list<char *> &new_labels) {
      std::vector<char *> new_lbls(new_labels);
      std::vector<std::string> vs(new_lbls.size());
      transform(new_lbls.begin(), new_lbls.end(), vs.begin(),
                [](char *x) -> std::string { return std::string(x); });

      this->_impl->set_labels(vs);
      return *this;
    }

    /**
    @brief Set the row rank of the UniTensor.
        @details You can use this function to set the row rank of the UniTensor. The row rank is
          important if you want to use the linear algebra process.
    @param[in] new_rowrank the new row rank of the UniTensor
    */
    UniTensor &set_rowrank_(const cytnx_uint64 &new_rowrank) {
      this->_impl->set_rowrank_(new_rowrank);
      return *this;
    }

    UniTensor set_rowrank(const cytnx_uint64 &new_rowrank) const {
      UniTensor out;
      out._impl = this->_impl->set_rowrank(new_rowrank);
      return out;
    }

    template <class T>
    T &item() {
      cytnx_error_msg(this->is_blockform(),
                      "[ERROR] cannot use item on UniTensor with Symmetry.\n suggestion: use "
                      "get_block()/get_blocks() first.%s",
                      "\n");

      DenseUniTensor *tmp = static_cast<DenseUniTensor *>(this->_impl.get());
      return tmp->_block.item<T>();
    }

    Scalar::Sproxy item() const {
      cytnx_error_msg(this->is_blockform(),
                      "[ERROR] cannot use item on UniTensor with Symmetry.\n suggestion: use "
                      "get_block()/get_blocks() first.%s",
                      "\n");

      DenseUniTensor *tmp = static_cast<DenseUniTensor *>(this->_impl.get());
      return tmp->_block.item();
    }
    /**
    @brief Return the number of blocks in the UniTensor.
    @return cytnx_uint64
    */
    cytnx_uint64 Nblocks() const { return this->_impl->Nblocks(); }

    /**
    @brief Return the rank of the UniTensor.
    @return cytnx_uint64
    */
    cytnx_uint64 rank() const { return this->_impl->rank(); }

    /**
    @brief Return the row rank of the UniTensor.
    @return cytnx_uint64
    */
    cytnx_uint64 rowrank() const { return this->_impl->rowrank(); }

    /**
    @brief Return the data type of the UniTensor.
        @details The function return the data type of the UniTensor.
    @return unsigned int
    */
    unsigned int dtype() const { return this->_impl->dtype(); }

    /**
    @brief Return the UniTensor type (cytnx::UTenType) of the UniTensor.
        @details The function return the UniTensor type of the UniTensor.
    @return int
        @see uten_type_str()
    */
    int uten_type() const { return this->_impl->uten_type(); }

    /**
    @brief Return the device of the UniTensor.
        @details The function return the device of the UniTensor.
    @return int
    */
    int device() const { return this->_impl->device(); }

    /**
    @brief Return the name of the UniTensor.
    @return std::string
    */
    std::string name() const { return this->_impl->name(); }

    /**
    @brief Return the data type of the UniTensor in 'string' form.
    @return std::string
        @see dtype()
    */
    std::string dtype_str() const { return this->_impl->dtype_str(); }

    /**
    @brief Return the device of the UniTensor in 'string' form.
    @return std::string
        @see device()
    */
    std::string device_str() const { return this->_impl->device_str(); }

    /**
    @brief Return the UniTensor type (cytnx::UTenType) of the UniTensor in 'string' form.
    @return std::string
        @see uten_type()
    */
    std::string uten_type_str() const { return this->_impl->uten_type_str(); }

    /**
    @brief To tell whether the UniTensor is contiguous.
    @return bool
        @see contiguous(), contiguous_()
    */
    bool is_contiguous() const { return this->_impl->is_contiguous(); }

    /**
    @brief To tell whether the UniTensor is in diagonal form.
    @return bool
    */
    bool is_diag() const { return this->_impl->is_diag(); }

    /**
    @brief To tell whether the UniTensor is tagged. That is, all of the Bond
    in the UniTensor is directional (all of them are bondType.BD_KET or bondType.BD_BRA)
    @return bool
    */
    bool is_tag() const { return this->_impl->is_tag(); }

    /**
    @brief Return the symmetry type of the UniTensor.
        @details We can get the Symmetry structure by calling this function.
    @return std::vector<Symmetry>
    */
    std::vector<Symmetry> syms() const { return this->_impl->syms(); }

    /**
     * @brief Check whether the UniTensor is in braket form.
     * @details This function will check whether the UniTensor is in braket form. That is,
     * 1. The bonds order, all of the in-bonds are before out-bonds.
     * 2. The number of in-bonds is equal to the rowrank.
     */
    const bool &is_braket_form() const { return this->_impl->is_braket_form(); }

    /**
    @brief Return the labels of the UniTensor.
    @return std::vector<std::string>
    */
    const std::vector<std::string> &labels() const { return this->_impl->labels(); }
    /**
     * @brief Get the index of an desired label string
     *
     * @param label Label you want to find
     * @return The index of the label. If not found, return -1
     */
    cytnx_int64 get_index(std::string label) const { return this->_impl->get_index(label); }

    /**
    @brief Get the bonds of the UniTensor.
    @return std::vector<Bond>
    */
    const std::vector<Bond> &bonds() const { return this->_impl->bonds(); }

    /**
        @see bonds();
    */
    std::vector<Bond> &bonds() { return this->_impl->bonds(); }

    const Bond &bond_(const cytnx_uint64 &idx) const { return this->_impl->bond_(idx); }
    Bond &bond_(const cytnx_uint64 &idx) { return this->_impl->bond_(idx); }

    const Bond &bond_(const std::string &label) const { return this->_impl->bond_(label); }
    Bond &bond_(const std::string &label) { return this->_impl->bond_(label); }

    Bond bond(const cytnx_uint64 &idx) const { return this->_impl->bond_(idx).clone(); }
    Bond bond(const std::string &label) const { return this->_impl->bond_(label).clone(); }

    /**
    @brief Get the shape of the UniTensor.
    @return std::vector<cytnx_uint64>
    */
    std::vector<cytnx_uint64> shape() const { return this->_impl->shape(); }

    /**
     * @brief Check whether the UniTensor is in block form.
     * @details This function will check whether the UniTensor is in block form. That is,
     * the uten_type() is UTenType::Block.
     */
    bool is_blockform() const { return this->_impl->is_blockform(); }

    /**
    @brief move the current UniTensor to the assigned device (inplace).
        @param[in] device the device-id(@ref cytnx::Device) that is moving to. It can by
            any device defined in cytnx::Device.
        @see to_(const int &device)
    */
    void to_(const int &device) { this->_impl->to_(device); }

    /**
    @brief move the current UniTensor to the assigned device.
        @warning if the device-id is the same as current Tensor's device, then return self.
          otherwise, return a copy of instance that located on the target device.
        @param[in] device the device-id(@ref cytnx::Device) that is moving to. It can by
        any device defined in cytnx::Device.
        @return UniTensor
        @see to(const int &device)
    */
    UniTensor to(const int &device) const {
      UniTensor out;
      out._impl = this->_impl->to(device);
      return out;
    }

    /**
    @brief Clone (deep copy) the UniTensor.
        @return UniTensor
    */
    UniTensor clone() const {
      UniTensor out;
      out._impl = this->_impl->clone();
      return out;
    }

    /**
    @brief Set new labels for all the bonds.
    @param[in] new_labels the new labels for each bond.
    @note
        1. the new assign label cannot be the same as the label of any other bonds in the
    UniTensor. ( cannot have duplicate labels )
        2. Compare to relabel(const std::vector<std::string> &new_labels) const, this
        function set the new label to itself.
    */
    UniTensor &relabel_(const std::vector<std::string> &new_labels) {
      this->_impl->relabel_(new_labels);
      return *this;
    }
    /**
    @deprecated
    @brief Set new labels for all the bonds.
    @param[in] new_labels the new labels for each bond.
    @note
        1. the new assign label cannot be the same as the label of any other bonds in the
    UniTensor. ( cannot have duplicate labels )
        2. Compare to relabels(const std::vector<std::string> &new_labels) const, this
        function set the new label to itself.
    */
    UniTensor &relabels_(const std::vector<std::string> &new_labels) {
      this->_impl->relabels_(new_labels);
      return *this;
    }

    /**
    @brief relable all of the labels in UniTensor.
    @param[in] new_labels the new labels for each bond.
    @note
        1. the new assign label cannot be the same as the label of any other bonds in the
    UniTensor. ( cannot have duplicate labels )

    @attention This function will return a new UniTensor with the new label, but the data is
    still shared with the original UniTensor. That is the meta data of the UniTensor is
    different, but the internal data is still shared.
     */
    UniTensor relabel(const std::vector<std::string> &new_labels) const {
      UniTensor out;
      out._impl = this->_impl->relabel(new_labels);
      return out;
    }
    /**
    @deprecated
    @brief relables all of the labels in UniTensor.
    @param[in] new_labels the new labels for each bond.
    @note
        1. the new assign label cannot be the same as the label of any other bonds in the
    UniTensor. ( cannot have duplicate labels )

    @attention This function will return a new UniTensor with the new label, but the data is
    still shared with the original UniTensor. That is the meta data of the UniTensor is
    different, but the internal data is still shared.
     */
    UniTensor relabels(const std::vector<std::string> &new_labels) const {
      UniTensor out;
      out._impl = this->_impl->relabels(new_labels);
      return out;
    }

    /**
    @see relabel(const std::vector<std::string> &new_labels) const
     */
    UniTensor relabel(const std::initializer_list<char *> &new_labels) const {
      std::vector<char *> new_lbls(new_labels);
      std::vector<std::string> vs(new_lbls.size());
      transform(new_lbls.begin(), new_lbls.end(), vs.begin(),
                [](char *x) -> std::string { return std::string(x); });

      UniTensor out;
      out._impl = this->_impl->relabel(vs);
      return out;
    }
    /**
    @deprecated
    @see relabels(const std::vector<std::string> &new_labels) const
     */
    UniTensor relabels(const std::initializer_list<char *> &new_labels) const {
      std::vector<char *> new_lbls(new_labels);
      std::vector<std::string> vs(new_lbls.size());
      transform(new_lbls.begin(), new_lbls.end(), vs.begin(),
                [](char *x) -> std::string { return std::string(x); });

      UniTensor out;
      out._impl = this->_impl->relabels(vs);
      return out;
    }
    /**
    @see relabel_(const std::vector<std::string> &new_labels)
     */
    UniTensor &relabel_(const std::initializer_list<char *> &new_labels) {
      std::vector<char *> new_lbls(new_labels);
      std::vector<std::string> vs(new_lbls.size());
      transform(new_lbls.begin(), new_lbls.end(), vs.begin(),
                [](char *x) -> std::string { return std::string(x); });

      this->_impl->relabel_(vs);
      return *this;
    }
    /**
    @deprecated
    @see relabels_(const std::vector<std::string> &new_labels)
     */
    UniTensor &relabels_(const std::initializer_list<char *> &new_labels) {
      std::vector<char *> new_lbls(new_labels);
      std::vector<std::string> vs(new_lbls.size());
      transform(new_lbls.begin(), new_lbls.end(), vs.begin(),
                [](char *x) -> std::string { return std::string(x); });

      this->_impl->relabels_(vs);
      return *this;
    }

    /**
    @brief replace part or all labels by given new labels for the bonds.
    @param[in] old_labels the old labels for each bond.
    @param[in] new_labels the new labels for each bond.
    @note
        1. the final output UniTensor cannot have duplicate labels.
    */
    UniTensor relabel(const std::vector<std::string> &old_labels,
                      const std::vector<std::string> &new_labels) const {
      UniTensor out;
      out._impl = this->_impl->relabel(old_labels, new_labels);
      return out;
    }
    /**
    @deprecated
    @brief replace part or all labels by given new labels for the bonds.
    @param[in] old_labels the old labels for each bond.
    @param[in] new_labels the new labels for each bond.
    @note
        1. the final output UniTensor cannot have duplicate labels.
    */
    UniTensor relabels(const std::vector<std::string> &old_labels,
                       const std::vector<std::string> &new_labels) const {
      UniTensor out;
      out._impl = this->_impl->relabels(old_labels, new_labels);
      return out;
    }

    /**
    @brief relable part or all of the labels in UniTensor by given new labels
    @param[in] old_labels the old labels for each bond.
    @param[in] new_labels the new labels for each bond.
    @note
        1. the final output UniTensor cannot have duplicate labels.
        2. Compare to relabel(const std::vector<std::string> &old_labels,  const
    std::vector<std::string> &new_labels) const , this function set the new label(s) to itself.

    @see relabel(const std::vector<std::string> &old_labels, const std::vector<std::string>
    &new_labels) const
    @attention This function will return a new UniTensor with the new labels, but the data is
    still shared with the original UniTensor. That is the meta data of the UniTensor is
    different, but the internal data is still shared.
     */
    UniTensor &relabel_(const std::vector<std::string> &old_labels,
                        const std::vector<std::string> &new_labels) {
      this->_impl->relabel_(old_labels, new_labels);
      return *this;
    }
    /**
    @deprecated
    @brief relables part or all of the labels in UniTensor by given new labels
    @param[in] old_labels the old labels for each bond.
    @param[in] new_labels the new labels for each bond.
    @note
        1. the final output UniTensor cannot have duplicate labels.
        2. Compare to relabels(const std::vector<std::string> &old_labels,  const
    std::vector<std::string> &new_labels) const , this function set the new label(s) to itself.

    @see relabels(const std::vector<std::string> &old_labels, const std::vector<std::string>
    &new_labels) const
    @attention This function will return a new UniTensor with the new labels, but the data is
    still shared with the original UniTensor. That is the meta data of the UniTensor is
    different, but the internal data is still shared.
     */
    UniTensor &relabels_(const std::vector<std::string> &old_labels,
                         const std::vector<std::string> &new_labels) {
      this->_impl->relabels_(old_labels, new_labels);
      return *this;
    }

    /**
    @see relabel(const std::vector<std::string> &old_labels, const std::vector<std::string>
    &new_labels) const
     */
    UniTensor relabel(const std::initializer_list<char *> &old_labels,
                      const std::initializer_list<char *> &new_labels) const {
      std::vector<char *> new_lbls(new_labels);
      std::vector<std::string> vs(new_lbls.size());
      transform(new_lbls.begin(), new_lbls.end(), vs.begin(),
                [](char *x) -> std::string { return std::string(x); });

      std::vector<char *> old_lbls(old_labels);
      std::vector<std::string> vs_old(old_lbls.size());
      transform(old_lbls.begin(), old_lbls.end(), vs_old.begin(),
                [](char *x) -> std::string { return std::string(x); });

      return this->relabel(vs_old, vs);
    }

    /**
    @deprecated
    @see relabels(const std::vector<std::string> &old_labels, const std::vector<std::string>
    &new_labels) const
     */
    UniTensor relabels(const std::initializer_list<char *> &old_labels,
                       const std::initializer_list<char *> &new_labels) const {
      std::vector<char *> new_lbls(new_labels);
      std::vector<std::string> vs(new_lbls.size());
      transform(new_lbls.begin(), new_lbls.end(), vs.begin(),
                [](char *x) -> std::string { return std::string(x); });

      std::vector<char *> old_lbls(old_labels);
      std::vector<std::string> vs_old(old_lbls.size());
      transform(old_lbls.begin(), old_lbls.end(), vs_old.begin(),
                [](char *x) -> std::string { return std::string(x); });

      return this->relabels(vs_old, vs);
    }

    /**
    @see relabel_(const std::vector<std::string> &old_labels, const std::vector<std::string>
    &new_labels)
     */
    UniTensor &relabel_(const std::initializer_list<char *> &old_labels,
                        const std::initializer_list<char *> &new_labels) {
      std::vector<char *> new_lbls(new_labels);
      std::vector<std::string> vs(new_lbls.size());
      transform(new_lbls.begin(), new_lbls.end(), vs.begin(),
                [](char *x) -> std::string { return std::string(x); });

      std::vector<char *> old_lbls(old_labels);
      std::vector<std::string> vs_old(old_lbls.size());
      transform(old_lbls.begin(), old_lbls.end(), vs_old.begin(),
                [](char *x) -> std::string { return std::string(x); });

      this->relabel_(vs_old, vs);
      return *this;
    }
    /**
    @deprecated
    @see relabels_(const std::vector<std::string> &old_labels, const std::vector<std::string>
    &new_labels)
     */
    UniTensor &relabels_(const std::initializer_list<char *> &old_labels,
                         const std::initializer_list<char *> &new_labels) {
      std::vector<char *> new_lbls(new_labels);
      std::vector<std::string> vs(new_lbls.size());
      transform(new_lbls.begin(), new_lbls.end(), vs.begin(),
                [](char *x) -> std::string { return std::string(x); });

      std::vector<char *> old_lbls(old_labels);
      std::vector<std::string> vs_old(old_lbls.size());
      transform(old_lbls.begin(), old_lbls.end(), vs_old.begin(),
                [](char *x) -> std::string { return std::string(x); });

      this->relabels_(vs_old, vs);
      return *this;
    }

    /**
    @brief rebable the lags in the UniTensor by given index.
    @details This function will relabel the lags in the UniTensor by a given index.
    This function will return a new UniTensor with the new label, but the data is
    still shared with the original UniTensor (that is if you use same_data() to check,
    it will return true). Compare to set_label(), this function will return a new
    UniTensor object with different meta data.
        @param[in] inx a given index
        @param[in] new_label the new label of the UniTensor in the index \p inx
    @attention This function will return a new UniTensor with the new label, but the data is
    still shared with the original UniTensor. That is the meta data of the UniTensor is
    different, but the internal data is still shared.
    @see set_label(const cytnx_int64 &idx, const std::string &new_label)
     */
    UniTensor relabel(const cytnx_int64 &inx, const std::string &new_label) const {
      UniTensor out;
      out._impl = this->_impl->relabel(inx, new_label);
      return out;
    }
    /**
    @brief rebable the lags in the UniTensor by given index.
    @details This function will relabel the lags in the UniTensor by a given index.
    This function will alter the label of the UniTensor directly.
        @param[in] inx a given index
        @param[in] new_label the new label of the UniTensor in the index \p inx
    @see set_label(const cytnx_int64 &idx, const std::string &new_label)
     */
    UniTensor &relabel_(const cytnx_int64 &inx, const std::string &new_label) {
      this->_impl->relabel_(inx, new_label);
      return *this;
    }

    /**
    @brief relabel the lags in the UniTensor by a given label.
    @details This function will relabel the lags in the UniTensor by a given label.
    This function will alter the label of the UniTensor directly.
          @param[in] old_label original label you want to replace
          @param[in] new_label the new label
    @see set_label(const std::string &old_label, const std::string &new_label)
     */
    UniTensor &relabel_(const std::string &old_label, const std::string &new_label) {
      this->_impl->relabel_(old_label, new_label);
      return *this;
    }

    /**
    @brief relabel the lags in the UniTensor by a given label.
    @details This function will relabel the lags in the UniTensor by a given label.
    This function will return a new UniTensor with the new label, but the data is
    still shared with the original UniTensor (that is if you use same_data() to check,
    it will return true). Compare to set_label(), this function will return a new
    UniTensor object with different meta data.
          @param[in] old_label original label you want to replace
          @param[in] new_label the new label
    @attention This function will return a new UniTensor with the new label, but the data is
    still shared with the original UniTensor. That is the meta data of the UniTensor is
    different, but the internal data is still shared.
    @see set_label(const std::string &old_label, const std::string &new_label)
     */
    UniTensor relabel(const std::string &old_label, const std::string &new_label) const {
      UniTensor out;
      out._impl = this->_impl->relabel(old_label, new_label);
      return out;
    }

    /**
    @brief Return a new UniTensor that cast to different data type.
        @param[in] new_type the new data type. It an be any type defined in cytnx::Type.
        @return UniTensor
        @attention If the \p new_type is same as dtype of the original UniTensor, return self.
     */
    UniTensor astype(const unsigned int &dtype) const {
      UniTensor out;
      if (this->dtype() == dtype) {
        out._impl = this->_impl;
      } else {
        out._impl = this->_impl->astype(dtype);
      }
      return out;
    }

    /**
     * @brief permute the lags of the UniTensor
     * @param[in] mapper the mapper of the permutation. This mapper is mapped by bond index if
     *    \p by_label is false, otherwise it is mapped by bond label.
     * @param[in] rowrank the new rowrank after the permutation
     * @return UniTensor
     * @warning \p by_label will be deprecated!
     */
    UniTensor permute(const std::vector<cytnx_int64> &mapper,
                      const cytnx_int64 &rowrank = -1) const {
      UniTensor out;
      out._impl = this->_impl->permute(mapper, rowrank);
      return out;
    }

    /**
     * @brief permute the lags of the UniTensor by labels
     * @param[in] mapper the mapper by babels
     * @param[in] rowrank the row rank
     * @return UniTensor
     */
    UniTensor permute(const std::vector<std::string> &mapper,
                      const cytnx_int64 &rowrank = -1) const {
      UniTensor out;
      out._impl = this->_impl->permute(mapper, rowrank);
      return out;
    }

    /**
        @see permute(const std::vector<std::string> &mapper, const cytnx_int64 &rowrank = -1)
        */
    UniTensor permute(const std::initializer_list<char *> &mapper,
                      const cytnx_int64 &rowrank = -1) const {
      std::vector<char *> mprs = mapper;
      std::vector<std::string> vs(mprs.size());
      transform(mprs.begin(), mprs.end(), vs.begin(),
                [](char *x) -> std::string { return std::string(x); });

      return this->permute(vs, rowrank);
    }

    /**
    @brief permute the lags of the UniTensor, inplacely.
    @deprecated It is recommended to use \ref
      permute_(const std::vector<std::string> &mapper, const cytnx_int64 &rowrank = -1)
    @param[in] mapper the mapper by labels
    @param[in] rowrank the row rank after the permutation
          @warning \p by_label will be deprecated!
    */
    void permute_(const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank = -1) {
      this->_impl->permute_(mapper, rowrank);
    }

    /**
    @brief permute the lags of the UniTensor, inplacely.
    @param[in] mapper the mapper by labels
    @param[in] rowrank the row rank after the permutation
        @see permute(const std::vector<std::string> &mapper, const cytnx_int64 &rowrank = -1)
    */
    void permute_(const std::vector<std::string> &mapper, const cytnx_int64 &rowrank = -1) {
      this->_impl->permute_(mapper, rowrank);
    }

    // void permute_( const std::initializer_list<char*> &mapper, const cytnx_int64 &rowrank= -1){
    //     std::vector<char*> mprs = mapper;
    //     std::vector<std::string> vs(mprs.size());
    //     transform(mprs.begin(),mprs.end(),vs.begin(),[](char * x) -> std::string { return
    //     std::string(x); });

    //     this->permute_(vs,rowrank);
    // }

    // void permute_(const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank = -1) {
    //   this->_impl->permute_(mapper, rowrank);
    // }

    /**
    @brief Make the UniTensor contiguous by coalescing the memory (storage).
        @see contiguous_()
    */
    UniTensor contiguous() const {
      UniTensor out;
      out._impl = this->_impl->contiguous();
      return out;
    }

    /**
    @brief Make the UniTensor contiguous by coalescing the memory (storage), inplacely.
        @see contiguous()
    */
    void contiguous_() { this->_impl = this->_impl->contiguous_(); }

    /**
    @brief Plot the diagram of the UniTensor.
        @param[in] bond_info whether need to print the information of the bonds of the UniTensor.
    */
    void print_diagram(const bool &bond_info = false) { this->_impl->print_diagram(bond_info); }

    /**
    @brief Print all of the blocks in the UniTensor.
        @param[in] full_info whether need to print the full information of the blocks
    */
    void print_blocks(const bool &full_info = true) const { this->_impl->print_blocks(full_info); }

    /**
    @brief Given a index and print out the corresponding block of the UniTensor.
        @param[in] idx the input index
        @param[in] full_info whether need to print the full information of the block
    */
    void print_block(const cytnx_int64 &idx, const bool &full_info = true) const {
      this->_impl->print_block(idx, full_info);
    }

    /**
    @brief Group the same quantum number basis together.
    @details If the UniTensor has duplicated quantum number, then this function will
        group the basis with the same quantum number.
    @pre The UniTensor must be in block form. That is, the UniTensor is UTenType::Block.
    */
    void group_basis_() { this->_impl->group_basis_(); }

    UniTensor group_basis() const {
      UniTensor out = this->clone();
      out.group_basis_();
      return out;
    }

    /**
    @brief Get an element at specific location.
        @param[in] locator the location of the element we want to access.
        @note this API is only for C++.
    */
    template <class T>
    T &at(const std::vector<cytnx_uint64> &locator) {
      // std::cout << "at " << this->is_blockform()  << std::endl;
      if (this->uten_type() == UTenType.Block) {
        // [NEW] this will not check if it exists, if it is not then error will throw!
        T aux;
        return this->_impl->at_for_sparse(locator, aux);

      } else if (this->uten_type() == UTenType.Sparse) {
        if (this->_impl->elem_exists(locator)) {
          T aux;
          return this->_impl->at_for_sparse(locator, aux);
        } else {
          cytnx_error_msg(true, "[ERROR][SparseUniTensor] invalid location. break qnum block.%s",
                          "\n");
        }
      } else {
        return this->get_block_().at<T>(locator);
      }
    }

    /**
    @brief Get an element at specific location.
        @param[in] locator the location of the element we want to access.
        @note this API is only for C++.
    */
    template <class T>
    const T &at(const std::vector<cytnx_uint64> &locator) const {
      // std::cout << "at " << this->is_blockform()  << std::endl;
      if (this->uten_type() == UTenType.Block) {
        // [NEW] this will not check if it exists, if it is not then error will throw!
        T aux;
        return this->_impl->at_for_sparse(locator, aux);

      } else if (this->uten_type() == UTenType.Sparse) {
        if (this->_impl->elem_exists(locator)) {
          T aux;  // [workaround] use aux to dispatch.
          return this->_impl->at_for_sparse(locator, aux);
        } else {
          cytnx_error_msg(true, "[ERROR][SparseUniTensor] invalid location. break qnum block.%s",
                          "\n");
        }
      } else {
        return this->get_block_().at<T>(locator);
      }
    }

    template <class T>
    const T &at(const std::vector<std::string> &labels,
                const std::vector<cytnx_uint64> &locator) const {
      // giving label <-> locator one to one corresponding, return the element:
      cytnx_error_msg(locator.size() != labels.size(),
                      "[ERROR][at] length of list should be the same for label and locator.%s",
                      "\n");
      cytnx_error_msg(
        labels.size() != this->rank(),
        "[ERROR][at] length of lists must be the same as UniTensor.rank (# of legs)%s", "\n");
      std::vector<cytnx_uint64> new_locator(this->rank());
      cytnx_uint64 new_loc;
      for (int i = 0; i < labels.size(); i++) {
        auto res = std::find(this->_impl->_labels.begin(), this->_impl->_labels.end(), labels[i]);
        cytnx_error_msg(res == this->_impl->_labels.end(),
                        "[ERROR] label:%s does not exist in current UniTensor.\n",
                        labels[i].c_str());
        new_loc = std::distance(this->_impl->_labels.begin(), res);
        new_locator[new_loc] = locator[i];
      }
      return this->at<T>(new_locator);
    }
    template <class T>
    T &at(const std::vector<std::string> &labels, const std::vector<cytnx_uint64> &locator) {
      // giving label <-> locator one to one corresponding, return the element:
      cytnx_error_msg(locator.size() != labels.size(),
                      "[ERROR][at] length of list should be the same for label and locator.%s",
                      "\n");
      cytnx_error_msg(
        labels.size() != this->rank(),
        "[ERROR][at] length of lists must be the same as UniTensor.rank (# of legs)%s", "\n");
      std::vector<cytnx_uint64> new_locator(this->rank());
      cytnx_uint64 new_loc;
      for (int i = 0; i < labels.size(); i++) {
        auto res = std::find(this->_impl->_labels.begin(), this->_impl->_labels.end(), labels[i]);
        cytnx_error_msg(res == this->_impl->_labels.end(),
                        "[ERROR] label:%s does not exist in current UniTensor.\n",
                        labels[i].c_str());
        new_loc = std::distance(this->_impl->_labels.begin(), res);
        new_locator[new_loc] = locator[i];
      }
      return this->at<T>(new_locator);
    }

    /**
    @brief Get an element at specific location.
    @details see more information at user guide 6.3.5.
    */
    const Scalar::Sproxy at(const std::vector<cytnx_uint64> &locator) const {
      if (this->uten_type() == UTenType.Block) {
        return this->_impl->at_for_sparse(locator);
      } else if (this->uten_type() == UTenType.Sparse) {
        if (this->_impl->elem_exists(locator)) {
          return this->_impl->at_for_sparse(locator);
        } else {
          cytnx_error_msg(true, "[ERROR][SparseUniTensor] invalid location. break qnum block.%s",
                          "\n");
        }
      } else {
        return this->get_block_().at(locator);
      }
    }

    /**
    @brief Get an element at specific location.
    @details see more information at user guide 6.3.5.
    */
    Scalar::Sproxy at(const std::vector<cytnx_uint64> &locator) {
      if (this->uten_type() == UTenType.Block) {
        return this->_impl->at_for_sparse(locator);
      } else if (this->uten_type() == UTenType.Sparse) {
        if (this->_impl->elem_exists(locator)) {
          return this->_impl->at_for_sparse(locator);
        } else {
          cytnx_error_msg(true, "[ERROR][SparseUniTensor] invalid location. break qnum block.%s",
                          "\n");
        }
      } else {
        return this->get_block_().at(locator);
      }
    }

    Scalar::Sproxy at(const std::vector<std::string> &labels,
                      const std::vector<cytnx_uint64> &locator) {
      // giving label <-> locator one to one corresponding, return the element:
      cytnx_error_msg(locator.size() != labels.size(),
                      "[ERROR][at] length of list should be the same for label and locator.%s",
                      "\n");
      cytnx_error_msg(
        labels.size() != this->rank(),
        "[ERROR][at] length of lists must be the same as UniTensor.rank (# of legs)%s", "\n");
      std::vector<cytnx_uint64> new_locator(this->rank());
      cytnx_uint64 new_loc;
      for (int i = 0; i < labels.size(); i++) {
        auto res = std::find(this->_impl->_labels.begin(), this->_impl->_labels.end(), labels[i]);
        cytnx_error_msg(res == this->_impl->_labels.end(),
                        "[ERROR] label:%s does not exist in current UniTensor.\n",
                        labels[i].c_str());
        new_loc = std::distance(this->_impl->_labels.begin(), res);
        new_locator[new_loc] = locator[i];
      }
      return this->at(new_locator);
    }

    const Scalar::Sproxy at(const std::vector<std::string> &labels,
                            const std::vector<cytnx_uint64> &locator) const {
      // giving label <-> locator one to one corresponding, return the element:
      cytnx_error_msg(locator.size() != labels.size(),
                      "[ERROR][at] length of list should be the same for label and locator.%s",
                      "\n");
      cytnx_error_msg(
        labels.size() != this->rank(),
        "[ERROR][at] length of lists must be the same as UniTensor.rank (# of legs)%s", "\n");
      std::vector<cytnx_uint64> new_locator(this->rank());
      cytnx_uint64 new_loc;
      for (int i = 0; i < labels.size(); i++) {
        auto res = std::find(this->_impl->_labels.begin(), this->_impl->_labels.end(), labels[i]);
        cytnx_error_msg(res == this->_impl->_labels.end(),
                        "[ERROR] label:%s does not exist in current UniTensor.\n",
                        labels[i].c_str());
        new_loc = std::distance(this->_impl->_labels.begin(), res);
        new_locator[new_loc] = locator[i];
      }
      return this->at(new_locator);
    }

    // return a clone of block
    /**
    @brief Get the block of the UniTensor for a given index.
        @param[in] idx the index of the block we want to get
        @return Tensor
    */
    Tensor get_block(const cytnx_uint64 &idx = 0) const { return this->_impl->get_block(idx); };
    //================================
    // return a clone of block
    /**
    @brief Get the block of the UniTensor for the given quantun indices.
        @param[in] qidx input the quantum indices you want to get the corresponding block.
        @param[in] force If force is true, it will return the tensor anyway (Even the
            corresponding block is empty, it will return void type tensor if \p force is
                set as true. Otherwise, it will trow the exception.)
        @return Tensor
    */
    Tensor get_block(const std::vector<cytnx_int64> &qidx, const bool &force = false) const {
      return this->_impl->get_block(qidx, force);
    }

    Tensor get_block(const std::vector<std::string> &labels, const std::vector<cytnx_int64> &qidx,
                     const bool &force = false) const {
      cytnx_error_msg(
        labels.size() != qidx.size(),
        "[ERROR][get_block] length of lists must be the same for both lables and qnidices%s", "\n");
      cytnx_error_msg(labels.size() != this->rank(),
                      "[ERROR][get_block] length of lists must be the rank (# of legs)%s", "\n");

      std::vector<cytnx_int64> loc_id(this->rank());
      std::vector<cytnx_int64> new_qidx(this->rank());

      cytnx_uint64 new_loc;
      std::vector<cytnx_uint64> new_order(this->rank());
      for (int i = 0; i < labels.size(); i++) {
        auto res = std::find(this->_impl->_labels.begin(), this->_impl->_labels.end(), labels[i]);
        cytnx_error_msg(res == this->_impl->_labels.end(),
                        "[ERROR][get_block] label:%s does not exists in current Tensor.\n",
                        labels[i].c_str());
        new_loc = std::distance(this->_impl->_labels.begin(), res);
        new_qidx[new_loc] = qidx[i];
        new_order[i] = new_loc;
      }
      auto out = this->_impl->get_block(new_qidx, force);
      if (out.dtype() != Type.Void) out.permute_(new_order);
      return out;
    }

    /**
     * @see
     * get_block(const std::vector<cytnx_int64> &qnum, const bool &force)const
     */
    Tensor get_block(const std::initializer_list<cytnx_int64> &qnum,
                     const bool &force = false) const {
      std::vector<cytnx_int64> tmp = qnum;
      return get_block(tmp, force);
    }

    /**
     * @see
     * get_block(const std::vector<cytnx_int64> &qnum, const bool &force)const
     */
    Tensor get_block(const std::vector<cytnx_uint64> &qnum, const bool &force = false) const {
      std::vector<cytnx_int64> iqnum(qnum.begin(), qnum.end());
      return this->_impl->get_block(iqnum, force);
    }

    Tensor get_block(const std::vector<std::string> &labels, const std::vector<cytnx_uint64> &qidx,
                     const bool &force = false) const {
      std::vector<cytnx_int64> iqnum(qidx.begin(), qidx.end());
      return this->get_block(labels, iqnum, force);
    }

    /**
    @brief Get the shared view of block for the given index.
        @param[in] idx input the index you want to get the corresponding block
        @return const Tensor&
    */
    const Tensor &get_block_(const cytnx_uint64 &idx = 0) const {
      return this->_impl->get_block_(idx);
    }

    /**
    @see get_block_(const cytnx_uint64 &idx) const
        @note This function only works for non-symmetric UniTensor.
    */
    Tensor &get_block_(const cytnx_uint64 &idx = 0) { return this->_impl->get_block_(idx); }

    /**
    @brief Get the shared view of block for the given quantum indices.
        @param[in] qidx input the quantum indices you want to get the corresponding block.
        @param[in] force If force is true, it will return the tensor anyway (Even the
            corresponding block is empty, it will return void type tensor if \p force is
                set as true. Otherwise, it will trow the exception.)
        @return Tensor&
    */
    Tensor &get_block_(const std::vector<cytnx_int64> &qidx, const bool &force = false) {
      return this->_impl->get_block_(qidx, force);
    }

    /**
    @brief Get the shared (data) view of block for the given quantum indices on given labels
        @param[in] labels the labels of the bonds.
        @param[in] qidx input the quantum indices you want to get the corresponding block.
        @param[in] force If force is true, it will return the tensor anyway (Even the
            corresponding block is empty, it will return void type tensor if \p force is
                set as true. Otherwise, it will trow the exception.)
        @return Tensor&

        @note labels and qidx forming one to one pairs. e.g. it means get `qidx[i]` qnum at Bond
    `labels[i]`. Also note that the return Tensor will have axes in the same order specified by
    labels.

    */
    // developer note: Tensor is not the same object (Thus Tensor instead of Tensor& ),
    //                 since we permute! but they have shared data memory.
    Tensor get_block_(const std::vector<std::string> &labels, const std::vector<cytnx_int64> &qidx,
                      const bool &force = false) {
      cytnx_error_msg(
        labels.size() != qidx.size(),
        "[ERROR][get_block] length of lists must be the same for both lables and qnidices%s", "\n");
      cytnx_error_msg(labels.size() != this->rank(),
                      "[ERROR][get_block] length of lists must be the rank (# of legs)%s", "\n");

      std::vector<cytnx_int64> loc_id(this->rank());
      std::vector<cytnx_int64> new_qidx(this->rank());

      cytnx_uint64 new_loc;
      std::vector<cytnx_uint64> new_order(this->rank());
      for (int i = 0; i < labels.size(); i++) {
        auto res = std::find(this->_impl->_labels.begin(), this->_impl->_labels.end(), labels[i]);
        cytnx_error_msg(res == this->_impl->_labels.end(),
                        "[ERROR][get_block] label:%s does not exists in current Tensor.\n",
                        labels[i].c_str());
        new_loc = std::distance(this->_impl->_labels.begin(), res);
        new_qidx[new_loc] = qidx[i];
        new_order[i] = new_loc;
      }
      auto out = this->_impl->get_block_(new_qidx, force);
      if (out.dtype() != Type.Void) {
        out = out.permute(new_order);
      }
      return out;
    }

    /**
    @see get_block_(const std::vector<cytnx_int64> &qidx, const bool &force)
    */
    Tensor &get_block_(const std::initializer_list<cytnx_int64> &qidx, const bool &force = false) {
      std::vector<cytnx_int64> tmp = qidx;
      return get_block_(tmp, force);
    }

    /**
    @see get_block_(const std::vector<cytnx_int64> &qidx, const bool &force)
    */
    Tensor &get_block_(const std::vector<cytnx_uint64> &qidx, const bool &force = false) {
      std::vector<cytnx_int64> iqidx(qidx.begin(), qidx.end());
      return get_block_(iqidx, force);
    }

    Tensor get_block_(const std::vector<std::string> &labels, const std::vector<cytnx_uint64> &qidx,
                      const bool &force = false) {
      std::vector<cytnx_int64> iqidx(qidx.begin(), qidx.end());
      return get_block_(labels, iqidx, force);
    }
    //================================

    // this only work for non-symm tensor. return a shared view of block
    /**
    @see get_block_(const std::vector<cytnx_int64> &qidx, const bool &force)
    */
    const Tensor &get_block_(const std::vector<cytnx_int64> &qidx,
                             const bool &force = false) const {
      return this->_impl->get_block_(qidx, force);
    }

    /**
    @see get_block_(const std::vector<cytnx_int64> &qidx, const bool &force)
    */
    const Tensor &get_block_(const std::initializer_list<cytnx_int64> &qidx,
                             const bool &force = false) const {
      std::vector<cytnx_int64> tmp = qidx;
      return this->_impl->get_block_(tmp, force);
    }

    /**
    @see get_block_(const std::vector<cytnx_int64> &qidx, const bool &force)
    */
    const Tensor &get_block_(const std::vector<cytnx_uint64> &qidx,
                             const bool &force = false) const {
      std::vector<cytnx_int64> iqidx(qidx.begin(), qidx.end());
      return get_block_(iqidx, force);
    }

    //================================
    /**
    @brief Get all the blocks of the UniTensor.
        @details get_blocks will return the blocks of the UniTensor. Furthermore, \n
        1. For symmetric UniTensor, it will call @ref contiguous() and then return the
    deep copy of blocks.
    2. For non-symmetric UniTensor, it will return the deep copy of blocks.
        @return std::vector<Tensor>
    */
    //[dev]
    std::vector<Tensor> get_blocks() const { return this->_impl->get_blocks(); }

    /**
    @brief Get all the blocks of the UniTensor, inplacely.
        @see get_blocks()
        @param[in] silent whether need to print out the warning messages.
    */
    //[dev]
    const std::vector<Tensor> &get_blocks_(const bool &silent = false) const {
      return this->_impl->get_blocks_(silent);
    }

    /**
        @see get_blocks()_
    */
    //[dev]
    std::vector<Tensor> &get_blocks_(const bool &silent = false) {
      return this->_impl->get_blocks_(silent);
    }

    /**
    @brief Put the block into the UniTensor with given index.
        @param[in] in the block you want to put into UniTensor
        @param[in] in the index of the UniTensor you want to put the block \p in in.
    */
    void put_block(const Tensor &in, const cytnx_uint64 &idx = 0) {
      this->_impl->put_block(in, idx);
    }

    /**
    @brief Put the block into the UniTensor with given quantum number.
        @param[in] in_tens the block you want to put into UniTensor
        @param[in] qidx the quantum indices of the UniTensor you want to put the block \p in_tens
  in.
  @warning @p force will be deprecated soon!
    */
    void put_block(const Tensor &in_tens, const std::vector<cytnx_int64> &qidx, const bool &force) {
      this->_impl->put_block(in_tens, qidx, force);
    }

    /**
     * @brief Put the block into the UniTensor with given quantum indices, will copy the input
     * tensor.
     */
    void put_block(Tensor &in, const std::vector<std::string> &lbls,
                   const std::vector<cytnx_int64> &qidx, const bool &force = false) {
      cytnx_error_msg(
        lbls.size() != qidx.size(),
        "[ERROR][put_block] length of lists must be the same for both lables and qnidices%s", "\n");
      cytnx_error_msg(lbls.size() != this->rank(),
                      "[ERROR][put_block] length of lists must be the rank (# of legs)%s", "\n");

      std::vector<cytnx_int64> loc_id(this->rank());
      std::vector<cytnx_int64> new_qidx(this->rank());

      cytnx_uint64 new_loc;
      // std::vector<cytnx_uint64> new_order(this->rank());
      std::vector<cytnx_uint64> inv_order(this->rank());
      for (int i = 0; i < lbls.size(); i++) {
        auto res = std::find(this->_impl->_labels.begin(), this->_impl->_labels.end(), lbls[i]);
        cytnx_error_msg(res == this->_impl->_labels.end(),
                        "[ERROR][put_block] label:%s does not exists in current Tensor.\n",
                        lbls[i].c_str());
        new_loc = std::distance(this->_impl->_labels.begin(), res);
        new_qidx[new_loc] = qidx[i];
        // new_order[i] = new_loc;
        inv_order[new_loc] = i;
      }
      this->_impl->put_block(in.permute(inv_order), new_qidx, force);
    }

    /**
    @brief Put the block into the UniTensor with given index, inplacely.
        @note the put block will have shared view with the internal block, i.e. non-clone.
        @see put_block(const Tensor &in, const cytnx_uint64 &idx)
        */
    void put_block_(Tensor &in, const cytnx_uint64 &idx = 0) { this->_impl->put_block_(in, idx); }

    /**
    @brief Put the block into the UniTensor with given quantum indices, inplacely.
        @note the put block will have shared view with the internal block, i.e. non-clone.
        @see put_block(const Tensor &in, const cytnx_uint64 &idx)
  @warning @p force will be deprecated soon!
        */
    void put_block_(Tensor &in, const std::vector<cytnx_int64> &qidx, const bool &force) {
      this->_impl->put_block_(in, qidx, force);
    }

    /**
     * @brief Put the block into the UniTensor with given quantum indices, inplacely.
     * @note the put block will have shared view with the internal block, i.e. non-clone.
     */
    void put_block_(Tensor &in, const std::vector<std::string> &lbls,
                    const std::vector<cytnx_int64> &qidx, const bool &force = false) {
      cytnx_error_msg(
        lbls.size() != qidx.size(),
        "[ERROR][put_block_] length of lists must be the same for both lables and qnidices%s",
        "\n");
      cytnx_error_msg(lbls.size() != this->rank(),
                      "[ERROR][put_block_] length of lists must be the rank (# of legs)%s", "\n");

      std::vector<cytnx_int64> loc_id(this->rank());
      std::vector<cytnx_int64> new_qidx(this->rank());

      cytnx_uint64 new_loc;
      std::vector<cytnx_uint64> new_order(this->rank());
      std::vector<cytnx_uint64> inv_order(this->rank());
      for (int i = 0; i < lbls.size(); i++) {
        auto res = std::find(this->_impl->_labels.begin(), this->_impl->_labels.end(), lbls[i]);
        cytnx_error_msg(res == this->_impl->_labels.end(),
                        "[ERROR][put_block_] label:%s does not exists in current Tensor.\n",
                        lbls[i].c_str());
        new_loc = std::distance(this->_impl->_labels.begin(), res);
        new_qidx[new_loc] = qidx[i];
        new_order[i] = new_loc;
        inv_order[new_loc] = i;
      }
      in.permute_(inv_order);
      this->_impl->put_block_(in, new_qidx, force);
      in.permute_(new_order);
    }
    UniTensor get(const std::vector<Accessor> &accessors) const {
      UniTensor out;
      out._impl = this->_impl->get(accessors);
      return out;
    }
    void set(const std::vector<Accessor> &accessors, const Tensor &rhs) {
      this->_impl->set(accessors, rhs);
    }
    void set(const std::vector<Accessor> &accessors, const UniTensor &rhs) {
      cytnx_error_msg(
        rhs.uten_type() != UTenType.Dense,
        "[ERROR] cannot set elements from UniTensor with symmetry. Use at() instead.%s", "\n");
      cytnx_error_msg(this->is_diag(), "[ERROR] cannot set UniTensor with is_diag=True.%s", "\n");
      cytnx_error_msg(rhs.is_diag(),
                      "[ERROR] cannot set UniTensor. incoming UniTensor is_diag=True.%s", "\n");

      this->_impl->set(accessors, rhs.get_block());
    }
    /**
    @brief Reshape the UniTensor.
          @param[in] new_shape the new shape you want to reshape to.
          @param[in] rowrank the rowrank of the UniTensor after you reshape it.
    @note This function only works for non-symmetric UniTensor. That is, the UniTensor
        cannot be UTenType::Block.
          */
    UniTensor reshape(const std::vector<cytnx_int64> &new_shape, const cytnx_uint64 &rowrank = 0) {
      UniTensor out;
      out._impl = this->_impl->reshape(new_shape, rowrank);
      return out;
    }

    /**
    @brief Reshape the UniTensor, inplacely.
    @note This function only works for non-symmetric UniTensor. That is, the UniTensor
        cannot be UTenType::Block.
          @see reshape(const std::vector<cytnx_int64> &new_shape, const cytnx_uint64 &rowrank)
        */
    void reshape_(const std::vector<cytnx_int64> &new_shape, const cytnx_uint64 &rowrank = 0) {
      this->_impl->reshape_(new_shape, rowrank);
    }

    /**
    @brief Convert the UniTensor to non-diagonal form.
        @details to_dense() convert the UniTensor from diagonal form to non-diagonal structure.
            That means input the UniTensor with \p is_diag = true to \p is_diag = false.
        @pre
            1. The UniTensor need to be Dense UniTensor, that means this function is only
                    support for UTenType.Dense.
            2. The UniTensor need to be diagonal form (that means is_diag is true.)
        @return UniTensor
        @see to_dense_(), is_diag()
        */
    UniTensor to_dense() {
      UniTensor out;
      out._impl = this->_impl->to_dense();
      return out;
    }

    /**
    @brief Convert the UniTensor to non-diagonal form, inplacely.
        @see to_dense(), is_diag()
        */
    void to_dense_() { this->_impl->to_dense_(); }

    /**
     * @deprecated This function is deprecated. Please use \n
     *   combineBond(const std::vector<std::string> &indicators, const bool &force) \n
     *   instead.
     */
    void combineBonds(const std::vector<cytnx_int64> &indicators, const bool &force,
                      const bool &by_label) {
      this->_impl->combineBonds(indicators, force, by_label);
    }

    /**
    @deprecated
    @brief Combine the sevral bonds of the UniTensor.
        @param[in] indicators the labels of the lags you want to combine.
        @param[in] force If force is true, it will combine the bonds anyway even the direction
      of the bonds are same. After combining, the direction of the bonds will be set as
      same as the first bond.
        @pre
            1. The size of \p indicators need to >= 2.
            2. The UniTensor cannot be diagonal form (that means is_diag cannot be true.)
        */
    void combineBonds(const std::vector<std::string> &indicators, const bool &force = false) {
      this->_impl->combineBonds(indicators, force);
    }

    /**
     * @deprecated This function is deprecated. Please use \n
     *   combineBond(const std::vector<std::string> &indicators, const bool &force) \n
     *   instead.
     */
    void combineBonds(const std::vector<cytnx_int64> &indicators, const bool &force = false) {
      this->_impl->combineBonds(indicators, force);
    }

    /**
    @brief Combine the sevral bonds of the UniTensor.
        @param[in] indicators the labels of the lags you want to combine.
        @param[in] force If force is true, it will combine the bonds anyway even the direction
      of the bonds are same. After combining, the direction of the bonds will be set as
      same as the first bond.
        @pre
            1. The size of \p indicators need to >= 2.
            2. The UniTensor cannot be diagonal form (that means is_diag cannot be true.)
        */
    void combineBond(const std::vector<std::string> &indicators, const bool &force = false) {
      this->_impl->combineBond(indicators, force);
    }

    /**
    @brief Contract the UniTensor with common labels.
          @details This function contract the UniTensor lags with common labels.
          @param[in] inR The UniTensor you want to contract with.
          @param[in] mv_elem_self Whether the self UniTensor need to do contiguous()
        after the contraction.
          @param[in] mv_elem_rhs Whether the rhs UniTensor \p inR need to do contiguous()
        after the contraction.
          @pre
      1. Two UniTensor need to have same UniTensor type, namely, same UTenType.
        You cannot contract symmetric to non-symmetric UniTensor.
      2. You cannot contract tagged UniTensor and untagged UniTensor.
      3. For Dense diagonal UniTensor, the type of Bond (bra-ket) should match.
      4. For symmetric UniTensor (UTenType.Block), Symmetry, degeneracy,
        quantum numbers and Bond type should be consistent.
        @return UniTensor
        @see uten_type(), linalg::Tensordot
        */
    UniTensor contract(const UniTensor &inR, const bool &mv_elem_self = false,
                       const bool &mv_elem_rhs = false) const {
      UniTensor out;
      out._impl = this->_impl->contract(inR._impl, mv_elem_self, mv_elem_rhs);
      return out;
    }

    /// @cond
    /**
    @brief Get the total quantum number of the UniTensor.
        @param[in] physical
        @pre
        The UniTensor need to be symmetric type, that is UTenType.Block.
        @return std::vector<Bond>
        @note This API just have not support.
        */
    std::vector<Bond> getTotalQnums(const bool physical = false) const {
      return this->_impl->getTotalQnums(physical);
    }

    /**
        @note This API just have not support.
        */
    std::vector<std::vector<cytnx_int64>> get_blocks_qnums() const {
      return this->_impl->get_blocks_qnums();
    }
    /// @endcond

    /**
     * @brief Check whether the Blocks address are the same.
     * @return bool
     */
    bool same_data(const UniTensor &rhs) const {
      // check same type:
      if (this->_impl->uten_type() != rhs._impl->uten_type()) return false;

      return this->_impl->same_data(rhs._impl);
    }

    /**
    @brief The addition function of the UniTensor.
        @details This is addition function of the UniTensor. Given the UniTensor
            \f$ UT_2\f$ as the argument, it will return
                \f[
                  UT_{self} = UT_{self} + UT_2
                \f]
                Perform element-wise addition of two UniTensor.
        @param[in] rhs The UniTensor you want to add by.
        @return UniTensor&
        @pre
        The two UniTensor need to have same structure.
        @note Compare to Add(const UniTensor&)const, this is an inplace function.
        @see Add_(const Scalar&), Add(const UniTensor&)const, Add(const Scalar&)const ,
        operator+=(const UniTensor&), operator+=(const Scalar&), \ref operator+
        */
    UniTensor &Add_(const UniTensor &rhs) {
      this->_impl->Add_(rhs._impl);
      return *this;
    }

    /**
    @brief The multiplcation function of the UniTensor.
        @details This is multiplcation function of the UniTensor. Given the UniTensor
            \f$ UT_2\f$ as the argument, it will return
                \f[
                  UT_{self} = UT_{self} \times UT_2
                \f]
                Perform element-wise multiplication of two UniTensor.
        @param[in] rhs The UniTensor you want to multiplcate by.
        @return UniTensor&
        @pre
        The two UniTensor need to have same structure.
        @note Compare to Mul(const UniTensor&)const, this is an inplace function.
        @see Mul_(const Scalar&), Mul(const UniTensor&)const, Mul(const Scalar&)const ,
        operator*=(const UniTensor&), operator*=(const Scalar&), \ref operator*
        */
    UniTensor &Mul_(const UniTensor &rhs) {
      this->_impl->Mul_(rhs._impl);
      return *this;
    }

    /**
    @brief The subtraction function of the UniTensor.
        @details This is subtraction function of the UniTensor. Given the UniTensor
            \f$ UT_2\f$ as the argument, it will return
                \f[
                  UT_{self} = UT_{self} - UT_2
                \f]
                Perform element-wise subtraction of two UniTensor.
        @param[in] rhs the subtrahend
        @return UniTensor&
        @pre
        The two UniTensor need to have same structure.
        @note Compare to Sub(const UniTensor&)const, this is an inplace function.
        @see Sub_(const Scalar&), Sub(const UniTensor&)const, Sub(const Scalar&)const ,
        operator-=(const UniTensor&), operator-=(const Scalar&), \ref operator-
        */
    UniTensor &Sub_(const UniTensor &rhs) {
      this->_impl->Sub_(rhs._impl);
      return *this;
    }

    /**
    @brief The division function of the UniTensor.
        @details This is division function of the UniTensor. Given the UniTensor
            \f$ UT_2\f$ as the argument, it will return
                \f[
                  UT_{self} = UT_{self} / UT_2
                \f]
                Perform element-wise division of two UniTensor.
        @param[in] rhs the divisor
        @return UniTensor&
        @pre
        The two UniTensor need to have same structure.
        @note Compare to Div(const UniTensor&)const, this is an inplace function.
        @see Div_(const Scalar&), Div(const UniTensor&)const, Div(const Scalar&)const ,
        operator/=(const UniTensor&), operator/=(const Scalar&), \ref operator/
        */
    UniTensor &Div_(const UniTensor &rhs) {
      this->_impl->Div_(rhs._impl);
      return *this;
    }

    /**
    @brief The addition function for a given scalar.
        @details Given the Scalar \p rhs, it will perform the addition for each element
            in UniTensor with this Scalar \p rhs.
        @param[in] rhs a Scalar you want to add in the UniTensor.
        @return UniTensor&
        @note Compare to Add(const Scalar&)const, this is an inplace function.
        @see Add_(const UniTensor&), Add(const UniTensor&)const, Add(const Scalar&)const ,
        operator+=(const UniTensor&), operator+=(const Scalar&), \ref operator+
        */
    UniTensor &Add_(const Scalar &rhs) {
      this->_impl->Add_(rhs);
      return *this;
    }

    /**
    @brief The multiplication function for a given scalar.
        @details Given the scalar \p rhs, it will perform the multiplication for each element
            in UniTensor with this scalar \p rhs.
        @param[in] rhs a scalar you want to multiplicate in the UniTensor.
        @return UniTensor&
        @note Compare to Mul(const Scalar&)const, this is an inplace function.
        @see Mul_(const UniTensor&), Mul(const UniTensor&)const, Mul(const Scalar&)const ,
        operator*=(const UniTensor&), operator*=(const Scalar&), \ref operator*
        */
    UniTensor &Mul_(const Scalar &rhs) {
      this->_impl->Mul_(rhs);
      return *this;
    }

    /**
    @brief The subtraction function for a given scalar.
        @details Given the scalar \p rhs, it will perform the subtraction for each element
            in UniTensor with this scalar \p rhs.
        @param[in] rhs a scalar you want to subtract in the UniTensor.
        @return UniTensor&
        @note Compare to Sub(const Scalar&)const, this is an inplace function.
        @see Sub_(const UniTensor&), Sub(const UniTensor&)const, Sub(const Scalar&)const ,
        operator-=(const UniTensor&), operator-=(const Scalar&), \ref operator-
        */
    UniTensor &Sub_(const Scalar &rhs) {
      this->_impl->Sub_(rhs);
      return *this;
    }

    /**
    @brief The division function for a given scalar.
        @details Given the scalar \p rhs, it will perform the division for each element
            in UniTensor with this scalar \p rhs.
        @param[in] rhs a scalar you want to divide in the UniTensor.
        @return UniTensor&
        @note Compare to Sub(const Scalar&)const, this is an inplace function.
        @see Div_(const UniTensor&), Div(const UniTensor&)const, Div(const Scalar&)const ,
        operator/=(const UniTensor&), operator/=(const Scalar&), \ref operator/
        */
    UniTensor &Div_(const Scalar &rhs) {
      this->_impl->Div_(rhs);
      return *this;
    }

    /**
    @brief The addition function of the UniTensor.
        @details This is addition function of the UniTensor. Given the UniTensor
            \f$ UT_2\f$ as the argument, it will return a new UniTensor
                \f[
                  UT = UT_{self} + UT_2
                \f]
                Perform element-wise addition of two UniTensor.
        @param[in] rhs The UniTensor you want to add by.
        @return UniTensor
        @pre
        The two UniTensor need to have same structure.
        @note Compare to Add_(const UniTensor&), this function will create a new UniTensor.
        @see Add_(const UniTensor&), Add_(const Scalar&), Add(const Scalar&)const ,
        operator+=(const UniTensor&), operator+=(const Scalar&), \ref operator+
        */
    UniTensor Add(const UniTensor &rhs) const;

    /**
    @brief The addition function for a given scalar.
        @details Given the scalar \p rhs, it will perform the addition for each element
            in UniTensor with this scalar \p rhs.
        @param[in] rhs a scalar you want to add in the UniTensor.
        @return UniTensor
        @note Compare to Add_(const Scalar&), this function will create a new UniTensor.
        @see Add_(const Scalar&), Add_(const UniTensor&), Add(const UniTensor&)const,
        operator+=(const UniTensor&), operator+=(const Scalar&), \ref operator+
        */
    UniTensor Add(const Scalar &rhs) const;

    /**
    @brief The multiplication function of the UniTensor.
        @details This is multiplication function of the UniTensor. Given the UniTensor
            \f$ UT_2\f$ as the argument, it will return a new UniTensor
                \f[
                  UT = UT_{self} \times UT_2
                \f]
                Perform element-wise multiplcation of two UniTensor.
        @param[in] rhs The UniTensor you want to multiplicate by.
        @return UniTensor
        @pre
        The two UniTensor need to have same structure.
        @note Compare to Mul_(const UniTensor&), this function will create a new UniTensor.
        @see Mul_(const UniTensor&), Mul_(const Scalar&), Mul(const Scalar&)const ,
        operator*=(const UniTensor&), operator*=(const Scalar&), \ref operator*
        */
    UniTensor Mul(const UniTensor &rhs) const;

    /**
    @brief The multiplication function for a given scalar.
        @details Given the scalar \p rhs, it will perform the multiplication for each element
            in UniTensor with this scalar \p rhs.
        @param[in] rhs a scalar you want to multiply in the UniTensor.
        @return UniTensor
        @note Compare to Mul_(const Scalar&), this function will create a new UniTensor.
        @see Mul_(const Scalar&), Mul_(const UniTensor&), Mul(const UniTensor&)const,
        operator*=(const UniTensor&), operator*=(const Scalar&), \ref operator*
        */
    UniTensor Mul(const Scalar &rhs) const;

    /**
    @brief The division function of the UniTensor.
        @details This is division function of the UniTensor. Given the UniTensor
            \f$ UT_2\f$ as the argument, it will return a new UniTensor
                \f[
                  UT = UT_{self} / UT_2
                \f]
                Perform element-wise division of two UniTensor.
        @param[in] rhs the divisor
        @return UniTensor
        @pre
        The two UniTensor need to have same structure.
        @note Compare to Div_(const UniTensor&), this function will create a new UniTensor.
        @see Div_(const UniTensor&), Div_(const Scalar&), Div(const Scalar&)const ,
        operator/=(const UniTensor&), operator/=(const Scalar&), \ref operator/
        */
    UniTensor Div(const UniTensor &rhs) const;

    /**
    @brief The division function for a given scalar.
        @details Given the scalar \p rhs, it will perform the division for each element
            in UniTensor with this scalar \p rhs.
        @param[in] rhs a scalar you want to divide in the UniTensor.
        @return UniTensor
        @note Compare to Div_(const Scalar&), this function will create a new UniTensor.
        @see Div_(const Scalar&), Div_(const UniTensor&), Div(const UniTensor&)const,
        operator/=(const UniTensor&), operator/=(const Scalar&), \ref operator/
        */
    UniTensor Div(const Scalar &rhs) const;

    /**
    @brief The subtraction function of the UniTensor.
        @details This is subtraction function of the UniTensor. Given the UniTensor
            \f$ UT_2\f$ as the argument, it will return a new UniTensor
                \f[
                  UT = UT_{self} - UT_2
                \f]
                Perform element-wise subtraction of two UniTensor.
        @param[in] rhs the subtrahend
        @return UniTensor
        @pre
        The two UniTensor need to have same structure.
        @note Compare to Sub_(const UniTensor&), this function will create a new UniTensor.
        @see Sub_(const UniTensor&), Sub_(const Scalar&), Sub(const Scalar&)const ,
        operator-=(const UniTensor&), operator-=(const Scalar&), \ref operator-
        */
    UniTensor Sub(const UniTensor &rhs) const;

    /**
    @brief The subtraction function for a given scalar.
        @details Given the scalar \p rhs, it will perform the subtraction for each element
            in UniTensor with this scalar \p rhs.
        @param[in] rhs the subtrahend
        @return UniTensor
        @note Compare to Sub_(const Scalar&), this function will create a new UniTensor.
        @see Sub_(const Scalar&), Sub_(const UniTensor&), Sub(const UniTensor&)const,
        operator-=(const UniTensor&), operator-=(const Scalar&), \ref operator-
        */
    UniTensor Sub(const Scalar &rhs) const;

    /**
    @brief Return the norm of the UniTensor.
        @details Norm() return the 2-norm of the UniTensor \f$UT\f$. Namely, it return
        \f[
        ||UT||_2
        \f]
        @return Tensor
        */
    Tensor Norm() const { return this->_impl->Norm(); };

    /**
    @brief The addition assignment operator of the UniTensor.
        @details This is addition assignment operator of the UniTensor. It will perform
            element-wise addition and return
                \f[
                UT += UT_R
                \f]
        @param[in] rhs The UniTensor you want to add by.
        @return UniTensor&
        @pre
        The two UniTensor need to have same structure.
        @see
        operator+=(const Scalar&), \ref operator+, Add_(const UniTensor&),
        Add_(const Scalar&), Add(const UniTensor&),	Add(const Scalar&)const
        */
    UniTensor &operator+=(const UniTensor &rhs) {
      this->Add_(rhs);
      return *this;
    }

    /**
    @brief The subtraction assignment operator of the UniTensor.
        @details This is subtraction assignment operator of the UniTensor. It will perform
            element-wise subtraction and return
                \f[
                UT -= UT_R
                \f]
        @param[in] rhs the subtrahend
        @return UniTensor&
        @pre
        The two UniTensor need to have same structure.
        @see
        operator-=(const Scalar&), \ref operator-, Sub_(const UniTensor&),
        Sub_(const Scalar&), Sub(const UniTensor&),	Sub(const Scalar&)const
        */
    UniTensor &operator-=(const UniTensor &rhs) {
      this->Sub_(rhs);
      return *this;
    }

    /**
    @brief The division assignment operator of the UniTensor.
        @details This is division assignment operator of the UniTensor. It will perform
            element-wise division and return
                \f[
                UT /= UT_R
                \f]
        @param[in] rhs the divisor
        @return UniTensor&
        @pre
        The two UniTensor need to have same structure.
        @see
        operator/=(const Scalar&), \ref operator/, Div_(const UniTensor&),
        Div_(const Scalar&), Div(const UniTensor&),	Div(const Scalar&)const
        */
    UniTensor &operator/=(const UniTensor &rhs) {
      this->Div_(rhs);
      return *this;
    }

    /**
    @brief The multiplication assignment operator of the UniTensor.
        @details This is multiplication assignment operator of the UniTensor. It will perform
            element-wise multiplication and return
                \f[
                UT *= UT_R
                \f]
        @param[in] rhs The UniTensor you want to multilicate by.
        @return UniTensor&
        @pre
        The two UniTensor need to have same structure.
        @see
        operator*=(const Scalar&), \ref operator*, Mul_(const UniTensor&),
        Mul_(const Scalar&), Mul(const UniTensor&),	Mul(const Scalar&)const
        */
    UniTensor &operator*=(const UniTensor &rhs) {
      this->Mul_(rhs);
      return *this;
    }

    /**
    @brief The addition assignment operator for a given scalar.
        @details Given the scalar \p rhs, it will perform the addition for each element
            in UniTensor with this scalar \p rhs.
        @param[in] rhs a scalar you want to add in the UniTensor.
        @return UniTensor&
        @see
        operator+=(const UniTensor&), \ref operator+, Add_(const UniTensor&),
        Add_(const Scalar&), Add(const UniTensor&),	Add(const Scalar&)const
        */
    UniTensor &operator+=(const Scalar &rhs) {
      this->Add_(rhs);
      return *this;
    }

    /**
    @brief The subtraction assignment operator for a given scalar.
        @details Given the scalar \p rhs, it will perform the subtraction for each element
            in UniTensor with this scalar \p rhs.
        @param[in] rhs the subtrahend
        @return UniTensor&
        @see
        operator-=(const UniTensor&), \ref operator-, Sub_(const UniTensor&),
        Sub_(const Scalar&), Sub(const UniTensor&),	Sub(const Scalar&)const
        */
    UniTensor &operator-=(const Scalar &rhs) {
      this->Sub_(rhs);
      return *this;
    }

    /**
    @brief The division assignment operator for a given scalar.
        @details Given the scalar \p rhs, it will perform the division for each element
            in UniTensor with this scalar \p rhs.
        @param[in] rhs the divisor
        @return UniTensor&
        @see
        operator/=(const UniTensor&), \ref operator/, Div_(const UniTensor&),
        Div_(const Scalar&), Div(const UniTensor&),	Div(const Scalar&)const
        */
    UniTensor &operator/=(const Scalar &rhs) {
      this->Div_(rhs);
      return *this;
    }

    /**
    @brief The multiplication assignment operator for a given scalar.
        @details Given the scalar \p rhs, it will perform the multiplication for each element
            in UniTensor with this scalar \p rhs.
        @param[in] rhs a scalar you want to multiply in the UniTensor.
        @return UniTensor&
        @see
        operator*=(const Scalar&), \ref operator*, Mul_(const UniTensor&),
        Mul_(const Scalar&), Mul(const UniTensor&),	Mul(const Scalar&)const
        */
    UniTensor &operator*=(const Scalar &rhs) {
      this->Mul_(rhs);
      return *this;
    }

    /**
    @brief Apply complex conjugate on each entry of the UniTensor.
        @details Conj() apply complex conjugate on each entry of the UniTensor.
        @return UniTensor
    @note Compare to Conj_(), this fucntion will create a new object UniTensor.
        @see Conj_()
        */
    UniTensor Conj() const {
      UniTensor out;
      out._impl = this->_impl->Conj();
      return out;
    }

    /**
    @brief Apply complex conjugate on each entry of the UniTensor.
        @details Conj_() apply complex conjugate on each entry of the UniTensor, inplacely.
        @return UniTensor
    @note Compare to Conj(), this fucntion is inplace function.
        @see Conj()
        */
    UniTensor &Conj_() {
      this->_impl->Conj_();
      return *this;
    }

    /**
    @brief Take the transpose of the UniTensor.
    @details This function will take the transpose of the UniTensor. If the UniTensor is
      tagged (i.e. the Bonds are directional), it will swap the direction of the Bonds but
      the rowrank will not change. If the UniTensor is untagged (i.e. the Bonds are
      BondType::BD_REG), it will change the rowrank to the opposite side.
        @return UniTensor
    @note Compare to Transpose_(), this fucntion will return new UniTensor object.
        @see Transpose_()
        */
    UniTensor Transpose() const {
      UniTensor out;
      out._impl = this->_impl->Transpose();
      return out;
    }

    /**
    @brief Take the transpose of the UniTensor, inplacely.
        @return UniTensor
    @note Compare to Transpose(), this fucntion is inplace function.
        @see Transpose()
        */
    UniTensor &Transpose_() {
      this->_impl->Transpose_();
      return *this;
    }

    /**
    @brief normalize the current UniTensor instance with 2-norm.
        @return UniTensor
    @note Compare to normalize_(), this fucntion will return new UniTensor object.
        @see normalize_()
        */
    UniTensor normalize() const {
      UniTensor out;
      out._impl = this->_impl->normalize();
      return out;
    }

    /**
    @brief normalize the UniTensor, inplacely.
        @return UniTensor
    @note Compare to normalize(), this fucntion is inplace function.
        @see normalize()
        */
    UniTensor &normalize_() {
      this->_impl->normalize_();
      return *this;
    }

    /**
    @brief Take the partial trance to the UniTensor.
        @details Take the partial trace to the UniTensor with the give two labels.
        @param[in] a label 1
        @param[in] b label 2
        @return UniTensor
    @note Compare to Trace_(), this fucntion will return a new UniTensor object.
        @see Trace_()
        */
    UniTensor Trace(const std::string &a, const std::string &b) const {
      UniTensor out;
      out._impl = this->_impl->Trace(a, b);
      return out;
    }

    /**
    @brief Take the partial trance to the UniTensor.
        @details Take the partial trace to the UniTensor with the give two labels.
        @param[in] a label 1
        @param[in] b label 2
        @return UniTensor
    @note Compare to Trace_(), this fucntion will return a new UniTensor object.
        @see Trace_()
        */
    UniTensor Trace(const cytnx_int64 &a = 0, const cytnx_int64 &b = 1) const {
      UniTensor out;
      out._impl = this->_impl->Trace(a, b);
      return out;
    }

    /**
    @brief Take the partial trance to the UniTensor, inplacely.
        @details Take the partial trace to the UniTensor with the give two labels.
        @param[in] a label 1
        @param[in] b label 2
        @return UniTensor&
    @note Compare to Trace(), this is an inplace function.
        @see Trace()
        */
    UniTensor &Trace_(const std::string &a, const std::string &b) {
      this->_impl->Trace_(a, b);
      if (this->uten_type() == UTenType.Block) {
        // handle if no leg left case for BlockUniTensor.
        if (this->rank() == 0) {
          DenseUniTensor *tmp = new DenseUniTensor();
          tmp->_block = this->get_blocks_(true)[0];
          this->_impl = boost::intrusive_ptr<UniTensor_base>(tmp);
        }
      }
      return *this;
    }

    /**
    @brief Take the partial trance to the UniTensor, inplacely.
        @details Take the partial trace to the UniTensor with the give two labels.
        @param[in] a label 1
        @param[in] b label 2
        @return UniTensor&
    @note Compare to Trace(), this is an inplace function.
        @see Trace()
        */
    UniTensor &Trace_(const cytnx_int64 &a = 0, const cytnx_int64 &b = 1) {
      this->_impl->Trace_(a, b);
      if (this->uten_type() == UTenType.Block) {
        // handle if no leg left case for BlockUniTensor.
        if (this->rank() == 0) {
          DenseUniTensor *tmp = new DenseUniTensor();
          tmp->_block = this->get_blocks_(true)[0];
          this->_impl = boost::intrusive_ptr<UniTensor_base>(tmp);
        }
      }
      return *this;
    }

    /**
    @brief Take the conjugate transpose to the UniTensor.
        @return UniTensor
    @note Compare to Dagger_(), this function will create a new UniTensor ojbect.
        @see Dagger_(), Transpose()
        */
    UniTensor Dagger() const {
      UniTensor out;
      out._impl = this->_impl->Dagger();
      return out;
    }

    /**
    @brief Take the conjugate transpose to the UniTensor, inplacely.
        @return UniTensor&
    @note Compare to Dagger(), this is an inplace function.
        @see Dagger()
        */
    UniTensor &Dagger_() {
      this->_impl->Dagger_();
      return *this;
    }

    /**
     * @brief Set the UniTensor as a tagged UniTensor.
     * @details This function will set the untagged UniTensor(all of the Bonds are
     * BondType::BD_REG) ) to tagged UniTensor (all of the Bonds are BondType::BD_BRA or
     * BondType::BD_KET). And it will change the UniTensor to bra-ket form. See is_braket_form()
     * for more information.
     * @see is_braket_form()
     */
    UniTensor &tag() {
      this->_impl->tag();
      return *this;
    }

    /**
    @brief Power function.
        @details Take power \p p on all the elements in the UniTensor.
        @param p power
        @return UniTensor
    @note Compare to Pow_(), this function will create a new UniTensor ojbect.
        @see Pow_()
        */
    UniTensor Pow(const double &p) const;

    /**
    @brief Power function.
        @details Take power \p p on all the elements in the UniTensor, inplacely.
        @param p power
        @return UniTensor&
    @note Compare to Pow(), this function is an inplacely function.
        @see Pow()
        */
    UniTensor &Pow_(const double &p);

    /**
     * @brief Geiven the locator, check if the element exists.
     * @param[in] locator the locator of the element.
     * @return [bool]
     * @note Same as at(\p locator).exists().
     */
    bool elem_exists(const std::vector<cytnx_uint64> &locator) const {
      return this->_impl->elem_exists(locator);
    }

    /**
     * @deprecated
     * This function is deprecated, please use at() instread.
     * @note C++: Deprecated soon, use at()
     */
    template <class T>
    T get_elem(const std::vector<cytnx_uint64> &locator) const {
      return this->at<T>(locator);
    }

    /**
     * @deprecated
     * This function is deprecated, please use at() instread.
     * @note C++: Deprecated soon, use at()
     */
    template <class T2>
    void set_elem(const std::vector<cytnx_uint64> &locator, const T2 &rc) {
      // cytnx_error_msg(true,"[ERROR] invalid type%s","\n");
      this->at(locator) = rc;
    }

    /**
    @brief save a UniTensor to file
    @details Save a UniTensor to file. The file extension will be extended as '.cytnx'
    @param[in] fname the file name (exclude the file extension).
    @see Load(const std::string &fname)
    */
    void Save(const std::string &fname) const;

    /**
    @brief save a UniTensor to file
    @details Save a UniTensor to file. The file extension will be extended as '.cytnx'
    @param[in] fname the file name (exclude the file extension).
    @see Load(const char *fname)
    */
    void Save(const char *fname) const;

    /**
    @brief load a UniTensor from file
    @param[in] fname the file name
    @return the loaded UniTensor
    @pre The file must be a UniTensor object. That is, the file must be created by
    UniTensor::Save().
    @see Save(const std::string &fname) const
    */
    static UniTensor Load(const std::string &fname);

    /**
    @brief load a UniTensor from file
    @param[in] fname: the file name
    @return the loaded UniTensor
    @pre The file must be a UniTensor object. That is, the file must be created by
    UniTensor::Save().
    @see Save(const char* fname) const
    */
    static UniTensor Load(const char *fname);

    /**
     * @brief truncate bond dimension of the UniTensor by the given bond label and dimension.
     * @param[in] label the bond label.
     * @param[in] dim the dimension to be truncated.
     * @return [UniTensor]
     * @note compare to truncate(const std::string &bond_idx, const cytnx_uint64 &dim),
     *   this is inplace function and will modify the current UniTensor object.
     */
    UniTensor &truncate_(const std::string &label, const cytnx_uint64 &dim) {
      this->_impl->truncate_(label, dim);
      return *this;
    }

    /**
     * @brief truncate bond dimension of the UniTensor by the given bond index and dimension.
     * @param[in] bond_idx the bond index.
     * @param[in] dim the dimension to be truncated.
     * @return [UniTensor]
     * @note compare to truncate(const cytnx_int64 &bond_idx, const cytnx_uint64 &dim),
     *  this is inplace function and will modify the current UniTensor object.
     */
    UniTensor &truncate_(const cytnx_int64 &bond_idx, const cytnx_uint64 &dim) {
      this->_impl->truncate_(bond_idx, dim);
      return *this;
    }

    /**
     * @brief truncate bond dimension of the UniTensor by the given bond label and dimension.
     * @param[in] label the bond label.
     * @param[in] dim the dimension to be truncated.
     * @return [UniTensor]
     * @note compare to truncate_(const std::string &bond_idx, const cytnx_uint64 &dim),
     *   this function will return a new UniTensor object.
     * @see UniTensor::truncate_(const std::string &bond_idx, const cytnx_uint64 &dim)
     */
    UniTensor truncate(const std::string &label, const cytnx_uint64 &dim) const {
      UniTensor out = this->clone();
      out.truncate_(label, dim);
      return out;
    }

    /**
     * @brief truncate bond dimension of the UniTensor by the given bond index and dimension.
     * @param[in] bond_idx the bond index.
     * @param[in] dim the dimension to be truncated.
     * @return [UniTensor]
     * @note compare to truncate_(const cytnx_int64 &bond_idx, const cytnx_uint64 &dim),
     *  this function will return a new UniTensor object.
     * @see UniTensor::truncate_(const cytnx_int64 &bond_idx, const cytnx_uint64 &dim)
     */
    UniTensor truncate(const cytnx_int64 &bond_idx, const cytnx_uint64 &dim) const {
      UniTensor out = this->clone();
      out.truncate_(bond_idx, dim);
      return out;
    }

    /**
    @brief get the q-indices on each leg for the [bidx]-th block
    @param bidx the bidx-th block in current block list.
    @return
        [vector]

    */
    const std::vector<cytnx_uint64> &get_qindices(const cytnx_uint64 &bidx) const {
      return this->_impl->get_qindices(bidx);
    }
    /**
    @brief get the q-indices on each leg for the [bidx]-th block
    @param bidx the bidx-th block in current block list.
    @return
        [vector]

    */
    std::vector<cytnx_uint64> &get_qindices(const cytnx_uint64 &bidx) {
      return this->_impl->get_qindices(bidx);
    }

    /**
    @brief get the q-indices on each leg for all the blocks
    @return
        [2d vector]

    */
    const vec2d<cytnx_uint64> &get_itoi() const { return this->_impl->get_itoi(); }
    vec2d<cytnx_uint64> &get_itoi() { return this->_impl->get_itoi(); }

    /// @cond
    void _Load(std::fstream &f);
    void _Save(std::fstream &f) const;
    /// @endcond

    UniTensor &convert_from(const UniTensor &rhs, const bool &force = false) {
      this->_impl->from_(rhs._impl, force);
      return *this;
    }

    // Generators:
    /**
    @brief Generate a one-bond UniTensor with all elements set to zero.
    @param[in] Nelem the number of elements.
    @param[in] in_labels the labels of the UniTensor.
    @param[in] dtype the data type of the UniTensor, see cytnx::Type for more information.
    @param[in] device the device type of the UniTensor, see cytnx::Device for more information.
    @param[in] name the name of the UniTensor.
    @return
        [UniTensor]

    @see zeros(const cytnx_uint64 &Nelem, const unsigned int &dtype, const int &device)
    */
    static UniTensor zeros(const cytnx_uint64 &Nelem,
                           const std::vector<std::string> &in_labels = {},
                           const unsigned int &dtype = Type.Double, const int &device = Device.cpu,
                           const std::string &name = "") {
      return UniTensor(cytnx::zeros(Nelem, dtype, device), false, -1, in_labels, name);
    }

    /**
    @brief Generate a UniTensor with all elements set to zero.
    @param[in] shape the shape of the UniTensor.
    @param[in] in_labels the labels of the UniTensor.
    @param[in] dtype the data type of the UniTensor, see cytnx::Type for more information.
    @param[in] device the device type of the UniTensor, see cytnx::Device for more information.
    @param[in] name the name of the UniTensor.
    @return
        [UniTensor]

    @see zeros(const std::vector< cytnx_uint64 > &Nelem, const unsigned int &dtype, const int
    &device)
    */
    static UniTensor zeros(const std::vector<cytnx_uint64> &shape,
                           const std::vector<std::string> &in_labels = {},
                           const unsigned int &dtype = Type.Double, const int &device = Device.cpu,
                           const std::string &name = "") {
      return UniTensor(cytnx::zeros(shape, dtype, device), false, -1, in_labels, name);
    }

    /**
    @brief Generate a one-bond UniTensor with all elements set to one.
    @param[in] Nelem the number of elements.
    @param[in] in_labels the labels of the UniTensor.
    @param[in] dtype the data type of the UniTensor, see cytnx::Type for more information.
    @param[in] device the device type of the UniTensor, see cytnx::Device for more information.
    @param[in] name the name of the UniTensor.
    @return
        [UniTensor]
    @see ones(const cytnx_uint64 &Nelem, const unsigned int &dtype, const int &device)
    */
    static UniTensor ones(const cytnx_uint64 &Nelem, const std::vector<std::string> &in_labels = {},
                          const unsigned int &dtype = Type.Double, const int &device = Device.cpu,
                          const std::string &name = "") {
      return UniTensor(cytnx::ones(Nelem, dtype, device), false, -1, in_labels, name);
    }

    /**
    @brief Generate a identity UniTensor.
    @param[in] dim the dimension of the diagnal.
    @param[in] in_labels the labels of the UniTensor.
    @param[in] is_diag determine if the UniTensor is diagonal or not. Default is false.
    @param[in] dtype the data type of the UniTensor, see cytnx::Type for more information.
    @param[in] device the device type of the UniTensor, see cytnx::Device for more information.
    @param[in] name the name of the UniTensor.
    @return
        [UniTensor]
    @note 2-bond if not diagonal. 1-bond if diagonal.
    */
    static UniTensor identity(const cytnx_uint64 &dim,
                              const std::vector<std::string> &in_labels = {},
                              const cytnx_bool &is_diag = false,
                              const unsigned int &dtype = Type.Double,
                              const int &device = Device.cpu, const std::string &name = "") {
      if (is_diag) {
        return UniTensor(cytnx::ones(dim, dtype, device), is_diag, -1, in_labels, name);
      } else {
        return UniTensor(cytnx::identity(dim, dtype, device), is_diag, -1, in_labels, name);
      }
    }

    /**
    @brief Generate a 2-bond identity UniTensor
    @param[in] dim the dimension of the diagnal.
    @param[in] in_labels the labels of the UniTensor.
    @param[in] is_diag determine if the UniTensor is diagonal or not. Default is false.
    @param[in] dtype the data type of the UniTensor, see cytnx::Type for more information.
    @param[in] device the device type of the UniTensor, see cytnx::Device for more information.
    @param[in] name the name of the UniTensor.
    @return
        [UniTensor]
    @note 2-bond if not diagonal. 1-bond if diagonal.
    @see identity(Nelem, in_labels, is_diag, dtype, device, name)
    Note:
      This function is a alias of cytnx::UniTensor::identity().
    */
    static UniTensor eye(const cytnx_uint64 &dim, const std::vector<std::string> &in_labels = {},
                         const cytnx_bool &is_diag = false, const unsigned int &dtype = Type.Double,
                         const int &device = Device.cpu, const std::string &name = "") {
      return identity(dim, in_labels, is_diag, dtype, device, name);
    }

    /**
    @brief Generate a UniTensor with all elements set to one.
    @param[in] shape the shape of the UniTensor.
    @param[in] in_labels the labels of the UniTensor.
    @param[in] dtype the data type of the UniTensor, see cytnx::Type for more information.
    @param[in] device the device type of the UniTensor, see cytnx::Device for more information.
    @param[in] name the name of the UniTensor.
    @return
        [UniTensor]
    @see ones(const std::vector< cytnx_uint64 > &shape, const unsigned int &dtype, const int
    &device)
    */
    static UniTensor ones(const std::vector<cytnx_uint64> &shape,
                          const std::vector<std::string> &in_labels = {},
                          const unsigned int &dtype = Type.Double, const int &device = Device.cpu,
                          const std::string &name = "") {
      return UniTensor(cytnx::ones(shape, dtype, device), false, -1, in_labels, name);
    }

    /**
    @brief Generate a one-bond UniTensor with all elements are arange from 0 to Nelem-1.
    @details Generate a UniTensor with all elements are arange from 0 to Nelem-1 with double data
    type on cpu device. The step is 1.
    @param[in] Nelem the number of elements.
    @param[in] in_labels the labels of the UniTensor.
    @param[in] name the name of the UniTensor.
    @return
        [UniTensor]
    @see arange(const cytnx_double &start, const cytnx_double &end, const cytnx_double &step, const
    unsigned int &dtype, const int &device) \n
    @see arange(const cytnx_int64 &Nelem)
    */
    static UniTensor arange(const cytnx_int64 &Nelem,
                            const std::vector<std::string> &in_labels = {},
                            const std::string &name = "") {
      return UniTensor(cytnx::arange(Nelem), false, -1, in_labels, name);
    }

    /**
    @brief Generate a UniTensor with all elements are arange from \p start to \p end.
    @details Generate a UniTensor with all elements are arange from \p start to \p end , the step is
    \p step .
    @param[in] start the start of the arange.
    @param[in] end the end of the arange.
    @param[in] step the step of the arange.
    @param[in] in_labels the labels of the UniTensor.
    @param[in] dtype the data type of the UniTensor, see cytnx::Type for more information.
    @param[in] device the device type of the UniTensor, see cytnx::Device for more information.
    @param[in] name the name of the UniTensor.
    @return
        [UniTensor]
    @see arange(const cytnx_double &start, const cytnx_double &end, const cytnx_double &step, const
    unsigned int &dtype, const int &device)
    */
    static UniTensor arange(const cytnx_double &start, const cytnx_double &end,
                            const cytnx_double &step = 1,
                            const std::vector<std::string> &in_labels = {},
                            const unsigned int &dtype = Type.Double, const int &device = Device.cpu,
                            const std::string &name = "") {
      return UniTensor(cytnx::arange(start, end, step, dtype, device), false, -1, in_labels, name);
    }

    /**
    @brief Generate a one-bond UniTensor with all elements are evenly spaced numbers over a
    specified interval.
    @details Generate a UniTensor with all elements are evenly spaced numbers over a specified
    interval. The interval is \p [start, end] and the number of elements is \p Nelem.
    @param[in] start the start of the linspace.
    @param[in] end the end of the linspace.
    @param[in] Nelem the number of elements.
    @param[in] endpoint if \p endpoint is true, the end of linspace is \p end, else the end of
    linspace is \p end - \p step.
    @param[in] in_labels the labels of the UniTensor.
    @param[in] dtype the data type of the UniTensor, see cytnx::Type for more information.
    @param[in] device the device type of the UniTensor, see cytnx::Device for more information.
    @param[in] name the name of the UniTensor.
    @return
        [UniTensor]
    @see linspace(const cytnx_double &start, const cytnx_double &end, const cytnx_uint64 &Nelem,
    const bool &endpoint, const unsigned int &dtype, const int &device)
    */
    static UniTensor linspace(const cytnx_double &start, const cytnx_double &end,
                              const cytnx_uint64 &Nelem, const bool &endpoint = true,
                              const std::vector<std::string> &in_labels = {},
                              const unsigned int &dtype = Type.Double,
                              const int &device = Device.cpu, const std::string &name = "") {
      return UniTensor(cytnx::linspace(start, end, Nelem, endpoint, dtype, device), false, -1,
                       in_labels, name);
    }

    // Random Generators:
    /**
     @brief Generate a one-bond UniTensor with all elements are random numbers sampled from a normal
     (Gaussian) distribution.
     @details Generate a UniTensor with all elements are random numbers sampled from a normal
     (Gaussian) distribution. The mean is \p mean and the standard deviation is \p std.
      @param[in] Nelem the number of elements.
      @param[in] mean the mean of the normal distribution.
      @param[in] std the standard deviation of the normal distribution.
      @param[in] in_labels the labels of the UniTensor.
      @param[in] seed the seed of the random generator.
      @param[in] dtype the data type of the UniTensor, see cytnx::Type for more information.
      @param[in] device the device type of the UniTensor, see cytnx::Device for more information.
      @param[in] name the name of the UniTensor.
      @return
          [UniTensor]
      @see random::normal(const cytnx_uint64 &Nelem, const double &mean, const double &std, const
     int &device, const unsigned int &seed, const unsigned int &dtype)
    */
    static UniTensor normal(const cytnx_uint64 &Nelem, const double &mean, const double &std,
                            const std::vector<std::string> &in_labels = {},
                            const unsigned int &seed = cytnx::random::__static_random_device(),
                            const unsigned int &dtype = Type.Double, const int &device = Device.cpu,
                            const std::string &name = "");

    /**
     @brief Generate a UniTensor with all elements are random numbers sampled from a normal
     (Gaussian) distribution.
     @details Generate a UniTensor with all elements are random numbers sampled from a normal
     (Gaussian) distribution. The mean is \p mean and the standard deviation is \p std.
      @param[in] shape the shape of the UniTensor.
      @param[in] mean the mean of the normal distribution.
      @param[in] std the standard deviation of the normal distribution.
      @param[in] in_labels the labels of the UniTensor.
      @param[in] seed the seed of the random generator.
      @param[in] dtype the data type of the UniTensor, see cytnx::Type for more information.
      @param[in] device the device type of the UniTensor, see cytnx::Device for more information.
      @param[in] name the name of the UniTensor.
      @return
          [UniTensor]
      @see random::normal(const std::vector< cytnx_uint64 > &shape, const double &mean, const double
     &std, const int &device, const unsigned int &seed, const unsigned int &dtype)
    */
    static UniTensor normal(const std::vector<cytnx_uint64> &shape, const double &mean,
                            const double &std, const std::vector<std::string> &in_labels = {},
                            const unsigned int &seed = cytnx::random::__static_random_device(),
                            const unsigned int &dtype = Type.Double, const int &device = Device.cpu,
                            const std::string &name = "");

    /**
    @brief Generate a one-bond UniTensor with all elements are random numbers sampled from a uniform
    distribution.
    @details Generate a UniTensor with all elements are random numbers sampled from a uniform
    distribution. The low bound is \p low and the high bound is \p high.
      @param[in] Nelem the number of elements.
      @param[in] low the low bound of the uniform distribution.
      @param[in] high the high bound of the uniform distribution.
      @param[in] in_labels the labels of the UniTensor.
      @param[in] seed the seed of the random generator.
      @param[in] dtype the data type of the UniTensor, see cytnx::Type for more information.
      @param[in] device the device type of the UniTensor, see cytnx::Device for more information.
      @param[in] name the name of the UniTensor.
      @return
          [UniTensor]
      @see random::uniform(const cytnx_uint64 &Nelem, const double &low, const double &high, const
    int &device, const unsigned int &seed, const unsigned int &dtype)
    */
    static UniTensor uniform(const cytnx_uint64 &Nelem, const double &low, const double &high,
                             const std::vector<std::string> &in_labels = {},
                             const unsigned int &seed = cytnx::random::__static_random_device(),
                             const unsigned int &dtype = Type.Double,
                             const int &device = Device.cpu, const std::string &name = "");

    /**
     @brief Generate a UniTensor with all elements are random numbers sampled from a uniform
     distribution.
     @details Generate a UniTensor with all elements are random numbers sampled from a uniform
     distribution. The low bound is \p low and the high bound is \p high.
      @param[in] shape the shape of the UniTensor.
      @param[in] low the low bound of the uniform distribution.
      @param[in] high the high bound of the uniform distribution.
      @param[in] in_labels the labels of the UniTensor.
      @param[in] seed the seed of the random generator.
      @param[in] dtype the data type of the UniTensor, see cytnx::Type for more information.
      @param[in] device the device type of the UniTensor, see cytnx::Device for more information.
      @param[in] name the name of the UniTensor.
      @return
          [UniTensor]
      @see random::uniform(const std::vector< cytnx_uint64 > &shape, const double &low, const double
     &high, const int &device, const unsigned int &seed, const unsigned int &dtype)
      */

    /**
     @brief Generate a UniTensor with all elements are random numbers sampled from a uniform
     distribution.
     @details Generate a UniTensor with all elements are random numbers sampled from a uniform
     distribution. The low bound is \p low and the high bound is \p high.
      @param[in] shape the shape of the UniTensor.
      @param[in] low the low bound of the uniform distribution.
      @param[in] high the high bound of the uniform distribution.
      @param[in] in_labels the labels of the UniTensor.
      @param[in] seed the seed of the random generator.
      @param[in] dtype the data type of the UniTensor, see cytnx::Type for more information.
      @param[in] device the device type of the UniTensor, see cytnx::Device for more information.
      @param[in] name the name of the UniTensor.
      @return
          [UniTensor]
      @see random::uniform(const std::vector< cytnx_uint64 > &shape, const double &low, const double
     &high, const int &device, const unsigned int &seed, const unsigned int &dtype)
    */
    static UniTensor uniform(const std::vector<cytnx_uint64> &shape, const double &low,
                             const double &high, const std::vector<std::string> &in_labels = {},
                             const unsigned int &seed = cytnx::random::__static_random_device(),
                             const unsigned int &dtype = Type.Double,
                             const int &device = Device.cpu, const std::string &name = "");

    // Inplace Random Generators:
    /**
     @brief Generate a one-bond UniTensor with all elements are random numbers sampled from a normal
     (Gaussian) distribution, inplacely.
     @details This function provides an inplacely way to generate a UniTensor with all elements are
     random numbers sampled from a normal (Gaussian) distribution. The mean is \p mean and the
     standard deviation is \p std.
      @param[in] mean the mean of the normal distribution.
      @param[in] std the standard deviation of the normal distribution.
      @param[in] seed the seed of the random generator.
      @see random::normal_(UniTensor &in, const double &mean, const double &std, const unsigned int
     &seed)
    */
    void normal_(const double &mean, const double &std,
                 const unsigned int &seed = cytnx::random::__static_random_device());

    /**
      @brief Generate a UniTensor with all elements are random numbers sampled from a uniform
      distribution, inplacely.
      @details This function provides an inplacely way to generate a UniTensor with all elements are
      random numbers sampled from a uniform distribution. The low bound is \p low and the high bound
      is \p high.
      @param[in] low the low bound of the uniform distribution.
      @param[in] high the high bound of the uniform distribution.
      @param[in] seed the seed of the random generator.
      @see random::uniform_(UniTensor &in, const double &low, const double &high, const unsigned int
      &seed)
     */
    void uniform_(const double &low = 0, const double &high = 1,
                  const unsigned int &seed = cytnx::random::__static_random_device());

  };  // class UniTensor

  ///@cond
  std::ostream &operator<<(std::ostream &os, const UniTensor &in);
  ///@endcond

  /**
  @brief Contract two UniTensor by tracing the ranks with common labels.
  @param[in] inL the Tensor #1
  @param[in] inR the Tensor #2
  @param[in] cacheL if the inL should be contiguous align after calling
  @param[in] cacheR if the inR should be contiguous align after calling
  @return
      [UniTensor]

  @see cytnx::UniTensor::contract

  */
  UniTensor Contract(const UniTensor &inL, const UniTensor &inR, const bool &cacheL = false,
                     const bool &cacheR = false);

  /**
  @brief Contract multiple UniTensor by tracing the ranks with common labels with pairwise
  operation.
  @param[in] TNs the Tensors.
  @param[in] order desired contraction order.
  @param[in] optimal wheather to find the optimal contraction order automatically.
  @return
      [UniTensor]

  See also \link cytnx::UniTensor::contract UniTensor.contract \endlink

  */
  UniTensor Contract(const std::vector<UniTensor> &TNs, const std::string &order,
                     const bool &optimal);

  /**
  @deprecated
  @brief Contract multiple UniTensor by tracing the ranks with common labels with pairwise
  operation.
  @param[in] TNs the Tensors.
  @param[in] order desired contraction order.
  @param[in] optimal wheather to find the optimal contraction order automatically.
  @return
      [UniTensor]

  See also \link cytnx::UniTensor::contract UniTensor.contract \endlink

  */
  UniTensor Contracts(const std::vector<UniTensor> &TNs, const std::string &order,
                      const bool &optimal);

  /// @cond
  void _resolve_CT(std::vector<UniTensor> &TNlist);
  template <class... T>
  void _resolve_CT(std::vector<UniTensor> &TNlist, const UniTensor &in, const T &...args) {
    TNlist.push_back(in);
    _resolve_CT(TNlist, args...);
  }
  /// @endcond

  /**
  @brief Contract multiple UniTensor by tracing the ranks with common labels with pairwise
  operation.
  @param in the Tensors.
  @param args the Tensors.
  @return
      [UniTensor]

  See also \link cytnx::UniTensor::contract UniTensor.contract \endlink

  */
  template <class... T>
  UniTensor Contract(const UniTensor &in, const T &...args, const std::string &order,
                     const bool &optimal) {
    std::vector<UniTensor> TNlist;
    _resolve_CT(TNlist, in, args...);
    return Contract(TNlist, order, optimal);
  }

  /**
  @deprecated
  @brief Contract multiple UniTensor by tracing the ranks with common labels with pairwise
  operation.
  @param in the Tensors.
  @param args the Tensors.
  @return
      [UniTensor]

  See also \link cytnx::UniTensor::contract UniTensor.contract \endlink

  */
  template <class... T>
  UniTensor Contracts(const UniTensor &in, const T &...args, const std::string &order,
                      const bool &optimal) {
    std::vector<UniTensor> TNlist;
    _resolve_CT(TNlist, in, args...);
    return Contracts(TNlist, order, optimal);
  }

}  // namespace cytnx

#endif  // BACKEND_TORCH

#endif
