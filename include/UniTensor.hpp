#ifndef _H_UniTensor_
#define _H_UniTensor_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include "Storage.hpp"
#include "Device.hpp"
#include "Tensor.hpp"
#include "Scalar.hpp"
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
//#include "linalg.hpp"

// namespace cytnx{
namespace cytnx {
  using namespace cytnx;
  /// @cond
  class UniTensorType_class {
   public:
    enum : int {
      Void = -99,
      Dense = 0,
      Sparse = 1,
    };
    std::string getname(const int &ut_type);
  };
  extern UniTensorType_class UTenType;
  /// @endcond

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
    std::vector<cytnx_int64> _labels;
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
    friend class SparseUniTensor;

    UniTensor_base()
        : _is_tag(false),
          _name(std::string("")),
          _is_braket_form(false),
          _rowrank(-1),
          _is_diag(false),
          uten_type_id(UTenType.Void){};

    // copy&assignment constr., use intrusive_ptr's !!
    UniTensor_base(const UniTensor_base &rhs);
    UniTensor_base &operator=(UniTensor_base &rhs);

    cytnx_uint64 rowrank() const { return this->_rowrank; }
    bool is_diag() const { return this->_is_diag; }
    const bool &is_braket_form() const { return this->_is_braket_form; }
    const bool &is_tag() const { return this->_is_tag; }
    const std::vector<cytnx_int64> &labels() const { return this->_labels; }
    const std::vector<Bond> &bonds() const { return this->_bonds; }
    std::vector<Bond> &bonds() { return this->_bonds; }
    const std::string &name() const { return this->_name; }
    cytnx_uint64 rank() const { return this->_labels.size(); }
    void set_name(const std::string &in) { this->_name = in; }
    void set_label(const cytnx_int64 &inx, const cytnx_int64 &new_label,
                   const bool &by_label = false) {
      cytnx_int64 idx;
      if (by_label) {
        auto res = std::find(this->_labels.begin(), this->_labels.end(), inx);
        cytnx_error_msg(res == this->_labels.end(), "[ERROR] label %d not exists.\n", inx);
        idx = std::distance(this->_labels.begin(), res);
      } else {
        idx = inx;
      }

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
    void set_labels(const std::vector<cytnx_int64> &new_labels);

    /*
    template<class T>
    T get_elem(const std::vector<cytnx_uint64> &locator) const{
        if(this->is_blockform()){
            if(this->elem_exists(locator)){
                T aux; // [workaround] use aux to dispatch.
                return this->at_for_sparse(locator,aux);
            }else{
                return 0;
            }
        }else{
            return this->at<T>(locator);
        }
    }
    template<class T>
    void set_elem(const std::vector<cytnx_uint64> &locator, const T &input){
        if(this->uten_type()==UTenType.Sparse){
            if(this->elem_exists(locator)){
                T aux;
                this->at_for_sparse(locator,aux) = input;
            }else{
                cytnx_error_msg(true,"[ERROR][SparseUniTensor] invalid location. break qnum
    block.%s","\n");
            }
        }else{
            this->at<T>(locator) = input;
        }
    }
    */

    int uten_type() { return this->uten_type_id; }
    std::string uten_type_str() { return UTenType.getname(this->uten_type_id); }

    virtual void Init(const std::vector<Bond> &bonds,
                      const std::vector<cytnx_int64> &in_labels = {},
                      const cytnx_int64 &rowrank = -1, const unsigned int &dtype = Type.Double,
                      const int &device = Device.cpu, const bool &is_diag = false,
                      const bool &no_alloc = false);
    virtual void Init_by_Tensor(const Tensor &in, const bool &is_diag = false,
                                const cytnx_int64 &rowrank = -1);
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
    virtual void set_rowrank(const cytnx_uint64 &new_rowrank);
    virtual boost::intrusive_ptr<UniTensor_base> permute(const std::vector<cytnx_int64> &mapper,
                                                         const cytnx_int64 &rowrank = -1,
                                                         const bool &by_label = false);
    virtual void permute_(const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank = -1,
                          const bool &by_label = false);
    virtual boost::intrusive_ptr<UniTensor_base> contiguous_();
    virtual boost::intrusive_ptr<UniTensor_base> contiguous();
    virtual void print_diagram(const bool &bond_info = false);

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
    virtual void combineBonds(const std::vector<cytnx_int64> &indicators,
                              const bool &permute_back = false, const bool &by_label = true);
    virtual boost::intrusive_ptr<UniTensor_base> contract(
      const boost::intrusive_ptr<UniTensor_base> &rhs, const bool &mv_elem_self = false,
      const bool &mv_elem_rhs = false);
    virtual std::vector<Bond> getTotalQnums(const bool &physical = false);
    virtual std::vector<std::vector<cytnx_int64>> get_blocks_qnums() const;
    virtual void Trace_(const cytnx_int64 &a, const cytnx_int64 &b, const bool &by_label = false);
    virtual boost::intrusive_ptr<UniTensor_base> Trace(const cytnx_int64 &a, const cytnx_int64 &b,
                                                       const bool &by_label = false);
    virtual boost::intrusive_ptr<UniTensor_base> relabels(
      const std::vector<cytnx_int64> &new_labels);
    virtual boost::intrusive_ptr<UniTensor_base> relabel(const cytnx_int64 &inx,
                                                         const cytnx_int64 &new_label,
                                                         const bool &by_label = false);

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

    virtual boost::intrusive_ptr<UniTensor_base> Conj();
    virtual void Conj_();

    virtual boost::intrusive_ptr<UniTensor_base> Transpose();
    virtual void Transpose_();

    virtual boost::intrusive_ptr<UniTensor_base> Dagger();
    virtual void Dagger_();

    virtual void tag();

    virtual void truncate_(const cytnx_int64 &bond_idx, const cytnx_uint64 &dim,
                           const bool &by_label = false);

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
    void Init(const std::vector<Bond> &bonds, const std::vector<cytnx_int64> &in_labels = {},
              const cytnx_int64 &rowrank = -1, const unsigned int &dtype = Type.Double,
              const int &device = Device.cpu, const bool &is_diag = false,
              const bool &no_alloc = false);
    // this only work for non-symm tensor
    void Init_by_Tensor(const Tensor &in_tensor, const bool &is_diag = false,
                        const cytnx_int64 &rowrank = -1);
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
        return this;
      } else {
        boost::intrusive_ptr<UniTensor_base> out = this->clone();
        out->to_(device);
        return out;
      }
    }
    void set_rowrank(const cytnx_uint64 &new_rowrank) {
      cytnx_error_msg(new_rowrank > this->_labels.size(),
                      "[ERROR] rowrank cannot exceed the rank of UniTensor.%s", "\n");
      this->_rowrank = new_rowrank;
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
    boost::intrusive_ptr<UniTensor_base> permute(const std::vector<cytnx_int64> &mapper,
                                                 const cytnx_int64 &rowrank = -1,
                                                 const bool &by_label = false);
    void permute_(const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank = -1,
                  const bool &by_label = false);
    boost::intrusive_ptr<UniTensor_base> relabels(const std::vector<cytnx_int64> &new_labels);
    boost::intrusive_ptr<UniTensor_base> relabel(const cytnx_int64 &inx,
                                                 const cytnx_int64 &new_label,
                                                 const bool &by_label = false);

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

    void combineBonds(const std::vector<cytnx_int64> &indicators, const bool &permute_back = true,
                      const bool &by_label = true);
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

    boost::intrusive_ptr<UniTensor_base> Dagger() {
      boost::intrusive_ptr<UniTensor_base> out = this->Conj();
      out->Transpose_();
      return out;
    }
    void Dagger_() {
      this->Conj_();
      this->Transpose_();
    }

    void Trace_(const cytnx_int64 &a, const cytnx_int64 &b, const bool &by_label = false);
    boost::intrusive_ptr<UniTensor_base> Trace(const cytnx_int64 &a, const cytnx_int64 &b,
                                               const bool &by_label = false) {
      boost::intrusive_ptr<UniTensor_base> out = this->clone();
      out->Trace_(a, b, by_label);
      return out;
    }

    Tensor Norm() const;

    const Scalar::Sproxy at_for_sparse(const std::vector<cytnx_uint64> &locator) const {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      // return cytnx_complex128(0,0);
    }
    const cytnx_complex128 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                          const cytnx_complex128 &aux) const {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      // return cytnx_complex128(0,0);
    }
    const cytnx_complex64 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                         const cytnx_complex64 &aux) const {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      // return cytnx_complex64(0,0);
    }
    const cytnx_double &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                      const cytnx_double &aux) const {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      // return 0;
    }
    const cytnx_float &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                     const cytnx_float &aux) const {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      // return 0;
    }
    const cytnx_uint64 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                      const cytnx_uint64 &aux) const {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      // return 0;
    }
    const cytnx_int64 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                     const cytnx_int64 &aux) const {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      // return 0;
    }
    const cytnx_uint32 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                      const cytnx_uint32 &aux) const {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      // return 0;
    }
    const cytnx_int32 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                     const cytnx_int32 &aux) const {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      // return 0;
    }
    const cytnx_uint16 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                      const cytnx_uint16 &aux) const {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      // return 0;
    }
    const cytnx_int16 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                     const cytnx_int16 &aux) const {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      // return 0;
    }

    Scalar::Sproxy at_for_sparse(const std::vector<cytnx_uint64> &locator) {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      // return cytnx_complex128(0,0);
    }
    cytnx_complex128 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                    const cytnx_complex128 &aux) {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      // return cytnx_complex128(0,0);
    }
    cytnx_complex64 &at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                   const cytnx_complex64 &aux) {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      // return cytnx_complex64(0,0);
    }
    cytnx_double &at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_double &aux) {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      // return 0;
    }
    cytnx_float &at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_float &aux) {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      // return 0;
    }
    cytnx_uint64 &at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint64 &aux) {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      // return 0;
    }
    cytnx_int64 &at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int64 &aux) {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      // return 0;
    }
    cytnx_uint32 &at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint32 &aux) {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      // return 0;
    }
    cytnx_int32 &at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int32 &aux) {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      // return 0;
    }
    cytnx_uint16 &at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint16 &aux) {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      // return 0;
    }
    cytnx_int16 &at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int16 &aux) {
      cytnx_error_msg(
        true, "[ERROR][Internal] This shouldn't be called by DenseUniTensor, something wrong.%s",
        "\n");
      // return 0;
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
    void truncate_(const cytnx_int64 &bond_idx, const cytnx_uint64 &dim,
                   const bool &by_label = false);

    void _save_dispatch(std::fstream &f) const;
    void _load_dispatch(std::fstream &f);
    // end virtual function
  };
  /// @endcond

  //======================================================================
  /// @cond
  class SparseUniTensor : public UniTensor_base {
   protected:
   public:
    cytnx_uint64 _inner_rowrank;
    std::vector<std::vector<cytnx_int64>> _blockqnums;
    std::vector<cytnx_uint64> _mapper;
    std::vector<cytnx_uint64> _inv_mapper;
    std::vector<std::vector<cytnx_uint64>> _inner2outer_row;
    std::vector<std::vector<cytnx_uint64>> _inner2outer_col;
    std::map<cytnx_uint64, std::pair<cytnx_uint64, cytnx_uint64>> _outer2inner_row;
    std::map<cytnx_uint64, std::pair<cytnx_uint64, cytnx_uint64>> _outer2inner_col;

    std::vector<Tensor> _blocks;

    bool _contiguous;
    void set_meta(SparseUniTensor *tmp, const bool &inner, const bool &outer) const {
      // outer meta
      if (outer) {
        tmp->_bonds = vec_clone(this->_bonds);
        tmp->_labels = this->_labels;
        tmp->_is_braket_form = this->_is_braket_form;
        tmp->_rowrank = this->_rowrank;
        tmp->_name = this->_name;
      }
      // comm meta
      tmp->_mapper = this->_mapper;
      tmp->_inv_mapper = this->_inv_mapper;
      tmp->_contiguous = this->_contiguous;
      tmp->_is_diag = this->_is_diag;

      // inner meta
      if (inner) {
        tmp->_inner_rowrank = this->_inner_rowrank;
        tmp->_inner2outer_row = this->_inner2outer_row;
        tmp->_inner2outer_col = this->_inner2outer_col;
        tmp->_outer2inner_row = this->_outer2inner_row;
        tmp->_outer2inner_col = this->_outer2inner_col;
        tmp->_blockqnums = this->_blockqnums;
      }
    }
    SparseUniTensor *clone_meta(const bool &inner, const bool &outer) const {
      SparseUniTensor *tmp = new SparseUniTensor();
      this->set_meta(tmp, inner, outer);
      return tmp;
    };

    //===================================
    friend class UniTensor;  // allow wrapper to access the private elems
    SparseUniTensor() {
      this->uten_type_id = UTenType.Sparse;
      this->_is_tag = true;
    };

    // virtual functions
    void Init(const std::vector<Bond> &bonds, const std::vector<cytnx_int64> &in_labels = {},
              const cytnx_int64 &rowrank = -1, const unsigned int &dtype = Type.Double,
              const int &device = Device.cpu, const bool &is_diag = false,
              const bool &no_alloc = false);
    void Init_by_Tensor(const Tensor &in_tensor, const bool &is_diag = false,
                        const cytnx_int64 &rowrank = -1) {
      cytnx_error_msg(
        true, "[ERROR][SparseUniTensor] cannot use Init_by_tensor() on a SparseUniTensor.%s", "\n");
    }
    std::vector<cytnx_uint64> shape() const {
      std::vector<cytnx_uint64> out(this->_bonds.size());
      for (cytnx_uint64 i = 0; i < out.size(); i++) {
        out[i] = this->_bonds[i].dim();
      }
      return out;
    }
    bool is_blockform() const { return true; }
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
      SparseUniTensor *tmp = this->clone_meta(true, true);
      tmp->_blocks = vec_clone(this->_blocks);
      boost::intrusive_ptr<UniTensor_base> out(tmp);
      return out;
    };

    bool is_contiguous() const { return this->_contiguous; };
    void set_rowrank(const cytnx_uint64 &new_rowrank) {
      cytnx_error_msg((new_rowrank < 1) || (new_rowrank >= this->rank()),
                      "[ERROR][SparseUniTensor] rowrank should be [>=1] and [<UniTensor.rank].%s",
                      "\n");
      cytnx_error_msg(new_rowrank >= this->_labels.size(),
                      "[ERROR] rowrank cannot exceed the rank of UniTensor.%s", "\n");
      if (this->_inner_rowrank != new_rowrank) this->_contiguous = false;

      this->_rowrank = new_rowrank;
      this->_is_braket_form = this->_update_braket();
    }
    boost::intrusive_ptr<UniTensor_base> relabels(const std::vector<cytnx_int64> &new_labels);
    boost::intrusive_ptr<UniTensor_base> relabel(const cytnx_int64 &inx,
                                                 const cytnx_int64 &new_label,
                                                 const bool &by_label = false);

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

    boost::intrusive_ptr<UniTensor_base> astype(const unsigned int &dtype) const {
      SparseUniTensor *tmp = this->clone_meta(true, true);
      tmp->_blocks.resize(this->_blocks.size());
      for (cytnx_int64 blk = 0; blk < this->_blocks.size(); blk++) {
        tmp->_blocks[blk] = this->_blocks[blk].astype(dtype);
      }
      boost::intrusive_ptr<UniTensor_base> out(tmp);
      return out;
    };

    void permute_(const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank = -1,
                  const bool &by_label = false);
    boost::intrusive_ptr<UniTensor_base> permute(const std::vector<cytnx_int64> &mapper,
                                                 const cytnx_int64 &rowrank = -1,
                                                 const bool &by_label = false);
    boost::intrusive_ptr<UniTensor_base> contiguous();
    boost::intrusive_ptr<UniTensor_base> contiguous_() {
      if (!this->_contiguous) {
        boost::intrusive_ptr<UniTensor_base> titr = this->contiguous();
        SparseUniTensor *tmp = (SparseUniTensor *)titr.get();
        tmp->set_meta(this, true, true);
        this->_blocks = tmp->_blocks;
      }
      return boost::intrusive_ptr<UniTensor_base>(this);
    }
    void print_diagram(const bool &bond_info = false);

    std::vector<Symmetry> syms() const;

    Tensor get_block(const cytnx_uint64 &idx = 0) const {
      cytnx_error_msg(idx >= this->_blocks.size(), "[ERROR][SparseUniTensor] index out of range%s",
                      "\n");
      if (this->_contiguous) {
        return this->_blocks[idx].clone();
      } else {
        cytnx_error_msg(true,
                        "[Developing] get block from a non-contiguous SparseUniTensor is currently "
                        "not support. Call contiguous()/contiguous_() first.%s",
                        "\n");
        return Tensor();
      }
    };
    cytnx_uint64 Nblocks() const { return this->_blocks.size(); };
    Tensor get_block(const std::vector<cytnx_int64> &qnum, const bool &force) const {
      if (!force)
        cytnx_error_msg(
          !this->is_braket_form(),
          "[ERROR][Un-physical] cannot get the block by qnums when bra-ket/in-out bonds mismatch "
          "the row/col space.\n permute to the correct physical space first, then get block.%s",
          "\n");
      // std::cout << "get_block" <<std::endl;
      if (this->_contiguous) {
        // std::cout << "contiguous" << std::endl;
        // get dtype from qnum:
        cytnx_int64 idx = -1;
        for (int i = 0; i < this->_blockqnums.size(); i++) {
          // for(int j=0;j<this->_blockqnums[i].size();j++)
          //     std::cout << this->_blockqnums[i][j]<< " ";
          // std::cout << std::endl;
          if (qnum == this->_blockqnums[i]) {
            idx = i;
            break;
          }
        }
        cytnx_error_msg(
          idx < 0,
          "[ERROR][SparseUniTensor] no block with [qnum] exists in the current UniTensor.%s", "\n");
        return this->get_block(idx);
      } else {
        cytnx_error_msg(true,
                        "[Developing] get block from a non-contiguous SparseUniTensor is currently "
                        "not support. Call contiguous()/contiguous_() first.%s",
                        "\n");
        return Tensor();
      }
      return Tensor();
    };

    // return a share view of block, this only work for symm tensor in contiguous form.
    Tensor &get_block_(const cytnx_uint64 &idx = 0) {
      cytnx_error_msg(
        this->is_contiguous() == false,
        "[ERROR][SparseUniTensor] cannot use get_block_() on non-contiguous UniTensor with "
        "symmetry.\n suggest options: \n  1) Call contiguous_()/contiguous() first, then call "
        "get_block_()\n  2) Try get_block()/get_blocks()%s",
        "\n");

      cytnx_error_msg(idx >= this->_blocks.size(),
                      "[ERROR][SparseUniTensor] index exceed the number of blocks.%s", "\n");

      return this->_blocks[idx];
    }
    const Tensor &get_block_(const cytnx_uint64 &idx = 0) const {
      cytnx_error_msg(
        this->is_contiguous() == false,
        "[ERROR][SparseUniTensor] cannot use get_block_() on non-contiguous UniTensor with "
        "symmetry.\n suggest options: \n  1) Call contiguous_()/contiguous() first, then call "
        "get_block_()\n  2) Try get_block()/get_blocks()%s",
        "\n");

      cytnx_error_msg(idx >= this->_blocks.size(),
                      "[ERROR][SparseUniTensor] index exceed the number of blocks.%s", "\n");

      return this->_blocks[idx];
    }

    Tensor &get_block_(const std::vector<cytnx_int64> &qnum, const bool &force) {
      if (!force)
        cytnx_error_msg(
          !this->is_braket_form(),
          "[ERROR][Un-physical] cannot get the block by qnums when bra-ket/in-out bonds mismatch "
          "the row/col space.\n permute to the correct physical space first, then get block.%s",
          "\n");

      cytnx_error_msg(
        this->is_contiguous() == false,
        "[ERROR][SparseUniTensor] cannot use get_block_() on non-contiguous UniTensor with "
        "symmetry.\n suggest options: \n  1) Call contiguous_()/contiguous() first, then call "
        "get_blocks_()\n  2) Try get_block()/get_blocks()%s",
        "\n");

      // get dtype from qnum:
      cytnx_int64 idx = -1;
      for (int i = 0; i < this->_blockqnums.size(); i++) {
        if (qnum == this->_blockqnums[i]) {
          idx = i;
          break;
        }
      }
      cytnx_error_msg(
        idx < 0, "[ERROR][SparseUniTensor] no block with [qnum] exists in the current UniTensor.%s",
        "\n");
      return this->get_block_(idx);
      // cytnx_error_msg(true,"[Developing]%s","\n");
    }
    const Tensor &get_block_(const std::vector<cytnx_int64> &qnum, const bool &force) const {
      if (!force)
        cytnx_error_msg(
          !this->is_braket_form(),
          "[ERROR][Un-physical] cannot get the block by qnums when bra-ket/in-out bonds mismatch "
          "the row/col space.\n permute to the correct physical space first, then get block.%s",
          "\n");

      cytnx_error_msg(
        this->is_contiguous() == false,
        "[ERROR][SparseUniTensor] cannot use get_block_() on non-contiguous UniTensor with "
        "symmetry.\n suggest options: \n  1) Call contiguous_()/contiguous() first, then call "
        "get_blocks_()\n  2) Try get_block()/get_blocks()%s",
        "\n");

      // get dtype from qnum:
      cytnx_int64 idx = -1;
      for (int i = 0; i < this->_blockqnums.size(); i++) {
        if (qnum == this->_blockqnums[i]) {
          idx = i;
          break;
        }
      }
      cytnx_error_msg(
        idx < 0, "[ERROR][SparseUniTensor] no block with [qnum] exists in the current UniTensor.%s",
        "\n");
      return this->get_block_(idx);
    }

    std::vector<Tensor> get_blocks() const {
      if (this->_contiguous) {
        return vec_clone(this->_blocks);
      } else {
        // cytnx_error_msg(true,"[Developing]%s","\n");
        boost::intrusive_ptr<UniTensor_base> tmp = this->clone();
        tmp->contiguous_();
        SparseUniTensor *ttmp = (SparseUniTensor *)tmp.get();
        return ttmp->_blocks;
      }
    };

    const std::vector<Tensor> &get_blocks_(const bool &silent = false) const {
      // cout << "[call this]" << endl;
      if (this->_contiguous) {
        return this->_blocks;
      } else {
        // cytnx_error_msg(true,"[Developing]%s","\n");
        if (!silent)
          cytnx_warning_msg(
            true,
            "[WARNING][SparseUniTensor] call get_blocks_() with a non-contiguous UniTensor should "
            "be used with caution. \ntry: \n1) get_blocks()\n2) call contiguous/contiguous_() "
            "first, then get_blocks_() to get concise results%s",
            "\n");

        return this->_blocks;
      }
    };
    std::vector<Tensor> &get_blocks_(const bool &silent = false) {
      // cout << "[call this]" << endl;
      if (this->_contiguous) {
        return this->_blocks;
      } else {
        if (!silent)
          cytnx_warning_msg(
            true,
            "[WARNING][SparseUniTensor] call get_blocks_() with a non-contiguous UniTensor should "
            "be used with caution. \ntry: \n1) get_blocks()\n2) call contiguous/contiguous_() "
            "first, then get_blocks_() to get concise results%s",
            "\n");

        return this->_blocks;
      }
    };

    bool same_data(const boost::intrusive_ptr<UniTensor_base> &rhs) const {
      if (rhs->uten_type() != UTenType.Sparse) return false;
      if (rhs->get_blocks_(1).size() != this->get_blocks_(1).size()) return false;

      for (int i = 0; i < rhs->get_blocks_(1).size(); i++)
        if (this->get_blocks_(1)[i].same_data(rhs->get_blocks_(1)[i]) == false) return false;

      return true;
    }

    void put_block_(Tensor &in, const cytnx_uint64 &idx = 0) {
      cytnx_error_msg(
        this->is_contiguous() == false,
        "[ERROR][SparseUniTensor] cannot use put_block_() on non-contiguous UniTensor with "
        "symmetry.\n suggest options: \n  1) Call contiguous_()/contiguous() first, then call "
        "put_blocks_()\n  2) Try put_block()/put_blocks()%s",
        "\n");

      cytnx_error_msg(idx >= this->_blocks.size(), "[ERROR][SparseUniTensor] index out of range%s",
                      "\n");
      cytnx_error_msg(in.shape() != this->_blocks[idx].shape(),
                      "[ERROR][SparseUniTensor] the shape of input tensor does not match the shape "
                      "of block @ idx=%d\n",
                      idx);
      this->_blocks[idx] = in;
    };
    void put_block(const Tensor &in, const cytnx_uint64 &idx = 0) {
      cytnx_error_msg(idx >= this->_blocks.size(), "[ERROR][SparseUniTensor] index out of range%s",
                      "\n");
      if (this->_contiguous) {
        cytnx_error_msg(in.shape() != this->_blocks[idx].shape(),
                        "[ERROR][SparseUniTensor] the shape of input tensor does not match the "
                        "shape of block @ idx=%d\n",
                        idx);
        this->_blocks[idx] = in.clone();
      } else {
        cytnx_error_msg(true,
                        "[Developing] put block to a non-contiguous SparseUniTensor is currently "
                        "not support. Call contiguous()/contiguous_() first.%s",
                        "\n");
      }
    };
    void put_block(const Tensor &in, const std::vector<cytnx_int64> &qnum, const bool &force) {
      if (!force)
        cytnx_error_msg(
          !this->is_braket_form(),
          "[ERROR][Un-physical] cannot get the block by qnums when bra-ket/in-out bonds mismatch "
          "the row/col space.\n permute to the correct physical space first, then get block.%s",
          "\n");

      // get dtype from qnum:
      cytnx_int64 idx = -1;
      for (int i = 0; i < this->_blockqnums.size(); i++) {
        if (qnum == this->_blockqnums[i]) {
          idx = i;
          break;
        }
      }
      cytnx_error_msg(
        idx < 0, "[ERROR][SparseUniTensor] no block with [qnum] exists in the current UniTensor.%s",
        "\n");
      this->put_block(in, idx);
    };
    void put_block_(Tensor &in, const std::vector<cytnx_int64> &qnum, const bool &force) {
      if (!force)
        cytnx_error_msg(
          !this->is_braket_form(),
          "[ERROR][Un-physical] cannot get the block by qnums when bra-ket/in-out bonds mismatch "
          "the row/col space.\n permute to the correct physical space first, then get block.%s",
          "\n");

      // get dtype from qnum:
      cytnx_int64 idx = -1;
      for (int i = 0; i < this->_blockqnums.size(); i++) {
        if (qnum == this->_blockqnums[i]) {
          idx = i;
          break;
        }
      }
      cytnx_error_msg(
        idx < 0, "[ERROR][SparseUniTensor] no block with [qnum] exists in the current UniTensor.%s",
        "\n");
      this->put_block_(in, idx);
    };

    // this will only work on non-symm tensor (DenseUniTensor)
    boost::intrusive_ptr<UniTensor_base> get(const std::vector<Accessor> &accessors) {
      cytnx_error_msg(true,
                      "[ERROR][SparseUniTensor][get] cannot use get on a UniTensor with "
                      "Symmetry.\n suggestion: try get_block()/get_blocks() first.%s",
                      "\n");
      return nullptr;
    }
    // this will only work on non-symm tensor (DenseUniTensor)
    void set(const std::vector<Accessor> &accessors, const Tensor &rhs) {
      cytnx_error_msg(true,
                      "[ERROR][SparseUniTensor][set] cannot use set on a UniTensor with "
                      "Symmetry.\n suggestion: try get_block()/get_blocks() first.%s",
                      "\n");
    }
    void reshape_(const std::vector<cytnx_int64> &new_shape, const cytnx_uint64 &rowrank = 0) {
      cytnx_error_msg(true, "[ERROR] cannot reshape a UniTensor with symmetry.%s", "\n");
    }
    boost::intrusive_ptr<UniTensor_base> reshape(const std::vector<cytnx_int64> &new_shape,
                                                 const cytnx_uint64 &rowrank = 0) {
      cytnx_error_msg(true, "[ERROR] cannot reshape a UniTensor with symmetry.%s", "\n");
      return nullptr;
    }
    boost::intrusive_ptr<UniTensor_base> to_dense() {
      cytnx_error_msg(true, "[ERROR] cannot to_dense a UniTensor with symmetry.%s", "\n");
      return nullptr;
    }
    void to_dense_() {
      cytnx_error_msg(true, "[ERROR] cannot to_dense_ a UniTensor with symmetry.%s", "\n");
    }
    void combineBonds(const std::vector<cytnx_int64> &indicators, const bool &permute_back = true,
                      const bool &by_label = true) {
      cytnx_error_msg(true, "[Developing]%s", "\n");
    };
    boost::intrusive_ptr<UniTensor_base> contract(const boost::intrusive_ptr<UniTensor_base> &rhs,
                                                  const bool &mv_elem_self = false,
                                                  const bool &mv_elem_rhs = false);
    std::vector<Bond> getTotalQnums(const bool &physical = false);
    std::vector<std::vector<cytnx_int64>> get_blocks_qnums() const { return this->_blockqnums; }
    ~SparseUniTensor(){};

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
    boost::intrusive_ptr<UniTensor_base> Trace(const cytnx_int64 &a, const cytnx_int64 &b,
                                               const bool &by_label = false);
    void Trace_(const cytnx_int64 &a, const cytnx_int64 &b, const bool &by_label = false) {
      cytnx_error_msg(true,
                      "[ERROR] Currently SparseUniTensor does not support inplace Trace!, call "
                      "Trace() instead!%s",
                      "\n");
    }

    void Transpose_();
    boost::intrusive_ptr<UniTensor_base> Transpose() {
      boost::intrusive_ptr<UniTensor_base> out = this->clone();
      out->Transpose_();
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

    Tensor Norm() const;

    void tag() {
      // no-use!
    }

    void truncate_(const cytnx_int64 &bond_idx, const cytnx_uint64 &dim,
                   const bool &by_label = false);
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

    bool elem_exists(const std::vector<cytnx_uint64> &locator) const;
    void _save_dispatch(std::fstream &f) const;
    void _load_dispatch(std::fstream &f);
    // end virtual func
  };
  /// @endcond

  //======================================================================

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
    @brief Initialize a UniTensor with cytnx::Tensor.
    @param in_tensor a cytnx::Tensor
    @param is_diag if the current UniTensor is a diagonal Tensor. This will requires input of Tensor
    to be 1D.
    @param rowrank the rowrank of the outcome UniTensor.

    [Note]
        1. The constructed UniTensor will have same rank as the input Tensor, with default labels,
    and a shared view (shared instance) of interal block as the input Tensor.
        2. The constructed UniTensor is always untagged.
        3. [Important] The internal block of UniTensor is a referece of input Tensor. That is, they
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
                       const cytnx_int64 &rowrank = -1)
        : _impl(new UniTensor_base()) {
      this->Init(in_tensor, is_diag, rowrank);
    }
    void Init(const Tensor &in_tensor, const bool &is_diag = false,
              const cytnx_int64 &rowrank = -1) {
      boost::intrusive_ptr<UniTensor_base> out(new DenseUniTensor());
      out->Init_by_Tensor(in_tensor, is_diag, rowrank);
      this->_impl = out;
    }
    //@}

    //@{
    /**
    @brief Initialize a UniTensor.
    @param bonds the bond list. when init, each bond will be copy( not a shared view of bond object
    with input bond)
    @param in_labels the labels for each rank(bond)
    @param rowrank the rank of physical row space.
    @param dtype the dtype of the UniTensor. It can be any type defined in cytnx::Type.
    @param device the device that the UniTensor is put on. It can be any device defined in
    cytnx::Device.
    @param is_diag if the constructed UniTensor is a diagonal UniTensor.
        This can only be assigned true when the UniTensor is square, untagged and rank-2 UniTensor.

    [Note]
        1. the bonds cannot contain simutaneously untagged bond(s) and tagged bond(s)
        2. If the bonds are with symmetry (qnums), the symmetry types should be the same across all
    the bonds.

    */
    UniTensor(const std::vector<Bond> &bonds, const std::vector<cytnx_int64> &in_labels = {},
              const cytnx_int64 &rowrank = -1, const unsigned int &dtype = Type.Double,
              const int &device = Device.cpu, const bool &is_diag = false)
        : _impl(new UniTensor_base()) {
#ifdef UNI_DEBUG
      cytnx_warning_msg(
        true,
        "[DEBUG] message: entry for UniTensor(const std::vector<Bond> &bonds, const "
        "std::vector<cytnx_int64> &in_labels={}, const cytnx_int64 &rowrank=-1, const unsigned int "
        "&dtype=Type.Double, const int &device = Device.cpu, const bool &is_diag=false)%s",
        "\n");
#endif
      this->Init(bonds, in_labels, rowrank, dtype, device, is_diag);
    }
    void Init(const std::vector<Bond> &bonds, const std::vector<cytnx_int64> &in_labels = {},
              const cytnx_int64 &rowrank = -1, const unsigned int &dtype = Type.Double,
              const int &device = Device.cpu, const bool &is_diag = false) {
      // checking type:
      bool is_sym = false;
      for (cytnx_uint64 i = 0; i < bonds.size(); i++) {
        // check
        if (bonds[i].syms().size() != 0)
          is_sym = true;
        else
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
        boost::intrusive_ptr<UniTensor_base> out(new SparseUniTensor());
        this->_impl = out;
      } else {
        boost::intrusive_ptr<UniTensor_base> out(new DenseUniTensor());
        this->_impl = out;
      }
      this->_impl->Init(bonds, in_labels, rowrank, dtype, device, is_diag, false);
    }
    //@}

    /**
    @brief set the name of the UniTensor
    @param in the name. It should be a string.

    */
    UniTensor &set_name(const std::string &in) {
      this->_impl->set_name(in);
      return *this;
    }
    /**
    @brief set a new label for bond at the assigned index.
    @param idx the index of the bond.
    @param new_label the new label that is assign to the bond.

    [Note]
        the new assign label cannot be the same as the label of any other bonds in the UniTensor.
        ( cannot have duplicate labels )

    */
    UniTensor &set_label(const cytnx_int64 &idx, const cytnx_int64 &new_label,
                         const bool &by_label = false) {
      this->_impl->set_label(idx, new_label, by_label);
      return *this;
    }

    /**
    @brief change a new label for bond with original label.
    @param old_lbl the original label of the bond that to be replaced.
    @param new_label the new label that is assign to replace the original label.

    [Note]
        the new assign label cannot be the same as the label of any other bonds in the UniTensor.
        ( cannot have duplicate labels )

    */
    /*
    UniTensor& change_label(const cytnx_int64 &old_lbl, const cytnx_int64 &new_label){
        this->_impl->change_label(old_lbl,new_label);
        return *this;
    }
    */

    /**
    @brief set new labels for all the bonds.
    @param new_labels the new labels for each bond.

    [Note]
        the new assign labels cannot have duplicate element(s), and should have the same size as the
    rank of UniTensor.

    */
    UniTensor &set_labels(const std::vector<cytnx_int64> &new_labels) {
      this->_impl->set_labels(new_labels);
      return *this;
    }
    UniTensor &set_rowrank(const cytnx_uint64 &new_rowrank) {
      this->_impl->set_rowrank(new_rowrank);
      return *this;
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

    cytnx_uint64 Nblocks() const { return this->_impl->Nblocks(); }
    cytnx_uint64 rank() const { return this->_impl->rank(); }
    cytnx_uint64 rowrank() const { return this->_impl->rowrank(); }
    unsigned int dtype() const { return this->_impl->dtype(); }
    int uten_type() const { return this->_impl->uten_type(); }
    int device() const { return this->_impl->device(); }
    std::string name() const { return this->_impl->name(); }
    std::string dtype_str() const { return this->_impl->dtype_str(); }
    std::string device_str() const { return this->_impl->device_str(); }
    std::string uten_type_str() const { return this->_impl->uten_type_str(); }
    bool is_contiguous() const { return this->_impl->is_contiguous(); }
    bool is_diag() const { return this->_impl->is_diag(); }
    bool is_tag() const { return this->_impl->is_tag(); }
    std::vector<Symmetry> syms() const { return this->_impl->syms(); }
    const bool &is_braket_form() const { return this->_impl->is_braket_form(); }
    const std::vector<cytnx_int64> &labels() const { return this->_impl->labels(); }
    const std::vector<Bond> &bonds() const { return this->_impl->bonds(); }
    std::vector<Bond> &bonds() { return this->_impl->bonds(); }
    std::vector<cytnx_uint64> shape() const { return this->_impl->shape(); }
    bool is_blockform() const { return this->_impl->is_blockform(); }

    void to_(const int &device) { this->_impl->to_(device); }
    UniTensor to(const int &device) const {
      UniTensor out;
      out._impl = this->_impl->to(device);
      return out;
    }
    UniTensor clone() const {
      UniTensor out;
      out._impl = this->_impl->clone();
      return out;
    }
    UniTensor relabels(const std::vector<cytnx_int64> &new_labels) const {
      UniTensor out;
      out._impl = this->_impl->relabels(new_labels);
      return out;
    }
    UniTensor relabel(const cytnx_int64 &inx, const cytnx_int64 &new_label,
                      const bool &by_label = false) const {
      UniTensor out;
      out._impl = this->_impl->relabel(inx, new_label, by_label);
      return out;
    }

    UniTensor astype(const unsigned int &dtype) const {
      UniTensor out;
      if (this->dtype() == dtype) {
        out._impl = this->_impl;
      } else {
        out._impl = this->_impl->astype(dtype);
      }
      return out;
    }

    UniTensor permute(const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank = -1,
                      const bool &by_label = false) {
      UniTensor out;
      out._impl = this->_impl->permute(mapper, rowrank, by_label);
      return out;
    }
    void permute_(const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank = -1,
                  const bool &by_label = false) {
      this->_impl->permute_(mapper, rowrank, by_label);
    }
    UniTensor contiguous() const {
      UniTensor out;
      out._impl = this->_impl->contiguous();
      return out;
    }
    void contiguous_() { this->_impl = this->_impl->contiguous_(); }
    void print_diagram(const bool &bond_info = false) { this->_impl->print_diagram(bond_info); }

    template <class T>
    T &at(const std::vector<cytnx_uint64> &locator) {
      // std::cout << "at " << this->is_blockform()  << std::endl;
      if (this->uten_type() == UTenType.Sparse) {
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

    template <class T>
    const T &at(const std::vector<cytnx_uint64> &locator) const {
      // std::cout << "at " << this->is_blockform()  << std::endl;
      if (this->uten_type() == UTenType.Sparse) {
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

    const Scalar::Sproxy at(const std::vector<cytnx_uint64> &locator) const {
      if (this->uten_type() == UTenType.Sparse) {
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

    Scalar::Sproxy at(const std::vector<cytnx_uint64> &locator) {
      if (this->uten_type() == UTenType.Sparse) {
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

    // return a clone of block
    Tensor get_block(const cytnx_uint64 &idx = 0) const { return this->_impl->get_block(idx); };
    //================================
    // return a clone of block
    Tensor get_block(const std::vector<cytnx_int64> &qnum, const bool &force = false) const {
      return this->_impl->get_block(qnum, force);
    }
    Tensor get_block(const std::initializer_list<cytnx_int64> &qnum,
                     const bool &force = false) const {
      std::vector<cytnx_int64> tmp = qnum;
      return get_block(tmp, force);
    }
    //================================
    // this only work for non-symm tensor. return a shared view of block
    const Tensor &get_block_(const cytnx_uint64 &idx = 0) const {
      return this->_impl->get_block_(idx);
    }
    //================================
    // this only work for non-symm tensor. return a shared view of block
    Tensor &get_block_(const cytnx_uint64 &idx = 0) { return this->_impl->get_block_(idx); }
    //================================
    // this only work for non-symm tensor. return a shared view of block
    Tensor &get_block_(const std::vector<cytnx_int64> &qnum, const bool &force = false) {
      return this->_impl->get_block_(qnum, force);
    }
    Tensor &get_block_(const std::initializer_list<cytnx_int64> &qnum, const bool &force = false) {
      std::vector<cytnx_int64> tmp = qnum;
      return get_block_(tmp, force);
    }
    //================================

    // this only work for non-symm tensor. return a shared view of block
    const Tensor &get_block_(const std::vector<cytnx_int64> &qnum,
                             const bool &force = false) const {
      return this->_impl->get_block_(qnum, force);
    }
    const Tensor &get_block_(const std::initializer_list<cytnx_int64> &qnum,
                             const bool &force = false) const {
      std::vector<cytnx_int64> tmp = qnum;
      return this->_impl->get_block_(tmp, force);
    }
    //================================
    // this return a shared view of blocks for non-symm tensor.
    // for symmetry tensor, it call contiguous first and return a shared view of blocks. [dev]
    std::vector<Tensor> get_blocks() const { return this->_impl->get_blocks(); }
    // this return a shared view of blocks for non-symm tensor.
    // for symmetry tensor, it call contiguous first and return a shared view of blocks. [dev]
    const std::vector<Tensor> &get_blocks_(const bool &silent = false) const {
      return this->_impl->get_blocks_(silent);
    }
    // for symmetry tensor, it call contiguous first and return a shared view of blocks. [dev]
    std::vector<Tensor> &get_blocks_(const bool &silent = false) {
      return this->_impl->get_blocks_(silent);
    }

    // the put block will have shared view with the internal block, i.e. non-clone.
    void put_block(const Tensor &in, const cytnx_uint64 &idx = 0) {
      this->_impl->put_block(in, idx);
    }
    // the put block will have shared view with the internal block, i.e. non-clone.
    void put_block(const Tensor &in, const std::vector<cytnx_int64> &qnum, const bool &force) {
      this->_impl->put_block(in, qnum, force);
    }
    // the put block will have shared view with the internal block, i.e. non-clone.
    void put_block_(Tensor &in, const cytnx_uint64 &idx = 0) { this->_impl->put_block_(in, idx); }
    // the put block will have shared view with the internal block, i.e. non-clone.
    void put_block_(Tensor &in, const std::vector<cytnx_int64> &qnum, const bool &force) {
      this->_impl->put_block_(in, qnum, force);
    }
    UniTensor get(const std::vector<Accessor> &accessors) const {
      UniTensor out;
      out._impl = this->_impl->get(accessors);
      return out;
    }
    void set(const std::vector<Accessor> &accessors, const Tensor &rhs) {
      this->_impl->set(accessors, rhs);
    }
    UniTensor reshape(const std::vector<cytnx_int64> &new_shape, const cytnx_uint64 &rowrank = 0) {
      UniTensor out;
      out._impl = this->_impl->reshape(new_shape, rowrank);
      return out;
    }
    void reshape_(const std::vector<cytnx_int64> &new_shape, const cytnx_uint64 &rowrank = 0) {
      this->_impl->reshape_(new_shape, rowrank);
    }
    UniTensor to_dense() {
      UniTensor out;
      out._impl = this->_impl->to_dense();
      return out;
    }
    void to_dense_() { this->_impl->to_dense_(); }
    void combineBonds(const std::vector<cytnx_int64> &indicators, const bool &permute_back = true,
                      const bool &by_label = true) {
      this->_impl->combineBonds(indicators, permute_back, by_label);
    }
    UniTensor contract(const UniTensor &inR, const bool &mv_elem_self = false,
                       const bool &mv_elem_rhs = false) const {
      UniTensor out;
      out._impl = this->_impl->contract(inR._impl, mv_elem_self, mv_elem_rhs);
      return out;
    }
    std::vector<Bond> getTotalQnums(const bool physical = false) const {
      return this->_impl->getTotalQnums(physical);
    }
    std::vector<std::vector<cytnx_int64>> get_blocks_qnums() const {
      return this->_impl->get_blocks_qnums();
    }

    bool same_data(const UniTensor &rhs) const {
      // check same type:
      if (this->_impl->uten_type() != rhs._impl->uten_type()) return false;

      return this->_impl->same_data(rhs._impl);
    }

    UniTensor &Add_(const UniTensor &rhs) {
      this->_impl->Add_(rhs._impl);
      return *this;
    }

    UniTensor &Mul_(const UniTensor &rhs) {
      this->_impl->Mul_(rhs._impl);
      return *this;
    }

    UniTensor &Sub_(const UniTensor &rhs) {
      this->_impl->Sub_(rhs._impl);
      return *this;
    }

    UniTensor &Div_(const UniTensor &rhs) {
      this->_impl->Div_(rhs._impl);
      return *this;
    }

    UniTensor &Add_(const Scalar &rhs) {
      this->_impl->Add_(rhs);
      return *this;
    }

    UniTensor &Mul_(const Scalar &rhs) {
      this->_impl->Mul_(rhs);
      return *this;
    }

    UniTensor &Sub_(const Scalar &rhs) {
      this->_impl->Sub_(rhs);
      return *this;
    }

    UniTensor &Div_(const Scalar &rhs) {
      this->_impl->Div_(rhs);
      return *this;
    }

    UniTensor Add(const UniTensor &rhs) const;
    UniTensor Add(const Scalar &rhs) const;
    UniTensor Mul(const UniTensor &rhs) const;
    UniTensor Mul(const Scalar &rhs) const;
    UniTensor Div(const UniTensor &rhs) const;
    UniTensor Div(const Scalar &rhs) const;
    UniTensor Sub(const UniTensor &rhs) const;
    UniTensor Sub(const Scalar &rhs) const;

    Tensor Norm() const { return this->_impl->Norm(); };

    UniTensor &operator+=(const UniTensor &rhs) {
      this->Add_(rhs);
      return *this;
    }
    UniTensor &operator-=(const UniTensor &rhs) {
      this->Sub_(rhs);
      return *this;
    }
    UniTensor &operator/=(const UniTensor &rhs) {
      this->Div_(rhs);
      return *this;
    }
    UniTensor &operator*=(const UniTensor &rhs) {
      this->Mul_(rhs);
      return *this;
    }
    UniTensor &operator+=(const Scalar &rhs) {
      this->Add_(rhs);
      return *this;
    }
    UniTensor &operator-=(const Scalar &rhs) {
      this->Sub_(rhs);
      return *this;
    }
    UniTensor &operator/=(const Scalar &rhs) {
      this->Div_(rhs);
      return *this;
    }
    UniTensor &operator*=(const Scalar &rhs) {
      this->Mul_(rhs);
      return *this;
    }

    UniTensor Conj() {
      UniTensor out;
      out._impl = this->_impl->Conj();
      return out;
    }

    UniTensor &Conj_() {
      this->_impl->Conj_();
      return *this;
    }

    UniTensor Transpose() const {
      UniTensor out;
      out._impl = this->_impl->Transpose();
      return out;
    }
    UniTensor &Transpose_() {
      this->_impl->Transpose_();
      return *this;
    }

    UniTensor Trace(const cytnx_int64 &a = 0, const cytnx_int64 &b = 1,
                    const bool &by_label = false) const {
      UniTensor out;
      out._impl = this->_impl->Trace(a, b, by_label);
      return out;
    }

    UniTensor &Trace_(const cytnx_int64 &a = 0, const cytnx_int64 &b = 1,
                      const bool &by_label = false) {
      this->_impl->Trace_(a, b, by_label);
      return *this;
    }

    UniTensor Dagger() const {
      UniTensor out;
      out._impl = this->_impl->Dagger();
      return out;
    }

    UniTensor &Dagger_() {
      this->_impl->Dagger_();
      return *this;
    }

    UniTensor &tag() {
      this->_impl->tag();
      return *this;
    }

    UniTensor Pow(const double &p) const;
    UniTensor &Pow_(const double &p);

    bool elem_exists(const std::vector<cytnx_uint64> &locator) const {
      return this->_impl->elem_exists(locator);
    }

    // [C++: Deprecated soon, use at]
    template <class T>
    T get_elem(const std::vector<cytnx_uint64> &locator) const {
      return this->at<T>(locator);
    }

    // [C++: Deprecated soon, use at]
    template <class T2>
    void set_elem(const std::vector<cytnx_uint64> &locator, const T2 &rc) {
      // cytnx_error_msg(true,"[ERROR] invalid type%s","\n");
      this->at(locator) = rc;
    }

    void Save(const std::string &fname) const;
    void Save(const char *fname) const;
    static UniTensor Load(const std::string &fname);
    static UniTensor Load(const char *fname);

    UniTensor &truncate_(const cytnx_int64 &bond_idx, const cytnx_uint64 &dim,
                         const bool &by_label = false) {
      this->_impl->truncate_(bond_idx, dim, by_label);
      return *this;
    }
    UniTensor truncate(const cytnx_int64 &bond_idx, const cytnx_uint64 &dim,
                       const bool &by_label = false) const {
      UniTensor out = this->clone();
      out.truncate_(bond_idx, dim, by_label);
      return out;
    }

    /// @cond
    void _Load(std::fstream &f);
    void _Save(std::fstream &f) const;
    /// @endcond

  };  // class UniTensor

  ///@cond
  std::ostream &operator<<(std::ostream &os, const UniTensor &in);
  ///@endcond

  /**
  @brief Contract two UniTensor by tracing the ranks with common labels.
  @param inL the Tensor #1
  @param inR the Tensor #2
  @param cacheL if the inL should be contiguous align after calling
  @param cacheR if the inR should be contiguous align after calling
  @return
      [UniTensor]

  See also \link cytnx::UniTensor::contract UniTensor.contract \endlink

  */
  UniTensor Contract(const UniTensor &inL, const UniTensor &inR, const bool &cacheL = false,
                     const bool &cacheR = false);

  /**
  @brief Contract multiple UniTensor by tracing the ranks with common labels with pairwise
  operation.
  @param TNs the Tensors.
  @return
      [UniTensor]

  See also \link cytnx::UniTensor::contract UniTensor.contract \endlink

  */
  UniTensor Contracts(const std::vector<UniTensor> &TNs);

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
  UniTensor Contracts(const UniTensor &in, const T &...args) {
    std::vector<UniTensor> TNlist;
    _resolve_CT(TNlist, in, args...);
    return Contracts(TNlist);
  }

}  // namespace cytnx
#endif
