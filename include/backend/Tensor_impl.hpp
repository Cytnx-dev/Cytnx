#ifndef _H_Tensor_impl_
#define _H_Tensor_impl_
#ifndef BACKEND_TORCH
  #include "Type.hpp"
  #include "cytnx_error.hpp"
  #include "backend/Storage.hpp"
  #include "Device.hpp"
  #include "intrusive_ptr_base.hpp"
  #include <iostream>
  #include <fstream>
  #include "utils/vec_range.hpp"
  #include "utils/vec_cast.hpp"
  #include "Accessor.hpp"
  #include <utility>
  #include <vector>
  #include <initializer_list>
  #include <string>
  #include "backend/Scalar.hpp"

namespace cytnx {

  ///@cond
  // real implementation
  class Tensor_impl : public intrusive_ptr_base<Tensor_impl> {
   private:
    // Interface:
    Storage_init_interface __SII;

    // Memory:
    Storage _storage;

    // tensor shape
    std::vector<cytnx_uint64> _shape;

    // pseudo-perm info
    std::vector<cytnx_uint64> _mapper;
    std::vector<cytnx_uint64> _invmapper;
    bool _contiguous;

   public:
    friend class Tensor;
    boost::intrusive_ptr<Tensor_impl> _clone_meta_only() const {
      boost::intrusive_ptr<Tensor_impl> out(new Tensor_impl());
      out->_mapper = this->_mapper;
      out->_invmapper = this->_invmapper;
      out->_shape = this->_shape;
      out->_contiguous = this->_contiguous;
      return out;
    }
    Tensor_impl() : _contiguous(true){};

    void Init(const std::vector<cytnx_uint64> &shape, const unsigned int &dtype = Type.Double,
              int device = -1, const bool &init_zero = true);
    void Init(const Storage &in);
    // void Init(const Storage &in, const std::vector<cytnx_uint64> &shape,
    // const unsigned int &dtype, int device);
    /*
    template<class T>
    void From_vec(const T &ndvec){
        cytnx_error_msg(std::string(typeid(T).name()).find("vector") ==
    std::string::npos,"[ERROR][Tensor][From_vec] the input argument should be a nd vector.%s","\n");
        //dispatch the rank!:



    }
    */
    // clone&assignment constr., use intrusive_ptr's
    Tensor_impl(const Tensor_impl &rhs);
    Tensor_impl &operator=(const Tensor_impl &rhs);  // add const

    unsigned int dtype() const { return this->_storage.dtype(); }
    int device() const { return this->_storage.device(); }

    std::string dtype_str() const { return Type.getname(this->_storage.dtype()); }
    std::string device_str() const { return Device.getname(this->_storage.device()); }

    const std::vector<cytnx_uint64> &shape() const { return _shape; }

    const bool &is_contiguous() const { return this->_contiguous; }

    const std::vector<cytnx_uint64> &mapper() const { return this->_mapper; }
    const std::vector<cytnx_uint64> &invmapper() const { return this->_invmapper; }
    Storage &storage() { return _storage; }

    const Storage &storage() const { return _storage; }

    boost::intrusive_ptr<Tensor_impl> clone() const {
      boost::intrusive_ptr<Tensor_impl> out = this->_clone_meta_only();
      out->_storage = this->_storage.clone();
      return out;
    }

    void to_(const int &device) { this->_storage.to_(device); }
    boost::intrusive_ptr<Tensor_impl> to(const int &device) {
      if (this->device() == device) {
        // boost::intrusive_ptr<Tensor_impl> out(this);
        return this;
      } else {
        boost::intrusive_ptr<Tensor_impl> out = this->_clone_meta_only();
        out->_storage = this->_storage.to(device);
        return out;
      }
    }

    void permute_(const std::vector<cytnx_uint64> &rnks);

    boost::intrusive_ptr<Tensor_impl> permute(const std::vector<cytnx_uint64> &rnks);

    template <class T>
    T &at(const std::vector<cytnx_uint64> &locator) const {
      cytnx_error_msg(locator.size() != this->_shape.size(), "%s",
                      "The input index does not match Tensor's rank.");

      cytnx_uint64 RealRank, mtplyr;
      // std::vector<cytnx_uint64> c_shape(this->_shape.size());
      // std::vector<cytnx_uint64> c_loc(this->_shape.size());
      cytnx_uint64 c_shape, c_loc;

      RealRank = 0;
      mtplyr = 1;

      for (cytnx_int64 i = this->_shape.size() - 1; i >= 0; i--) {
        if (locator[i] >= this->_shape[i]) {
          cytnx_error_msg(true, "%s", "Attempting to access out-of-bound index in Tensor.");
        }
        // c_shape[i] = this->_shape[this->_invmapper[i]];
        // c_loc[i] = locator[this->_invmapper[i]];
        c_shape = this->_shape[this->_invmapper[i]];
        c_loc = locator[this->_invmapper[i]];
        RealRank += mtplyr * c_loc;
        mtplyr *= c_shape;
      }
      return this->_storage.at<T>(RealRank);
    }

    const Scalar::Sproxy at(const std::vector<cytnx_uint64> &locator) const {
      cytnx_error_msg(locator.size() != this->_shape.size(), "%s",
                      "The input index does not match Tensor's rank.");

      cytnx_uint64 RealRank, mtplyr;
      // std::vector<cytnx_uint64> c_shape(this->_shape.size());
      // std::vector<cytnx_uint64> c_loc(this->_shape.size());

      cytnx_uint64 c_shape, c_loc;
      RealRank = 0;
      mtplyr = 1;

      for (cytnx_int64 i = this->_shape.size() - 1; i >= 0; i--) {
        if (locator[i] >= this->_shape[i]) {
          cytnx_error_msg(true, "%s", "Attempting to access out-of-bound index in Tensor.");
        }
        // c_shape[i] = this->_shape[this->_invmapper[i]];
        // c_loc[i] = locator[this->_invmapper[i]];
        c_shape = this->_shape[this->_invmapper[i]];
        c_loc = locator[this->_invmapper[i]];
        RealRank += mtplyr * c_loc;
        mtplyr *= c_shape;
      }
      return this->_storage.at(RealRank);
    }

    Scalar::Sproxy at(const std::vector<cytnx_uint64> &locator) {
      cytnx_error_msg(locator.size() != this->_shape.size(), "%s",
                      "The input index does not match Tensor's rank.");

      cytnx_uint64 RealRank, mtplyr;
      // std::vector<cytnx_uint64> c_shape(this->_shape.size());
      // std::vector<cytnx_uint64> c_loc(this->_shape.size());
      cytnx_uint64 c_shape, c_loc;

      RealRank = 0;
      mtplyr = 1;

      for (cytnx_int64 i = this->_shape.size() - 1; i >= 0; i--) {
        if (locator[i] >= this->_shape[i]) {
          cytnx_error_msg(true, "%s", "Attempting to access out-of-bound index in Tensor.");
        }
        // c_shape[i] = this->_shape[this->_invmapper[i]];
        // c_loc[i] = locator[this->_invmapper[i]];
        c_shape = this->_shape[this->_invmapper[i]];
        c_loc = locator[this->_invmapper[i]];
        RealRank += mtplyr * c_loc;
        mtplyr *= c_shape;
      }
      return this->_storage.at(RealRank);
    }

    boost::intrusive_ptr<Tensor_impl> get(const std::vector<cytnx::Accessor> &accessors);
    boost::intrusive_ptr<Tensor_impl> get_deprecated(const std::vector<cytnx::Accessor> &accessors);
    void set(const std::vector<cytnx::Accessor> &accessors,
             const boost::intrusive_ptr<Tensor_impl> &rhs);

    template <class T>
    void set(const std::vector<cytnx::Accessor> &accessors, const T &rc);

    void set(const std::vector<cytnx::Accessor> &accessors, const Scalar::Sproxy &rc);

    template <class Tx>
    void fill(const Tx &val) {
      this->storage().fill(val);
    }

    boost::intrusive_ptr<Tensor_impl> contiguous() {
      // return new instance if act on non-contiguous tensor
      // return self if act on contiguous tensor
      if (this->_contiguous) {
        boost::intrusive_ptr<Tensor_impl> out(this);
        // out->_storage = this->_storage;
        return out;
      } else {
        boost::intrusive_ptr<Tensor_impl> out(new Tensor_impl());
        std::vector<cytnx_uint64> oldshape(this->_shape.size());
        for (cytnx_uint64 i = 0; i < this->_shape.size(); i++) {
          oldshape[i] = this->_shape[this->_invmapper[i]];
        }

        out->_storage._impl =
          this->_storage._impl->Move_memory(oldshape, this->_mapper, this->_invmapper);
        // this->_storage._impl->Move_memory_(oldshape, this->_mapper, this->_invmapper);
        // out->_storage._impl = this->_storage._impl;
        // std::cout << out->_storage << std::endl;
        out->_invmapper = vec_range(this->_invmapper.size());
        out->_mapper = out->_invmapper;
        out->_shape = this->_shape;
        out->_contiguous = true;
        return out;
      }
    }

    void contiguous_() {
      // return new instance if act on non-contiguous tensor
      // return self if act on contiguous tensor
      if (!this->_contiguous) {
        std::vector<cytnx_uint64> oldshape(this->_shape.size());
        for (cytnx_uint64 i = 0; i < this->_shape.size(); i++) {
          oldshape[i] = this->_shape[this->_invmapper[i]];
        }

        this->_storage._impl =
          this->_storage._impl->Move_memory(oldshape, this->_mapper, this->_invmapper);
        // this->_storage._impl->Move_memory_(oldshape, this->_mapper, this->_invmapper);
        // this->_mapper = vec_range(this->_invmapper.size());
        vec_range_(this->_mapper, this->invmapper().size());
        this->_invmapper = this->_mapper;
        this->_contiguous = true;
      }
    }

    void reshape_(const std::vector<cytnx_int64> &new_shape) {
      if (!this->_contiguous) {
        this->contiguous_();
      }
      // std::vector<cytnx_uint64> result_shape(new_shape.size());
      cytnx_uint64 new_N = 1;
      bool has_undetermine = false;
      unsigned int Udet_id = 0;
      // this->_shape = vec_cast<cytnx_int64,cytnx_uint64>(new_shape);
      this->_shape.resize(new_shape.size());
      for (cytnx_uint64 i = 0; i < new_shape.size(); i++) {
        this->_shape[i] = new_shape[i];
      }
      for (int i = 0; i < new_shape.size(); i++) {
        if (new_shape[i] < 0) {
          if (new_shape[i] != -1)
            cytnx_error_msg(
              new_shape[i] != -1, "%s",
              "[ERROR] reshape can only have dimension > 0 and one undetermine rank specify as -1");
          if (has_undetermine)
            cytnx_error_msg(
              new_shape[i] != -1, "%s",
              "[ERROR] reshape can only have dimension > 0 and one undetermine rank specify as -1");
          Udet_id = i;
          has_undetermine = true;
        } else {
          new_N *= new_shape[i];
          // result_shape[i] = new_shape[i];
        }
      }

      if (has_undetermine) {
        cytnx_error_msg(new_N > this->_storage.size(), "%s",
                        "[ERROR] new shape exceed the total number of elements.");
        cytnx_error_msg(this->_storage.size() % new_N, "%s",
                        "[ERROR] unmatch size when reshape with undetermine dimension");
        // result_shape[Udet_id] = this->_storage.size() / new_N;
        this->_shape[Udet_id] = this->_storage.size() / new_N;
      } else {
        cytnx_error_msg(new_N != this->_storage.size(), "%s",
                        "[ERROR] new shape does not match the number of elements.");
      }

      // this->_shape = result_shape;
      // this->_mapper = std::move(vec_range(new_shape.size()));
      this->_mapper.resize(new_shape.size());
      vec_range_(this->_mapper, new_shape.size());
      this->_invmapper = this->_mapper;
    }

    boost::intrusive_ptr<Tensor_impl> reshape(const std::vector<cytnx_int64> &new_shape) {
      boost::intrusive_ptr<Tensor_impl> out(new Tensor_impl());
      if (this->is_contiguous()) {
        out = this->_clone_meta_only();
        out->_storage = this->_storage;
      } else {
        out = this->contiguous();
      }
      // out = this->clone();

      out->reshape_(new_shape);
      return out;
    }

    boost::intrusive_ptr<Tensor_impl> astype(const int &new_type) {
      // boost::intrusive_ptr<Tensor_impl> out(new Tensor_impl());
      // out->_storage = this->_storage.astype(new_type);
      if (this->dtype() == new_type) {
        return this;
      } else {
        boost::intrusive_ptr<Tensor_impl> out = this->_clone_meta_only();
        out->_storage = this->_storage.astype(new_type);
        return out;
      }
    }
  };
  ///@endcond

}  // namespace cytnx

#endif
#endif
