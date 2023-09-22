#ifndef _H_Tensor_
#define _H_Tensor_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include "backend/Storage.hpp"
#include "Device.hpp"
#include "intrusive_ptr_base.hpp"
#include <iostream>
#include <fstream>
#include "utils/vec_range.hpp"
#include "utils/vec_cast.hpp"
#include "utils/dynamic_arg_resolver.hpp"
// #include "linalg.hpp"
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

  class Tensor;

  ///@cond
  // [Note] these are fwd from linalg.hpp
  template <class T>
  Tensor operator+(const Tensor &lhs, const T &rc);
  template <class T>
  Tensor operator-(const Tensor &lhs, const T &rhs);
  template <class T>
  Tensor operator*(const Tensor &lhs, const T &rhs);
  template <class T>
  Tensor operator/(const Tensor &lhs, const T &rhs);
  ///@endcond

  /// @brief an tensor (multi-dimensional array)
  class Tensor {
   private:
   public:
    /// @cond
    // this is a proxy class to allow get/set element using [] as python!
    struct Tproxy {
      boost::intrusive_ptr<Tensor_impl> _insimpl;
      std::vector<cytnx::Accessor> _accs;
      Tproxy(boost::intrusive_ptr<Tensor_impl> _ptr, const std::vector<cytnx::Accessor> &accs)
          : _insimpl(std::move(_ptr)), _accs(accs) {}

      // when used to set elems:
      const Tensor &operator=(const Tensor &rhs) {
        this->_insimpl->set(_accs, rhs._impl);
        return rhs;
      }

      template <class T>
      const T &operator=(const T &rc) {
        this->_insimpl->set(_accs, rc);
        return rc;
      }
      const Tproxy &operator=(const Tproxy &rc) {
        Tensor tmp = Tensor(rc);
        this->_insimpl->set(_accs, tmp._impl);
        return rc;
      }

      template <class T>
      Tensor operator+=(const T &rc) {
        Tensor self;
        self._impl = _insimpl->get(_accs);
        self += rc;
        _insimpl->set(_accs, self._impl);
        self._impl = this->_insimpl;
        return self;
      }
      Tensor operator+=(const Tproxy &rc);

      template <class T>
      Tensor operator-=(const T &rc) {
        Tensor self;
        self._impl = _insimpl->get(_accs);
        self -= rc;
        _insimpl->set(_accs, self._impl);
        self._impl = this->_insimpl;
        return self;
      }
      Tensor operator-=(const Tproxy &rc);

      template <class T>
      Tensor operator/=(const T &rc) {
        Tensor self;
        self._impl = _insimpl->get(_accs);
        self /= rc;
        _insimpl->set(_accs, self._impl);
        self._impl = this->_insimpl;
        return self;
      }
      Tensor operator/=(const Tproxy &rc);

      template <class T>
      Tensor operator*=(const T &rc) {
        Tensor self;
        self._impl = _insimpl->get(_accs);
        self *= rc;
        _insimpl->set(_accs, self._impl);
        self._impl = this->_insimpl;
        return self;
      }
      Tensor operator*=(const Tproxy &rc);

      // alias to resolve conflict with op ovld for rc=Tensor
      /*
      template<class T>
      Tensor _operatorADD(const T &rc) const{
          Tensor out;
          out._impl = _insimpl->get(_accs);
          return out.Add(rc);
      }
      */
      Tensor operator+(const cytnx_complex128 &rc) const;  //{return this->_operatorADD(rc);};
      Tensor operator+(const cytnx_complex64 &rc) const;  //{return this->_operatorADD(rc);};
      Tensor operator+(const cytnx_double &rc) const;  //{return this->_operatorADD(rc);};
      Tensor operator+(const cytnx_float &rc) const;  //{return this->_operatorADD(rc);};
      Tensor operator+(const cytnx_uint64 &rc) const;  //{return this->_operatorADD(rc);};
      Tensor operator+(const cytnx_int64 &rc) const;  //{return this->_operatorADD(rc);};
      Tensor operator+(const cytnx_uint32 &rc) const;  //{return this->_operatorADD(rc);};
      Tensor operator+(const cytnx_int32 &rc) const;  //{return this->_operatorADD(rc);};
      Tensor operator+(const cytnx_uint16 &rc) const;  //{return this->_operatorADD(rc);};
      Tensor operator+(const cytnx_int16 &rc) const;  //{return this->_operatorADD(rc);};
      Tensor operator+(const cytnx_bool &rc) const;  //{return this->_operatorADD(rc);};
      Tensor operator+(const Tproxy &rc) const;

      /*
      template<class T>
      Tensor _operatorSUB(const T &rc) const{
          Tensor out;
          out._impl = _insimpl->get(_accs);
          return out.Sub(rc);
      }
      */
      Tensor operator-(const cytnx_complex128 &rc) const;  //{return this->_operatorSUB(rc);};
      Tensor operator-(const cytnx_complex64 &rc) const;  //{return this->_operatorSUB(rc);};
      Tensor operator-(const cytnx_double &rc) const;  //{return this->_operatorSUB(rc);};
      Tensor operator-(const cytnx_float &rc) const;  //{return this->_operatorSUB(rc);};
      Tensor operator-(const cytnx_uint64 &rc) const;  //{return this->_operatorSUB(rc);};
      Tensor operator-(const cytnx_int64 &rc) const;  //{return this->_operatorSUB(rc);};
      Tensor operator-(const cytnx_uint32 &rc) const;  //{return this->_operatorSUB(rc);};
      Tensor operator-(const cytnx_int32 &rc) const;  //{return this->_operatorSUB(rc);};
      Tensor operator-(const cytnx_uint16 &rc) const;  //{return this->_operatorSUB(rc);};
      Tensor operator-(const cytnx_int16 &rc) const;  //{return this->_operatorSUB(rc);};
      Tensor operator-(const cytnx_bool &rc) const;  //{return this->_operatorSUB(rc);};
      Tensor operator-(const Tproxy &rc) const;

      Tensor operator-() const;

      /*
      template<class T>
      Tensor _operatorMUL(const T &rc) const{
          Tensor out;
          out._impl = _insimpl->get(_accs);
          return out.Mul(rc);
      }
      */
      Tensor operator*(const cytnx_complex128 &rc) const;  //{return this->_operatorMUL(rc);};
      Tensor operator*(const cytnx_complex64 &rc) const;  //{return this->_operatorMUL(rc);};
      Tensor operator*(const cytnx_double &rc) const;  //{return this->_operatorMUL(rc);};
      Tensor operator*(const cytnx_float &rc) const;  //{return this->_operatorMUL(rc);};
      Tensor operator*(const cytnx_uint64 &rc) const;  //{return this->_operatorMUL(rc);};
      Tensor operator*(const cytnx_int64 &rc) const;  //{return this->_operatorMUL(rc);};
      Tensor operator*(const cytnx_uint32 &rc) const;  //{return this->_operatorMUL(rc);};
      Tensor operator*(const cytnx_int32 &rc) const;  //{return this->_operatorMUL(rc);};
      Tensor operator*(const cytnx_uint16 &rc) const;  //{return this->_operatorMUL(rc);};
      Tensor operator*(const cytnx_int16 &rc) const;  //{return this->_operatorMUL(rc);};
      Tensor operator*(const cytnx_bool &rc) const;  //{return this->_operatorMUL(rc);};
      Tensor operator*(const Tproxy &rc) const;

      /*
      template<class T>
      Tensor _operatorDIV(const T &rc) const{
          Tensor out;
          out._impl = _insimpl->get(_accs);
          return out.Div(rc);
      }
      */
      Tensor operator/(const cytnx_complex128 &rc) const;  //{return this->_operatorDIV(rc);};
      Tensor operator/(const cytnx_complex64 &rc) const;  //{return this->_operatorDIV(rc);};
      Tensor operator/(const cytnx_double &rc) const;  //{return this->_operatorDIV(rc);};
      Tensor operator/(const cytnx_float &rc) const;  //{return this->_operatorDIV(rc);};
      Tensor operator/(const cytnx_uint64 &rc) const;  //{return this->_operatorDIV(rc);};
      Tensor operator/(const cytnx_int64 &rc) const;  //{return this->_operatorDIV(rc);};
      Tensor operator/(const cytnx_uint32 &rc) const;  //{return this->_operatorDIV(rc);};
      Tensor operator/(const cytnx_int32 &rc) const;  //{return this->_operatorDIV(rc);};
      Tensor operator/(const cytnx_uint16 &rc) const;  //{return this->_operatorDIV(rc);};
      Tensor operator/(const cytnx_int16 &rc) const;  //{return this->_operatorDIV(rc);};
      Tensor operator/(const cytnx_bool &rc) const;  //{return this->_operatorDIV(rc);};
      Tensor operator/(const Tproxy &rc) const;

      template <class T>
      T item() const {
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.item<T>();
      }

      Scalar::Sproxy item() const {
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.item();
      }

      // when used to get elems:
      operator Tensor() const {
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out;
      }

      Storage storage() const {
        Tensor out;
        out._impl = _insimpl->get(_accs);
        return out.storage();
      }

    };  // proxy class of Tensor.

    /// @endcond

    /// @cond
    // these two are using the python way!
    //----------------------------------------
    template <class... Ts>
    Tproxy operator()(const std::string &e1, const Ts &...elems) {
      // std::cout << e1 << std::endl;
      std::vector<cytnx::Accessor> tmp = Indices_resolver(e1, elems...);
      return (*this)[tmp];
    }
    template <class... Ts>
    Tproxy operator()(const cytnx_int64 &e1, const Ts &...elems) {
      // std::cout << e1<< std::endl;
      std::vector<cytnx::Accessor> tmp = Indices_resolver(e1, elems...);
      return (*this)[tmp];
    }
    template <class... Ts>
    Tproxy operator()(const cytnx::Accessor &e1, const Ts &...elems) {
      // std::cout << e1 << std::endl;
      std::vector<cytnx::Accessor> tmp = Indices_resolver(e1, elems...);
      return (*this)[tmp];
    }
    template <class... Ts>
    const Tproxy operator()(const std::string &e1, const Ts &...elems) const {
      // std::cout << e1 << std::endl;
      std::vector<cytnx::Accessor> tmp = Indices_resolver(e1, elems...);
      return (*this)[tmp];
    }
    template <class... Ts>
    const Tproxy operator()(const cytnx_int64 &e1, const Ts &...elems) const {
      std::vector<cytnx::Accessor> tmp = Indices_resolver(e1, elems...);
      return (*this)[tmp];
    }
    template <class... Ts>
    const Tproxy operator()(const cytnx::Accessor &e1, const Ts &...elems) const {
      std::vector<cytnx::Accessor> tmp = Indices_resolver(e1, elems...);
      return (*this)[tmp];
    }

    //-----------------------------------------

    Tproxy operator[](const std::initializer_list<cytnx::Accessor> &accs) {
      std::vector<cytnx::Accessor> tmp = accs;
      return (*this)[tmp];
    }
    Tproxy operator[](const std::vector<cytnx::Accessor> &accs) {
      return Tproxy(this->_impl, accs);
    }

    const Tproxy operator[](const std::vector<cytnx::Accessor> &accs) const {
      return Tproxy(this->_impl, accs);
    }
    const Tproxy operator[](const std::initializer_list<cytnx::Accessor> &accs) const {
      std::vector<cytnx::Accessor> tmp = accs;
      return (*this)[tmp];
    }

    Tproxy operator[](const std::initializer_list<cytnx_int64> &accs) {
      std::vector<cytnx_int64> tmp = accs;
      return (*this)[tmp];
    }
    Tproxy operator[](const std::vector<cytnx_int64> &accs) {
      std::vector<cytnx::Accessor> acc_in;
      for (int i = 0; i < accs.size(); i++) {
        acc_in.push_back(cytnx::Accessor(accs[i]));
      }
      return Tproxy(this->_impl, acc_in);
    }
    const Tproxy operator[](const std::initializer_list<cytnx_int64> &accs) const {
      std::vector<cytnx_int64> tmp = accs;
      return (*this)[tmp];
    }
    const Tproxy operator[](const std::vector<cytnx_uint64> &accs) const {
      std::vector<cytnx::Accessor> acc_in;
      for (int i = 0; i < accs.size(); i++) {
        acc_in.push_back(cytnx::Accessor(accs[i]));
      }
      return Tproxy(this->_impl, acc_in);
    }
    const Tproxy operator[](const std::vector<cytnx_int64> &accs) const {
      std::vector<cytnx::Accessor> acc_in;
      for (int i = 0; i < accs.size(); i++) {
        acc_in.push_back(cytnx::Accessor(accs[i]));
      }
      return Tproxy(this->_impl, acc_in);
    }
    ///@endcond
    //-------------------------------------------

    /// @cond
    void _Save(std::fstream &f) const;
    void _Load(std::fstream &f);

    /// @endcond
    /**
    @brief Save current Tensor to file
    @param[in] fname file name (without file extension)

    @details
        save the Tensor to file with file path specify with input param \p fname with postfix
    ".cytn"
    @see Load(const std::string &fname)
    */
    void Save(const std::string &fname) const;
    /**
     * @see Save(const std::string &fname) const
     */
    void Save(const char *fname) const;

    /**
     * @brief Save current Tensor to the binary file
     * @details This function will save the Tensor to the binary file with file
     *   name \p fname .
     * @param fname[in] the file name of the binary file.
     * @pre The file name @p fname must be valid.
     * @see cytnx::Tensor::Fromfile
     */
    void Tofile(const std::string &fname) const;

    /**
     * @see Tofile(const std::string &fname) const
     */
    void Tofile(const char *fname) const;

    /**
     * @see Tofile(const std::string &fname) const
     */
    void Tofile(std::fstream &f) const;

    /**
    @brief Load current Tensor from file
    @param fname[in] file name
    @details
        load the Storage from file with file path specify with input param 'fname'
    @pre the file must be a Tensor object which is saved by cytnx::Tensor::Save.
    */

    static Tensor Load(const std::string &fname);
    /**
     * @see Load(const std::string &fname)
     */
    static Tensor Load(const char *fname);

    /**
     * @brief Load current Tensor from the binary file
     * @details This function will load the Tensor from the binary file which is saved by
     *    cytnx::Tensor::Tofile. Given the file name \p fname , data type \p dtype and
     *    number of elements \p count, this function will load the first \p count elements
     *    from the binary file \p fname with data type \p dtype.
     * @param fname[in] the file name of the binary file.
     * @param dtype[in] the data type of the binary file. This can be any of the type defined in
     *   cytnx::Type.
     * @param count[in] the number of elements to be loaded from the binary file. If set to -1,
     *  all elements in the binary file will be loaded.
     * @return Tensor
     * @pre
     *  1. The @p dtype cannot be Type.Void.
     *  2. The @p dtype must be the same as the data type of the binary file.
     *  3. The @p Nelem cannot be 0.
     *  4. The @p Nelem cannot be larger than the number of elements in the binary file.
     *  5. The file name @p fname must be valid.
     * @see cytnx::Tensor::Tofile
     */
    static Tensor Fromfile(const std::string &fname, const unsigned int &dtype,
                           const cytnx_int64 &count = -1);
    static Tensor Fromfile(const char *fname, const unsigned int &dtype,
                           const cytnx_int64 &count = -1);

    // static Tensor Frombinary(const std::string &fname);

    ///@cond
    boost::intrusive_ptr<Tensor_impl> _impl;
    Tensor() : _impl(new Tensor_impl()){};
    Tensor(const Tensor &rhs) { _impl = rhs._impl; }

    /*
    template<class Tp>
    Tensor(const std::initializer_list<Tp> &rhs){
        Storage stmp = std::vector<Tp>(rhs);
        boost::intrusive_ptr<Tensor_impl> tmp(new Tensor_impl());
        tmp->Init(stmp);
        this->_impl = tmp;
    }
    */

    Tensor &operator=(const Tensor &rhs) {
      _impl = rhs._impl;
      return *this;
    }

    void operator=(const Tproxy &rhsp) {  // this is used to handle proxy assignment
      this->_impl = rhsp._insimpl->get(rhsp._accs);
    }
    ///@endcond

    //@{
    // default device==Device.cpu (-1)
    /**
    @brief initialize a Tensor
    @param[in] shape the shape of tensor.
    @param[in] dtype the dtype of tensor. This can be any of type defined in cytnx::Type
    @param[in] device the device that tensor to be created. This can be cytnx::Device.cpu or
    @param[in] init_zero if true, the content of Tensor will be initialized to zero. if false, the
    content of Tensor will be un-initialize.
    cytnx::Device.cuda+<gpuid>, see cytnx::Device for more detail.

    @note
        The content of Tensor created will be un-initialize! See \link cytnx::zeros
    zeros()\endlink, \link cytnx::ones ones() \endlink or \link cytnx::arange arange() \endlink for
    generating an Tensor.

    ## Example:
    ### c++ API:
    \include example/Tensor/Init.cpp
    #### output>
    \verbinclude example/Tensor/Init.cpp.out
    ### python API:
    \include example/Tensor/Init.py
    #### output>
    \verbinclude example/Tensor/Init.py.out
    */
    void Init(const std::vector<cytnx_uint64> &shape, const unsigned int &dtype = Type.Double,
              const int &device = -1, const bool &init_zero = true) {
      boost::intrusive_ptr<Tensor_impl> tmp(new Tensor_impl());
      this->_impl = tmp;
      this->_impl->Init(shape, dtype, device, init_zero);
    }
    // void Init(const Storage& storage) {
    //   boost::intrusive_ptr<Tensor_impl> tmp(new Tensor_impl());
    //   this->_impl = tmp;
    //   this->_impl->Init(storage);
    // }
    // void Init(const Storage& storage, const std::vector<cytnx_uint64> &shape,
    //   const unsigned int &dtype = Type.Double, const int &device = -1) {
    //   boost::intrusive_ptr<Tensor_impl> tmp(new Tensor_impl());
    //   this->_impl = tmp;
    //   this->_impl->Init(storage, shape, dtype, device);
    // }

    /**
     * @brief Construct a new Tensor object
     * @details This is the constructor of Tensor. It will call
     *     cytnx::Tensor::Init() to initialize the Tensor.
     * @param[in] shape the shape of tensor
     * @param[in] dtype the dtype of tensor. This can be any of type defined in cytnx::Type.
     * @param[in] device the device that tensor to be created. This can be cytnx::Device.cpu or
     *    cytnx::Device.cuda+<gpuid>, see cytnx::Device for more detail.
     * @param[in] init_zero if true, the content of Tensor will be initialized to zero. If false,
     *   the content of Tensor will be un-initialized.
     * @see cytnx::Tensor::Init
     */
    Tensor(const std::vector<cytnx_uint64> &shape, const unsigned int &dtype = Type.Double,
           const int &device = -1, const bool &init_zero = 1)
        : _impl(new Tensor_impl()) {
      this->Init(shape, dtype, device, init_zero);
    }
    // Tensor(const Storage& storage)
    //     : _impl(new Tensor_impl()) {
    //   this->Init(storage);
    // }
    // Tensor(const Storage& storage, const std::vector<cytnx_uint64> &shape,
    //   const unsigned int &dtype = Type.Double, const int &device = -1)
    //     : _impl(new Tensor_impl()) {
    //   this->Init(storage, shape, dtype, device);
    // }
    //@}

    /**
    @brief Convert a Storage to Tensor
    @param[in] in the Storage to be converted
    @return [Tensor] a Tensor with the same dtype and device as the input Storage
    */
    static Tensor from_storage(const Storage &in) {
      Tensor out;
      boost::intrusive_ptr<Tensor_impl> tmp(new Tensor_impl());
      out._impl = tmp;
      out._impl->Init(in);
      return out;
    }

    /**
    @brief the dtype-id of the Tensor
    @see cytnx::Type
    @return [unsigned int] the dtype_id of the Tensor
    */
    unsigned int dtype() const { return this->_impl->dtype(); }

    /**
    @brief the device-id of the Tensor
    @see cytnx::Device
    @return [int] the device_id of the Tensor
    */
    int device() const { return this->_impl->device(); }

    /**
    @brief the dtype (in string) of the Tensor
    @see cytnx::Type, dtype() const
    @return [std::string] the dtype of the Tensor
    */
    std::string dtype_str() const { return this->_impl->dtype_str(); }

    /**
    @brief the device (in string) of the Tensor
    @see cytnx::Device, device() const
    @return [std::string] the device of the Tensor
    */
    std::string device_str() const { return this->_impl->device_str(); }

    /**
    @brief the shape of the Tensor
    @return [std::vector<cytnx_uint64>] the shape of the Tensor
    */
    const std::vector<cytnx_uint64> &shape() const { return this->_impl->shape(); }

    /**
        @brief the rank of the Tensor
        @return [cytnx_uint64] the rank of the Tensor
    */
    cytnx_uint64 rank() const { return this->_impl->shape().size(); }

    /**
    @brief return a clone of the current Tensor.
    @return [Tensor]
    @details
        In C++ API, the behavior of assignment operator is designed to have same behavior as
    python,\n to have a copy of the current tensor, we call clone to return a copy.

    ## Example:
    ### c++ API:
    \include example/Tensor/clone.cpp
    #### output>
    \verbinclude example/Tensor/clone.cpp.out
    ### python API:
    \include example/Tensor/clone.py
    #### output>
    \verbinclude example/Tensor/clone.py.out
    */
    Tensor clone() const {
      Tensor out;
      out._impl = this->_impl->clone();
      return out;
    }

    /**
    @brief copy a tensor to new device
    @param[in] device the device-id that is moving to. it can be any device defined in cytnx::Device
    @return [Tensor]

    description:\n
        if the device-id is the same as current Tensor's device, then return self.\n
        otherwise, return a copy of instance that located on the target device.   \n
        see also: \link cytnx::Tensor::to_ Tensor.to_ \endlink                    \n

    ## Example:
    ### c++ API:
    \include example/Tensor/to.cpp
    #### output>
    \verbinclude example/Tensor/to.cpp.out
    ### python API:
    \include example/Tensor/to.py
    #### output>
    \verbinclude example/Tensor/to.py.out
    */
    Tensor to(const int &device) const {
      Tensor out;
      out._impl = this->_impl->to(device);
      return out;
    }

    /**
    @brief move the current Tensor to the device.
    @param[in] device the device-id that is moving to. it can be any device defined in cytnx::Device

    description:\n
        see also: \link cytnx::Tensor::to Tensor.to \endlink\n

    ## Example:
    ### c++ API:
    \include example/Tensor/to_.cpp
    #### output>
    \verbinclude example/Tensor/to_.cpp.out
    ### python API:
    \include example/Tensor/to_.py
    #### output>
    \verbinclude example/Tensor/to_.py.out
    */
    void to_(const int &device) { this->_impl->to_(device); }

    /**
    @brief return whether the Tensor is contiguous or not.
    @return [bool] true if the Tensor is contiguous, false otherwise.
    */
    const bool &is_contiguous() const { return this->_impl->is_contiguous(); }

    Tensor permute_(const std::vector<cytnx_uint64> &rnks) {
      this->_impl->permute_(rnks);
      return *this;
    }
    /// @cond
    template <class... Ts>
    Tensor permute_(const cytnx_uint64 &e1, const Ts &...elems) {
      std::vector<cytnx_uint64> argv = dynamic_arg_uint64_resolver(e1, elems...);
      this->_impl->permute_(argv);
      return *this;
    }
    /// @endcond

    /**
    @brief perform tensor permute on the cytnx::Tensor and return a new instance.
    @param[in] rnks the permute indices, should have No. of elements equal to the rank of tensor.
    @return [Tensor] a permuted new Tensor
    @pre
        1. The size of input and output Tensor should be the same.
        2. \p rnks cannot contain duplicated elements.

    ## Example:
    ### c++ API:
    \include example/Tensor/permute.cpp
    #### output>
    \verbinclude example/Tensor/permute.cpp.out
    ### python API:
    \include example/Tensor/permute.py
    #### output>
    \verbinclude example/Tensor/permute.py.out
    */
    Tensor permute(const std::vector<cytnx_uint64> &rnks) const {
      Tensor out;
      out._impl = this->_impl->permute(rnks);
      return out;
    }
    /// @cond
    template <class... Ts>
    Tensor permute(const cytnx_uint64 &e1, const Ts &...elems) const {
      std::vector<cytnx_uint64> argv = dynamic_arg_uint64_resolver(e1, elems...);
      return this->permute(argv);
    }
    /// @endcond

    /**
    @brief Make the Tensor contiguous by coalescing the memory (storage).
    @return [Tensor] a new Tensor that is with contiguous memory (storage).
    @see \link Tensor::contiguous_ Tensor::contiguous_() \endlink

    ## Example:
    ### c++ API:
    \include example/Tensor/contiguous.cpp
    #### output>
    \verbinclude example/Tensor/contiguous.cpp.out
    ### python API:
    \include example/Tensor/contiguous.py
    #### output>
    \verbinclude example/Tensor/contiguous.py.out
    */
    Tensor contiguous() const {
      Tensor out;
      out._impl = this->_impl->contiguous();
      return out;
    }

    /**
    @brief Make the Tensor contiguous by coalescing the memory (storage), inplacely
    @see \link Tensor::contiguous Tensor::contiguous() \endlink

    ## Example:
    ### c++ API:
    \include example/Tensor/contiguous_.cpp
    #### output>
    \verbinclude example/Tensor/contiguous_.cpp.out
    ### python API:
    \include example/Tensor/contiguous_.py
    #### output>
    \verbinclude example/Tensor/contiguous_.py.out
    */
    Tensor contiguous_() {
      this->_impl->contiguous_();
      return *this;
    }

    /**
    @brief reshape the Tensor, inplacely
    @param[in] new_shape the new shape of the Tensor.
    @pre
        1. The size of input and output Tensor should be the same.
        2. \p new_shape cannot be empty.
    @see \link Tensor::reshape Tensor::reshape() \endlink
    @note
        Compare to reshape(), this function will not create a new Tensor,
          but reshape the current Tensor inplacely.

    ## Example:
    ### c++ API:
    \include example/Tensor/reshape_.cpp
    #### output>
    \verbinclude example/Tensor/reshape_.cpp.out
    ### python API:
    \include example/Tensor/reshape_.py
    #### output>
    \verbinclude example/Tensor/reshape_.py.out
    */
    void reshape_(const std::vector<cytnx_int64> &new_shape) { this->_impl->reshape_(new_shape); }
    /// @cond
    void reshape_(const std::vector<cytnx_uint64> &new_shape) {
      std::vector<cytnx_int64> shape(new_shape.begin(), new_shape.end());
      this->_impl->reshape_(shape);
    }
    void reshape_(const std::initializer_list<cytnx_int64> &new_shape) {
      std::vector<cytnx_int64> shape = new_shape;
      this->_impl->reshape_(shape);
    }
    template <class... Ts>
    void reshape_(const cytnx_int64 &e1, const Ts... elems) {
      std::vector<cytnx_int64> shape = dynamic_arg_int64_resolver(e1, elems...);
      // std::cout << shape << std::endl;
      this->_impl->reshape_(shape);
    }
    /// @endcond

    /**
    @brief return a new Tensor that is reshaped.
    @param[in] new_shape the new shape of the Tensor.
    @return [Tensor]
    @pre
        1. The size of input and output Tensor should be the same.
        2. \p new_shape cannot be empty.
    @note
        1. This function will not change the original Tensor.
        2. You can use Tensor::reshape_() to reshape the Tensor inplacely.
        3. You can set \p new_shape to -1, which will be automatically determined
          by the size of the Tensor. The behavior is the same as numpy.reshape().
    @see \link Tensor::reshape_ Tensor::reshape_() \endlink

    ## Example:
    ### c++ API:
    \include example/Tensor/reshape.cpp
    #### output>
    \verbinclude example/Tensor/reshape.cpp.out
    ### python API:
    \include example/Tensor/reshape.py
    #### output>
    \verbinclude example/Tensor/reshape.py.out
    */
    Tensor reshape(const std::vector<cytnx_int64> &new_shape) const {
      Tensor out;
      out._impl = this->_impl->reshape(new_shape);
      return out;
    }

    /**
     * @see reshape(const std::vector<cytnx_int64> &new_shape) const
     */
    Tensor reshape(const std::vector<cytnx_uint64> &new_shape) const {
      std::vector<cytnx_int64> tmp(new_shape.size());
      memcpy(&tmp[0], &new_shape[0], sizeof(cytnx_uint64) * new_shape.size());
      Tensor out;
      out._impl = this->_impl->reshape(tmp);
      return out;
    }

    /**
     * @see reshape(const std::vector<cytnx_int64> &new_shape) const
     */
    Tensor reshape(const std::initializer_list<cytnx_int64> &new_shape) const {
      return this->reshape(std::vector<cytnx_int64>(new_shape));
    }

    /// @cond
    template <class... Ts>
    Tensor reshape(const cytnx_int64 &e1, const Ts &...elems) const {
      std::vector<cytnx_int64> argv = dynamic_arg_int64_resolver(e1, elems...);
      return this->reshape(argv);
    }
    /// @endcond

    /**
    @brief return a new Tensor that cast to different dtype.
    @param[in] new_type the new dtype. It can be any type defined in cytnx::Type
    @return [Tensor]
    @note
        If the new_type is the same as dtype of the current Tensor, return self.
    @attention
        This function cannot convert complex type to real type, please use
        Tensor::real() or Tensor::imag() to get the real or imaginary part of
        the complex Tensor instead.

    ## Example:
    ### c++ API:
    \include example/Tensor/astype.cpp
    #### output>
    \verbinclude example/Tensor/astype.cpp.out
    ### python API:
    \include example/Tensor/astype.py
    #### output>
    \verbinclude example/Tensor/astype.py.out
    */
    Tensor astype(const int &new_type) const {
      Tensor out;
      out._impl = this->_impl->astype(new_type);
      return out;
    }

    // Tensor diagonal(){
    //     for(unsigned int i=0;i<this->shape().size();i++){
    //         if(this->shape()[i] != this->shape()[0],"[ERROR] Tensor.diagonal() can only be called
    //         when the subject has equal dimension in each rank.%s","\n");
    //     }
    //
    // }

    /**
    @brief Get an element at specific location.
    @details This function is used to get an element at specific location. If the template type is
    not given, the return will be a Scalar.
    @param[in] locator the location of the element
    @return [ref]

    @note
        1. This is for C++ API only!
        2. need template instantiation to resolve the type, which should be consist with
          the dtype of the Tensor. An error will be issued if the template type is inconsist
          with the current dtype of Tensor.
        3. For python API, use [] directly to get element.

    ## Example:
    ### c++ API:
    \include example/Tensor/at.cpp
    #### output>
    \verbinclude example/Tensor/at.cpp.out
    */
    template <class T>
    T &at(const std::vector<cytnx_uint64> &locator) {
      return this->_impl->at<T>(locator);
    }

    /**
     * @see at(const std::vector<cytnx_uint64> &locator)
     */
    template <class T>
    const T &at(const std::vector<cytnx_uint64> &locator) const {
      return this->_impl->at<T>(locator);
    }
    /// @cond
    template <class T, class... Ts>
    const T &at(const cytnx_uint64 &e1, const Ts &...elems) const {
      std::vector<cytnx_uint64> argv = dynamic_arg_uint64_resolver(e1, elems...);
      return this->at<T>(argv);
    }
    template <class T, class... Ts>
    T &at(const cytnx_uint64 &e1, const Ts &...elems) {
      std::vector<cytnx_uint64> argv = dynamic_arg_uint64_resolver(e1, elems...);
      return this->at<T>(argv);
    }

    const Scalar::Sproxy at(const std::vector<cytnx_uint64> &locator) const {
      return this->_impl->at(locator);
    }

    Scalar::Sproxy at(const std::vector<cytnx_uint64> &locator) { return this->_impl->at(locator); }
    /// @endcond

    /**
    @brief get the element from a rank-0 Tensor.
    @details This function is used to get the element from a rank-0 Tensor. If the template type is
    not given, the return will be a Scalar.
    @return [T]

    @note
        1. This can only be called on a rank-0 Tensor (scalar). For C++ API, a template
    instantiation of type is needed to resolve the type, which should be connsist with the dtype of
    the Tensor. An error will be issued if the template type if inconsist with the current dtype of
    Tensor.
        2. Although the return is by reference in C++ part, the return in python is not.
        3. From 2., We recommend user to use at<T> (C++ API) and [] (python API) to modify the value
    of the element to have consistant syntax across two languages.

    ## Example:
    ### c++ API:
    \include example/Tensor/item.cpp
    #### output>
    \verbinclude example/Tensor/item.cpp.out
    ### python API:
    \include example/Tensor/item.py
    #### output>
    \verbinclude example/Tensor/item.py.out
    */
    template <class T>
    T &item() {
      cytnx_error_msg(this->_impl->storage().size() != 1, "[ERROR][Tensor.item<T>]%s",
                      "item can only be called from a Tensor with only one element\n");
      return this->_impl->storage().at<T>(0);
    }

    ///@cond
    template <class T>
    const T &item() const {
      cytnx_error_msg(this->_impl->storage().size() != 1, "[ERROR][Tensor.item<T>]%s",
                      "item can only be called from a Tensor with only one element\n");
      return this->_impl->storage().at<T>(0);
    }

    const Scalar::Sproxy item() const {
      Scalar::Sproxy out(this->storage()._impl, 0);
      return out;
    }

    Scalar::Sproxy item() {
      Scalar::Sproxy out(this->storage()._impl, 0);
      return out;
    }

    ///@endcond

    /**
    @brief get elements using Accessor (C++ API) / slices (python API)
    @param[in] accessors the Accessor (C++ API) / slices (python API) to get the elements.
    @return [Tensor]
    @see \link cytnx::Accessor Accessor\endlink for cordinate with Accessor in C++ API.
    @note
        1. the return will be a new Tensor instance, which not share memory with the current Tensor.

    ## Equivalently:
        One can also using more intruisive way to get the slice using [] operator.

    ## Example:
    ### c++ API:
    \include example/Tensor/get.cpp
    #### output>
    \verbinclude example/Tensor/get.cpp.out
    ### python API:
    \include example/Tensor/get.py
    #### output>
    \verbinclude example/Tensor/get.py.out
    */
    Tensor get(const std::vector<cytnx::Accessor> &accessors) const {
      Tensor out;
      out._impl = this->_impl->get(accessors);
      return out;
    }

    /*
    Tensor get_v2(const std::vector<cytnx::Accessor> &accessors) const{
        Tensor out;
        out._impl = this->_impl->get_v2(accessors);
        return out;
    }
    */

    /**
    @brief set elements with the input Tensor using Accessor (C++ API) / slices (python API)
    @param[in] accessors the list(vector) of accessors.
    @param rhs [Tensor]
    @note:
        the shape of the input Tensor should be the same as the shape that indicated using Accessor.
    The memory is not shared with the input Tensor.

    ## Example:
    ### c++ API:
    \include example/Tensor/set.cpp
    #### output>
    \verbinclude example/Tensor/set.cpp.out
    ### python API:
    \include example/Tensor/set.py
    #### output>
    \verbinclude example/Tensor/set.py.out
    */
    void set(const std::vector<cytnx::Accessor> &accessors, const Tensor &rhs) {
      this->_impl->set(accessors, rhs._impl);
    }

    /**
    @brief set elements with the input constant using Accessor (C++ API) / slices (python API)
    @param[in] accessors the list(vector) of accessors.
    @param rc [Const]

    @see \link cytnx::Tensor::fill Tensor::fill \endlink for filling all elements with assigned
    constant.

    ## Example:
    ### c++ API:
    \include example/Tensor/set.cpp
    #### output>
    \verbinclude example/Tensor/set.cpp.out
    ### python API:
    \include example/Tensor/set.py
    #### output>
    \verbinclude example/Tensor/set.py.out
    */
    template <class T>
    void set(const std::vector<cytnx::Accessor> &accessors, const T &rc) {
      this->_impl->set(accessors, rc);
    }
    ///@cond
    template <class T>
    void set(const std::initializer_list<cytnx::Accessor> &accessors, const T &rc) {
      std::vector<cytnx::Accessor> args = accessors;
      this->set(args, rc);
    }
    ///@endcond

    /**
    @brief return the storage of current Tensor.
    @return [Storage]

    @note
      The return storage shares the same instance of the storage of current Tensor. Use
      Storage.clone() to create a new instance of the returned Storage.

    */
    Storage &storage() const { return this->_impl->storage(); }

    /**
    @brief fill all the element of current Tensor with the value.
    @param[in] val the assigned value

    ## Example:
    ### c++ API:
    \include example/Tensor/fill.cpp
    #### output>
    \verbinclude example/Tensor/fill.cpp.out
    ### python API
    \include example/Tensor/fill.py
    #### output>
    \verbinclude example/Tensor/fill.py.out
    */
    template <class T>
    void fill(const T &val) {
      this->_impl->fill(val);
    }

    /**
     * @brief compare the shape of two tensors.
     * @param[in] rhs the tensor to be compared.
     */
    bool equivshape(const Tensor &rhs) {
      if (this->shape() != rhs.shape()) return false;
      return true;
    }

    /**
     * @brief return the real part of the tensor.
     * @return [Tensor] the real part of the tensor.
     * @pre the tensor must be complex type (Type.ComplexDouble or
     *     Type.ComplexFloat).
     * @see cytnx::Type
     */
    Tensor real();

    /**
     * @brief return the imaginary part of the tensor.
     * @return [Tensor] the imaginary part of the tensor.
     * @pre the tensor must be complex type (Type.ComplexDouble or
     *    Type.ComplexFloat).
     * @see cytnx::Type
     */
    Tensor imag();

    // Arithmic:
    /**
     * @brief addition assignment operator with a Tensor or a scalar.
     * @details This function will add the template type to the current tensor, inplacely.
     *     The template can be either a scalar or a tensor. If the template is a
     *     scalar, then the scalar will be added to all the elements of the
     *     current tensor. If the template is a tensor, then the shape of the
     *     template tensor must be the same as the current tensor. The supported
     *     type of the template are Tensor, Scalar or any scalar type (see
     *     \ref cytnx_complex128, \ref cytnx_complex64, \ref cytnx_double, \ref cytnx_float,
     *     \ref cytnx_int64, \ref cytnx_int32, \ref cytnx_int16,
     *     \ref cytnx_uint64, \ref cytnx_uint32, \ref cytnx_uint16, \ref cytnx_bool).
     * @param[in] rc the added Tensor or scalar.
     * @pre
     *     If the template type is Tensor, then the shape of the template tensor
     *       must be the same as the current tensor.
     */
    template <class T>
    Tensor &operator+=(const T &rc);

    /**
     * @brief subtraction assignment operator with a Tensor or a scalar.
     * @details This function will subtract the template type to the current tensor, inplacely.
     *    The template can be either a scalar or a tensor. If the template is a
     *    scalar, then the scalar will be subtracted to all the elements of the
     *    current tensor. If the template is a tensor, then the shape of the
     *    template tensor must be the same as the current tensor. The supported
     *    type of the template are Tensor, Scalar or any scalar type (see
     *    \ref cytnx_complex128, \ref cytnx_complex64, \ref cytnx_double, \ref cytnx_float,
     *    \ref cytnx_int64, \ref cytnx_int32, \ref cytnx_int16,
     *    \ref cytnx_uint64, \ref cytnx_uint32, \ref cytnx_uint16, \ref cytnx_bool).
     * @param[in] rc the subtracted Tensor or scalar.
     * @pre
     *   If the template type is Tensor, then the shape of the template tensor
     *     must be the same as the current tensor.
     */
    template <class T>
    Tensor &operator-=(const T &rc);

    /**
     * @brief multiplication assignment operator with a Tensor or a scalar.
     * @details This function will multiply the template type to the current tensor, inplacely.
     * The template can be either a scalar or a tensor. If the template is a
     * scalar, then the scalar will be multiplied to all the elements of the
     * current tensor. If the template is a tensor, then the shape of the
     * template tensor must be the same as the current tensor. The supported
     * type of the template are Tensor, Scalar or any scalar type (see
     * \ref cytnx_complex128, \ref cytnx_complex64, \ref cytnx_double, \ref cytnx_float,
     * \ref cytnx_int64, \ref cytnx_int32, \ref cytnx_int16,
     * \ref cytnx_uint64, \ref cytnx_uint32, \ref cytnx_uint16, \ref cytnx_bool).
     * @param[in] rc the multiplied Tensor or scalar.
     * @pre
     *  If the template type is Tensor, then the shape of the template tensor
     *   must be the same as the current tensor.
     */
    template <class T>
    Tensor &operator*=(const T &rc);

    /**
     * @brief division assignment operator with a Tensor or a scalar.
     * @details This function will divide the template type to the current tensor, inplacely.
     * The template can be either a scalar or a tensor. If the template is a
     * scalar, then the scalar will be divided to all the elements of the
     * current tensor. If the template is a tensor, then the shape of the
     * template tensor must be the same as the current tensor. The supported
     * type of the template are Tensor, Scalar or any scalar type (see
     * \ref cytnx_complex128, \ref cytnx_complex64, \ref cytnx_double, \ref cytnx_float,
     * \ref cytnx_int64, \ref cytnx_int32, \ref cytnx_int16,
     * \ref cytnx_uint64, \ref cytnx_uint32, \ref cytnx_uint16, \ref cytnx_bool).
     * @param[in] rc the divided Tensor or scalar.
     * @pre
     * 1. If the template type is Tensor, then the shape of the template tensor
     *  must be the same as the current tensor.
     * 2. \p rc cannot be zero.
     */
    template <class T>
    Tensor &operator/=(const T &rc);

    // Tensor &operator+=(const Tproxy &rc);
    // Tensor &operator-=(const Tproxy &rc);
    // Tensor &operator*=(const Tproxy &rc);
    // Tensor &operator/=(const Tproxy &rc);
    /*
    Tensor operator+(const Tproxy &rc){
         return *this + Tensor(rc);
    }
    Tensor operator-(const Tproxy &rc){
         return *this - Tensor(rc);
    }
    Tensor operator*(const Tproxy &rc){
         return *this * Tensor(rc);
    }
    Tensor operator/(const Tproxy &rc){
         return *this / Tensor(rc);
    }
    */
    /**
     * @brief Addition function with a Tensor or a scalar. Same as
     * cytnx::operator+(const Tensor &self, const T &rhs).
     * @param[in] rhs the added Tensor or scalar.
     */
    template <class T>
    Tensor Add(const T &rhs) {
      return *this + rhs;
    }

    /**
     * @brief Addition function with a Tensor or a scalar, inplacely.
     * Same as operator+=(const T &rhs).
     * @param[in] rhs the added Tensor or scalar.
     */
    template <class T>
    Tensor &Add_(const T &rhs) {
      return *this += rhs;
    }

    /**
     * @brief Subtraction function with a Tensor or a scalar. Same as
     * cytnx::operator-(const Tensor &self, const T &rhs).
     * @param[in] rhs the subtracted Tensor or scalar.
     */
    template <class T>
    Tensor Sub(const T &rhs) {
      return *this - rhs;
    }

    /**
     * @brief Subtraction function with a Tensor or a scalar, inplacely.
     * Same as operator-=(const T &rhs).
     * @param[in] rhs the subtracted Tensor or scalar.
     */
    template <class T>
    Tensor &Sub_(const T &rhs) {
      return *this -= rhs;
    }

    /**
     * @brief Multiplication function with a Tensor or a scalar. Same as
     * cytnx::operator*(const Tensor &self, const T &rhs).
     * @param[in] rhs the multiplied Tensor or scalar.
     */
    template <class T>
    Tensor Mul(const T &rhs) {
      return *this * rhs;
    }

    /**
     * @brief Multiplication function with a Tensor or a scalar, inplacely.
     * Same as operator*=(const T &rhs).
     * @param[in] rhs the multiplied Tensor or scalar.
     */
    template <class T>
    Tensor &Mul_(const T &rhs) {
      return *this *= rhs;
    }

    /**
     * @brief Division function with a Tensor or a scalar. Same as
     * cytnx::operator/(const Tensor &self, const T &rhs).
     * @param[in] rhs the divided Tensor or scalar.
     * @attension \p rhs cannot be zero.
     */
    template <class T>
    Tensor Div(const T &rhs) {
      return *this / rhs;
    }

    /**
     * @brief Division function with a Tensor or a scalar, inplacely.
     * Same as operator/=(const T &rhs).
     * @param[in] rhs the divided Tensor or scalar.
     * @attension \p rhs cannot be zero.
     */
    template <class T>
    Tensor &Div_(const T &rhs) {
      return *this /= rhs;
    }

    /**
     * @brief The comparison function.
     * @details This function is the comparison function. Same as
     * cytnx::operator==(const Tensor &self, const T &rhs).
     * @param[in] rhs the compared object.
     */
    template <class T>
    Tensor Cpr(const T &rhs) {
      return *this == rhs;
    }

    // /**
    //  * @brief Compare each element of the current tensor with the input tensor.
    //  * @details This function Compare each element of the current tensor with the input tensor.
    //  * @param[in] rhs the compared tensor.
    //  */
    // bool approx_eq(const Tensor &rhs, const cytnx_double tol = 0) {
    //   if (this->device() != rhs.device()) {
    //     if (User_debug)
    //       std::cout << "[approx_eq] Tensor device " << this->device()
    //                 << "not equal to rhs tensor device " << rhs.device() << std::endl;
    //     return false;
    //   }
    //   // if (this->dtype() != rhs.dtype()) {
    //   //   std::cout << "[approx_eq] Tensor dtype " << this->dtype()
    //   //             << "not equal to rhs tensor dtype " << rhs.dtype() << std::endl;
    //   //   return false;
    //   // }
    //   if (this->shape() != rhs.shape()) {
    //     if (User_debug)
    //       std::cout << "[approx_eq] Tensor shape " << this->shape()
    //                 << "not equal to rhs tensor shape " << rhs.shape() << std::endl;
    //     return false;
    //   }
    //   if (this->is_contiguous() != rhs.is_contiguous()) {
    //     if (User_debug)
    //       std::cout << "[AreNearlyEqTensor] Tensor contiguous flag " << this->is_contiguous()
    //                 << "not equal to rhs tensor flag " << rhs.is_contiguous() << std::endl;
    //     return false;
    //   }
    //   return this->_impl->_storage.approx_eq(rhs._impl->_storage._impl, tol);
    // }

    // template<class T>
    // Tensor& Cpr_(const T &rhs){
    //
    //      return *this == rhs;
    // }

    template <class T>
    Tensor Mod(const T &rhs) {
      return *this % rhs;
    }

    /**
     * @brief The negation function.
     * @details This function is the negation function. Namely, if the current
     * tensor is \f$A\f$, then the output tensor is \f$-A\f$.
     * @return The negation of the current tensor.
     */
    Tensor operator-() { return this->Mul(-1.); }

    /**
     * @brief The flatten function.
     * @details This function is the flatten function. It will clone (deep copy)
     * , contiguos the current tensor and reshape it to 1-rank Tensor.
     * @note compare to the flatten_() function, this function will return a new
     * tensor and the current tensor will not be changed.
     */
    Tensor flatten() const {
      Tensor out = this->clone();
      out.contiguous_();
      out.reshape_({-1});
      return out;
    }

    /**
     * @brief The flatten function, inplacely.
     * @details This function is the flatten function, inplacely. It will
     * contiguos the current tensor and reshape it to 1-rank Tensor.
     * @note compare to the flatten() function, this is an inplacely function,
     * the current tensor will be changed.
     */
    void flatten_() {
      this->contiguous_();
      this->reshape_({-1});
    }

    /**
     * @brief the append function.
     * @details This function is the append function. It will append the \p rhs
     * tensor to the current tensor. The \p rhs tensor must have the same shape
     * as the current tensor, except the first dimension. For example, if the
     * current tensor is \f$A(i,j,k)\f$ and the \p rhs tensor is \f$B(j,k)\f$, then
     * the output tensor is \f$C(i,j,k)\f$ where
     * \f[
     * C(i,j,k) = \begin{cases}
     * A(i,j,k) & \text{if } i \neq N \\
     * B(j,k) & \text{if } i = N
     * \end{cases}
     * \f]
     * where \f$N\f$ is the number of the first dimension of the current tensor.
     * Here indices \f$i\f$, \f$j\f$ and \f$k\f$ start from 0.
     * @param[in] rhs the appended tensor.
     * @return The appended tensor.
     * @pre
     * 1. The \p rhs tensor and the current tensor cannot be empty.
     * 2. The \p rhs tensor must have the same shape as the current tensor,
     * except the first dimension. Namely, rhs.shape()[i] == this->shape()[i+1]
     * and rhs.shape().size() == this->shape().size()-1.
     * @note If the dtype of the \p rhs is different from the current tensor,
     * the \p rhs will be casted to the dtype of the current tensor.
     * @see append(const Storage &rhs)
     */
    void append(const Tensor &rhs) {
      // Tensor in;
      if (!this->is_contiguous()) this->contiguous_();

      // check Tensor in shape:
      cytnx_error_msg(rhs.shape().size() == 0 || this->shape().size() == 0,
                      "[ERROR] try to append a null Tensor.%s", "\n");
      cytnx_error_msg(rhs.shape().size() != (this->shape().size() - 1),
                      "[ERROR] try to append a Tensor with rank not match.%s", "\n");
      cytnx_uint64 Nelem = 1;
      for (unsigned int i = 0; i < rhs.shape().size(); i++) {
        cytnx_error_msg(rhs.shape()[i] != this->shape()[i + 1],
                        "[ERROR] dimension mismatch @ rhs.rank: [%d] this: [%d] rhs: [%d]\n", i,
                        this->shape()[i + 1], rhs.shape()[i]);
        Nelem *= rhs.shape()[i];
      }

      // check type:
      Tensor in;
      if (rhs.dtype() != this->dtype()) {
        in = rhs.astype(this->dtype());
        if (!in.is_contiguous()) in.contiguous_();
      } else {
        if (!in.is_contiguous())
          in = rhs.contiguous();
        else
          in = rhs;
      }
      this->_impl->_shape[0] += 1;
      cytnx_uint64 oldsize = this->_impl->_storage.size();
      this->_impl->_storage.resize(oldsize + Nelem);
      memcpy(((char *)this->_impl->_storage.data()) +
               oldsize * Type.typeSize(this->dtype()) / sizeof(char),
             in._impl->_storage.data(), Type.typeSize(in.dtype()) * Nelem);
    }
    /**
     * @brief the append function of the Storage.
     * @details This function is the append function of the Storage. It will
     * append the \p srhs Storage to the current tensor. The current tensor must
     * be rank-2 and the \p srhs Storage must have the same size as the second
     * dimension of the current tensor. For example, if the current tensor is
     * \f$A\f$ with size \f$M \times N\f$ and the \p srhs Storage is \f$B\f$
     * with size \f$N\f$, then the output tensor is \f$C\f$ with size \f$M \times
     * (N+1)\f$ where
     * \f[
     * C(i,j) = \begin{cases}
     * A(i,j) & \text{if } j \neq N \\
     * B(i) & \text{if } j = N
     * \end{cases}
     * \f]
     * Here indices \f$i\f$ and \f$j\f$ start from 0.
     * @param[in] srhs the appended Storage.
     * @return The appended tensor.
     * @pre
     * 1. The \p srhs Storage and the current tensor cannot be empty.
     * 2. The current tensor must be rank-2.
     * 3. The \p srhs Storage must have the same size as the second dimension of
     * the current tensor. Namely, srhs.size() == this->shape()[1].
     * @note If the dtype of the \p srhs is different from the current tensor,
     * the \p srhs will be casted to the dtype of the current tensor.
     * @see append(const Tensor &rhs)
     */
    void append(const Storage &srhs) {
      if (!this->is_contiguous()) this->contiguous_();

      // check Tensor in shape:
      cytnx_error_msg(srhs.size() == 0 || this->shape().size() == 0,
                      "[ERROR] try to append a null Tensor.%s", "\n");
      cytnx_error_msg((this->shape().size() - 1) != 1,
                      "[ERROR] append a storage to Tensor can only accept rank-2 Tensor.%s", "\n");
      cytnx_error_msg(this->shape().back() != srhs.size(), "[ERROR] Tensor dmension mismatch!%s",
                      "\n");

      // check type:
      Storage in;
      if (srhs.dtype() != this->dtype()) {
        in = srhs.astype(this->dtype());
      } else {
        in = srhs;
      }
      this->_impl->_shape[0] += 1;
      cytnx_uint64 oldsize = this->_impl->_storage.size();
      this->_impl->_storage.resize(oldsize + in.size());
      memcpy(((char *)this->_impl->_storage.data()) +
               oldsize * Type.typeSize(this->dtype()) / sizeof(char),
             in._impl->Mem, Type.typeSize(in.dtype()) * in.size());
    }
    /*
    void append(const Tensor &rhs){
        // convert to the same type.
        Tensor in;
        if(rhs.dtype() != this->dtype()){
            in = rhs.astype(this->dtype());
        }else{
            in = rhs;
        }

        // 1) check rank
        if(this->shape().size()==1){
            // check if rhs is a scalar tensor (only one element)
            cytnx_error_msg(!(rhs.shape().size()==1 && rhs.shape()[0]==1),"[ERROR] trying to append
    a scalar into multidimentional Tensor is not allow.\n Only rank-1 Tensor can accept scalar
    append.%s","\n"); this->_impl->_shape[0]+=1; this->_impl->_storage.append(0);

        }else{
            cytnx_error_msg(rhs.shape().size() != this->shape().size()-1,"[ERROR] try to append a
    Tensor with rank not match.%s","\n");

        }
        cytnx_error_msg(!this->is_contiguous(),"[ERROR] append require the Tensor to be contiguous.
    suggestion: call contiguous() or contiguous_() first.","\n");
    }
    */
    /**
     * @brief the append function of the scalar.
     * @details This function is the append function of the scalar. It can only append
     * scalar into rank-1 Tensor.
     * @param[in] rhs the appended scalar.
     * @return The appended tensor.
     * @pre
     * 1. The current Tensor must be rank-1. (1D array)
     * 2. The current Tensor must be contiguous.
     * 3. \p rhs must be a scalar.
     */
    template <class T>
    void append(const T &rhs) {
      cytnx_error_msg(this->shape().size() != 1,
                      "[ERROR] trying to append a scalar into multidimentional Tensor is not "
                      "allow.\n Only rank-1 Tensor can accept scalar append.%s",
                      "\n");
      cytnx_error_msg(!this->is_contiguous(),
                      "[ERROR] append require the Tensor to be contiguous. suggestion: call "
                      "contiguous() or contiguous_() first.",
                      "\n");
      this->_impl->_shape[0] += 1;
      this->_impl->_storage.append(rhs);
    }

    /**
     * @brief Check whether two tensors share the same internal memory.
     * @details This function will check whether two tensors share the same
     * internal memory. If the two tensors share the same internal memory, then
     * the function will return true. Otherwise, it will return false. See user
     * guide for more details.
     * @param[in] rhs the tensor to be compared.
     */
    bool same_data(const Tensor &rhs) const;

    // linalg:
    /**
     * @brief the SVD member function. Same as
     * \ref cytnx::linalg::Svd(const Tensor &Tin, const bool &is_UvT)
     * , where \p Tin is the current Tensor.
     */
    std::vector<Tensor> Svd(const bool &is_UvT = true) const;

    /**
     * @brief the Eigh member function. Same as
     * \ref cytnx::linalg::Eigh(const Tensor &Tin, const bool &is_V, const bool &row_v)
     * , where \p Tin is the current Tensor.
     */
    std::vector<Tensor> Eigh(const bool &is_V = true, const bool &row_v = false) const;

    /**
     * @brief the InvM_ member function. Same as
     * \ref cytnx::linalg::InvM_(Tensor &Tin), where \p Tin is the current Tensor.
     */
    Tensor &InvM_();

    /**
     * @brief the InvM member function. Same as
     * \ref cytnx::linalg::InvM(const Tensor &Tin), where \p Tin is the current Tensor.
     */
    Tensor InvM() const;

    /**
     * @brief the Inv_ member function. Same as
     * \ref cytnx::linalg::Inv_(Tensor &Tin, const double &clip)
     */
    Tensor &Inv_(const double &clip);

    /**
     * @brief the Inv member function. Same as
     * \ref cytnx::linalg::Inv(const Tensor &Tin, const double &clip)
     */
    Tensor Inv(const double &clip) const;

    /**
     * @brief the Conj_ member function. Same as
     * \ref cytnx::linalg::Conj_(Tensor &Tin), where \p Tin is the current Tensor.
     */
    Tensor &Conj_();

    /**
     * @brief the Conj member function. Same as
     * \ref cytnx::linalg::Conj(const Tensor &Tin), where \p Tin is the current Tensor.
     */
    Tensor Conj() const;

    /**
     * @brief the Exp_ member function. Same as linalg::Exp_(Tensor &Tin), where \p Tin is the
     * current Tensor.
     */
    Tensor &Exp_();

    /**
     * @brief the Exp member function. Same as linalg::Exp(const Tensor &Tin), where \p Tin is the
     * current Tensor.
     */
    Tensor Exp() const;

    /**
     * @brief the Norm member function. Same as linalg::Norm(const Tensor &Tin), where \p Tin is
     * the current Tensor.
     */
    Tensor Norm() const;

    /**
     * @brief the Pow member function. Same as linalg::Pow(const Tensor &Tin, const cytnx_double
     * &p), where \p Tin is the current Tensor.
     */
    Tensor Pow(const cytnx_double &p) const;

    /**
     * @brief the Pow_ member function. Same as linalg::Pow_(Tensor &Tin, const cytnx_double
     * &p), where \p Tin is the current Tensor.
     */
    Tensor &Pow_(const cytnx_double &p);

    /**
     * @brief the Trace member function. Same as linalg::Trace(const Tensor &Tin, const
     * cytnx_uint64 &a, const cytnx_uint64 &b), where \p Tin is the current Tensor.
     */
    Tensor Trace(const cytnx_uint64 &a = 0, const cytnx_uint64 &b = 1) const;

    /**
     * @brief the Abs member function. Same as linalg::Abs(const Tensor &Tin), where \p Tin is the
     * current Tensor.
     */
    Tensor Abs() const;

    /**
     * @brief the Abs_ member function. Same as linalg::Abs_(Tensor &Tin), where \p Tin is the
     * current Tensor.
     */
    Tensor &Abs_();

    /**
     * @brief the Max member function. Same as linalg::Max(const Tensor &Tin),
     * where \p Tin is the current Tensor.
     */
    Tensor Max() const;

    /**
     * @brief the Min member function. Same as linalg::Min(const Tensor &Tin),
     * where \p Tin is the current Tensor.
     */
    Tensor Min() const;

  };  // class Tensor

  Tensor operator+(const Tensor &lhs, const Tensor::Tproxy &rhs);
  Tensor operator-(const Tensor &lhs, const Tensor::Tproxy &rhs);
  Tensor operator*(const Tensor &lhs, const Tensor::Tproxy &rhs);
  Tensor operator/(const Tensor &lhs, const Tensor::Tproxy &rhs);

  Tensor operator+(const Tensor &lhs, const Scalar::Sproxy &rhs);
  Tensor operator-(const Tensor &lhs, const Scalar::Sproxy &rhs);
  Tensor operator*(const Tensor &lhs, const Scalar::Sproxy &rhs);
  Tensor operator/(const Tensor &lhs, const Scalar::Sproxy &rhs);

  ///@cond
  std::ostream &operator<<(std::ostream &os, const Tensor &in);
  std::ostream &operator<<(std::ostream &os, const Tensor::Tproxy &in);
  ///@endcond
  //{ os << Tensor(in);};
}  // namespace cytnx

#endif
