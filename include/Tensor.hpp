#ifndef _H_Tensor_
#define _H_Tensor_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include "Storage.hpp"
#include "Device.hpp"
#include "intrusive_ptr_base.hpp"
#include <iostream>
#include <fstream>
#include "utils/vec_range.hpp"
#include "utils/dynamic_arg_resolver.hpp"
//#include "linalg.hpp"
#include "Accessor.hpp"
#include <utility>
#include <vector>
#include <initializer_list>
#include <string>
#include "Scalar.hpp"

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
              int device = -1);
    void Init(const Storage &in);
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
        this->_mapper = vec_range(this->_invmapper.size());
        this->_invmapper = this->_mapper;
        this->_contiguous = true;
      }
    }

    void reshape_(const std::vector<cytnx_int64> &new_shape) {
      if (!this->_contiguous) {
        this->contiguous_();
      }
      std::vector<cytnx_uint64> result_shape(new_shape.size());
      cytnx_uint64 new_N = 1;
      bool has_undetermine = false;
      unsigned int Udet_id = 0;
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
          result_shape[i] = new_shape[i];
        }
      }

      if (has_undetermine) {
        cytnx_error_msg(new_N > this->_storage.size(), "%s",
                        "[ERROR] new shape exceed the total number of elements.");
        cytnx_error_msg(this->_storage.size() % new_N, "%s",
                        "[ERROR] unmatch size when reshape with undetermine dimension");
        result_shape[Udet_id] = this->_storage.size() / new_N;
      } else {
        cytnx_error_msg(new_N != this->_storage.size(), "%s",
                        "[ERROR] new shape does not match the number of elements.");
      }

      this->_shape = result_shape;
      this->_mapper = vec_range(result_shape.size());
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
    @param fname file name

    description:
        save the Storage to file with file path specify with input param 'fname' with postfix
    ".cytn"

    */
    void Save(const std::string &fname) const;
    void Save(const char *fname) const;
    void Tofile(const std::string &fname) const;
    void Tofile(const char *fname) const;
    void Tofile(std::fstream &f) const;
    /**
    @brief Load current Tensor to file
    @param fname file name

    description:
        load the Storage from file with file path specify with input param 'fname'

    */
    static Tensor Load(const std::string &fname);
    static Tensor Load(const char *fname);
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
    @param shape the shape of tensor.
    @param dtype the dtype of tensor. This can be any of type defined in cytnx::Type
    @param device the device that tensor to be created. This can be cytnx::Device.cpu or
    cytnx::Device.cuda+<gpuid>

    [Note]
        1. the content of Tensor created will be un-initialize! See \link cytnx::zeros
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
              const int &device = -1) {
      boost::intrusive_ptr<Tensor_impl> tmp(new Tensor_impl());
      this->_impl = tmp;
      this->_impl->Init(shape, dtype, device);
    }
    Tensor(const std::vector<cytnx_uint64> &shape, const unsigned int &dtype = Type.Double,
           const int &device = -1)
        : _impl(new Tensor_impl()) {
      this->Init(shape, dtype, device);
    }
    //@}

    static Tensor from_storage(const Storage &in) {
      Tensor out;
      boost::intrusive_ptr<Tensor_impl> tmp(new Tensor_impl());
      out._impl = tmp;
      out._impl->Init(in);
      return out;
    }

    /**
    @brief the dtype-id of the Tensor
    @return [cytnx_uint64] the dtype_id of the Tensor

    */
    unsigned int dtype() const { return this->_impl->dtype(); }

    /**
    @brief the device-id of the Tensor
    @return [cytnx_int64] the device_id of the Tensor

    */
    int device() const { return this->_impl->device(); }

    /**
    @brief the dtype (in string) of the Tensor
    @return [std::string] the dtype of the Tensor

    */
    std::string dtype_str() const { return this->_impl->dtype_str(); }

    /**
    @brief the device (in string) of the Tensor
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

    description:\n
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
    @param device the device-id that is moving to. it can be any device defined in cytnx::Device
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
    @param device the device-id that is moving to. it can be any device defined in cytnx::Device

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
    @param rnks the permute indices, should have No. of elements equal to the rank of tensor.
    @return [Tensor] a permuted new Tensor

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

    See also \link Tensor::contiguous_ Tensor::contiguous_() \endlink

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

    See also \link Tensor::contiguous Tensor::contiguous() \endlink

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
    @param new_shape the new shape of the Tensor.

    See also \link Tensor::reshape Tensor::reshape() \endlink

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
    @param new_shape the new shape of the Tensor.
    @return [Tensor]

    See also \link Tensor::reshape_ Tensor::reshape_() \endlink

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

    Tensor reshape(const std::vector<cytnx_uint64> &new_shape) const {
      std::vector<cytnx_int64> tmp(new_shape.size());
      memcpy(&tmp[0], &new_shape[0], sizeof(cytnx_uint64) * new_shape.size());
      Tensor out;
      out._impl = this->_impl->reshape(tmp);
      return out;
    }

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
    @param new_type the new dtype. It can be any type defined in cytnx::Type
    @return [Tensor]

    ## Note:
        If the new_type is the same as dtype of the current Tensor, return self.

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
    @brief [C++ only] get an element at specific location.
    @param locator the location of the element
    @return [ref]

    ## Note:
        1. This is for C++ API only!
        2. need template instantiation to resolve the type, which should be consist with the dtype
    of the Tensor. An error will be issued if the template type is inconsist with the current dtype
    of Tensor.
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
    @brief get an from a rank-0 Tensor
    @return [T]

    ## Note:
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
    @return [Tensor]

    See also \link cytnx::Accessor Accessor\endlink for cordinate with Accessor in C++ API.


    ## Note:
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
    @param accessors the list(vector) of accessors.
    @param rhs [Tensor]


    ## Note:
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
    @param accessors the list(vector) of accessors.
    @param rc [Const]

    See also \link cytnx::Tensor::fill Tensor::fill \endlink for filling all elements with assigned
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

    ## Note:
        1. The return storage shares the same instance of the storage of current Tensor. Use \link
    Storage::clone Storage.clone() \endlink to create a new instance of the returned Storage.

    */
    Storage &storage() const { return this->_impl->storage(); }

    /**
    @brief fill all the element of current Tensor with the value.
    @param val the assigned value


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

    bool equiv(const Tensor &rhs) {
      if (this->shape() != rhs.shape()) return false;
      return true;
    }

    Tensor real();
    Tensor imag();

    // Arithmic:
    template <class T>
    Tensor &operator+=(const T &rc);
    template <class T>
    Tensor &operator-=(const T &rc);
    template <class T>
    Tensor &operator*=(const T &rc);
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

    template <class T>
    Tensor Add(const T &rhs) {
      return *this + rhs;
    }
    template <class T>
    Tensor &Add_(const T &rhs) {
      return *this += rhs;
    }

    template <class T>
    Tensor Sub(const T &rhs) {
      return *this - rhs;
    }
    template <class T>
    Tensor &Sub_(const T &rhs) {
      return *this -= rhs;
    }

    template <class T>
    Tensor Mul(const T &rhs) {
      return *this * rhs;
    }
    template <class T>
    Tensor &Mul_(const T &rhs) {
      return *this *= rhs;
    }

    template <class T>
    Tensor Div(const T &rhs) {
      return *this / rhs;
    }
    template <class T>
    Tensor &Div_(const T &rhs) {
      return *this /= rhs;
    }

    template <class T>
    Tensor Cpr(const T &rhs) {
      return *this == rhs;
    }

    // template<class T>
    // Tensor& Cpr_(const T &rhs){
    //
    //      return *this == rhs;
    // }

    template <class T>
    Tensor Mod(const T &rhs) {
      return *this % rhs;
    }

    Tensor operator-() { return this->Mul(-1.); }

    Tensor flatten() const {
      Tensor out = this->clone();
      out.contiguous_();
      out.reshape_({-1});
      return out;
    }

    void flatten_() {
      this->contiguous_();
      this->reshape_({-1});
    }

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

    bool same_data(const Tensor &rhs) const;

    // linalg:
    std::vector<Tensor> Svd(const bool &is_U = true, const bool &is_vT = true) const;
    std::vector<Tensor> Eigh(const bool &is_V = true, const bool &row_v = false) const;
    Tensor &InvM_();
    Tensor InvM() const;
    Tensor &Inv_(const double &clip);
    Tensor Inv(const double &clip) const;

    Tensor &Conj_();
    Tensor Conj() const;
    Tensor &Exp_();
    Tensor Exp() const;
    Tensor Norm() const;
    Tensor Pow(const cytnx_double &p) const;
    Tensor &Pow_(const cytnx_double &p);
    Tensor Trace(const cytnx_uint64 &a = 0, const cytnx_uint64 &b = 1) const;
    Tensor Abs() const;
    Tensor &Abs_();
    Tensor Max() const;
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

  std::ostream &operator<<(std::ostream &os, const Tensor &in);
  std::ostream &operator<<(std::ostream &os, const Tensor::Tproxy &in);
  //{ os << Tensor(in);};
}  // namespace cytnx

#endif
