#ifndef _H_Scalar_
#define _H_Scalar_
#ifndef BACKEND_TORCH
  #include "Type.hpp"
  #include "cytnx_error.hpp"
  #include "intrusive_ptr_base.hpp"
  #include <vector>
  #include <initializer_list>
  #include <string>
  #include <iostream>
  #include <cmath>
  #include <type_traits>
  #include <limits>
namespace cytnx {

  ///@cond
  class Storage_base;
  class Tensor_base;

  // real implementation
  class Scalar_base {
   private:
   public:
    int _dtype;

    // Scalar_base(const Scalar_base &rhs);
    // Scalar_base& operator=(const Scalar_base &rhs);
    Scalar_base() : _dtype(Type.Void){};

    virtual cytnx_float to_cytnx_float() const {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot cast to anytype!!%s", "\n");
      return 0;
    };
    virtual cytnx_double to_cytnx_double() const {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot cast to anytype!!%s", "\n");
      return 0;
    };
    virtual cytnx_complex64 to_cytnx_complex64() const {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot cast to anytype!!%s", "\n");
      return cytnx_complex64(0, 0);
    };
    virtual cytnx_complex128 to_cytnx_complex128() const {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot cast to anytype!!%s", "\n");
      return cytnx_complex128(0, 0);
    };
    virtual cytnx_int64 to_cytnx_int64() const {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot cast to anytype!!%s", "\n");
      return 0;
    };
    virtual cytnx_uint64 to_cytnx_uint64() const {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot cast to anytype!!%s", "\n");
      return 0;
    };
    virtual cytnx_int32 to_cytnx_int32() const {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot cast to anytype!!%s", "\n");
      return 0;
    };
    virtual cytnx_uint32 to_cytnx_uint32() const {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot cast to anytype!!%s", "\n");
      return 0;
    };
    virtual cytnx_int16 to_cytnx_int16() const {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot cast to anytype!!%s", "\n");
      return 0;
    };
    virtual cytnx_uint16 to_cytnx_uint16() const {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot cast to anytype!!%s", "\n");
      return 0;
    };
    virtual cytnx_bool to_cytnx_bool() const {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot cast to anytype!!%s", "\n");
      return 0;
    }

    virtual void iadd(const Scalar_base *c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void iadd(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void iadd(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void iadd(const cytnx_double &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void iadd(const cytnx_float &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void iadd(const cytnx_uint64 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void iadd(const cytnx_int64 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void iadd(const cytnx_uint32 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void iadd(const cytnx_int32 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void iadd(const cytnx_uint16 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void iadd(const cytnx_int16 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void iadd(const cytnx_bool &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }

    virtual void isub(const Scalar_base *c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void isub(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void isub(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void isub(const cytnx_double &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void isub(const cytnx_float &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void isub(const cytnx_uint64 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void isub(const cytnx_int64 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void isub(const cytnx_uint32 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void isub(const cytnx_int32 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void isub(const cytnx_uint16 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void isub(const cytnx_int16 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void isub(const cytnx_bool &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }

    virtual void imul(const Scalar_base *c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void imul(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void imul(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void imul(const cytnx_double &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void imul(const cytnx_float &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void imul(const cytnx_uint64 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void imul(const cytnx_int64 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void imul(const cytnx_uint32 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void imul(const cytnx_int32 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void imul(const cytnx_uint16 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void imul(const cytnx_int16 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void imul(const cytnx_bool &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }

    virtual void idiv(const Scalar_base *c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void idiv(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void idiv(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void idiv(const cytnx_double &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void idiv(const cytnx_float &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void idiv(const cytnx_uint64 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void idiv(const cytnx_int64 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void idiv(const cytnx_uint32 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void idiv(const cytnx_int32 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void idiv(const cytnx_uint16 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void idiv(const cytnx_int16 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void idiv(const cytnx_bool &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }

    virtual bool less(const Scalar_base *c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return 0;
    }
    virtual bool less(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return 0;
    }
    virtual bool less(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return 0;
    }
    virtual bool less(const cytnx_double &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return 0;
    }
    virtual bool less(const cytnx_float &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return 0;
    }
    virtual bool less(const cytnx_uint64 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return 0;
    }
    virtual bool less(const cytnx_int64 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return 0;
    }
    virtual bool less(const cytnx_uint32 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return 0;
    }
    virtual bool less(const cytnx_int32 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return 0;
    }
    virtual bool less(const cytnx_uint16 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return 0;
    }
    virtual bool less(const cytnx_int16 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return 0;
    }
    virtual bool less(const cytnx_bool &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return 0;
    }

    virtual bool greater(const Scalar_base *c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return 0;
    }
    virtual bool greater(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return 0;
    }
    virtual bool greater(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return 0;
    }
    virtual bool greater(const cytnx_double &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return 0;
    }
    virtual bool greater(const cytnx_float &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return 0;
    }
    virtual bool greater(const cytnx_uint64 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return 0;
    }
    virtual bool greater(const cytnx_int64 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return 0;
    }
    virtual bool greater(const cytnx_uint32 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return 0;
    }
    virtual bool greater(const cytnx_int32 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return 0;
    }
    virtual bool greater(const cytnx_uint16 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return 0;
    }
    virtual bool greater(const cytnx_int16 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return 0;
    }
    virtual bool greater(const cytnx_bool &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return 0;
    }

    virtual bool eq(const Scalar_base *c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return 0;
    }
    // virtual bool approx_eq(const Scalar_base *c, const double &tol = 1e-8) {
    //   cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    //   return 0;
    // }

    virtual void set_maxval() {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    };
    virtual void set_minval() {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    };

    virtual void conj_() {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual Scalar_base *get_real() {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return nullptr;
    }
    virtual Scalar_base *get_imag() {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return nullptr;
    }

    virtual void iabs() {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void isqrt() {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }

    virtual void assign_selftype(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void assign_selftype(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void assign_selftype(const cytnx_double &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void assign_selftype(const cytnx_float &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void assign_selftype(const cytnx_uint64 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void assign_selftype(const cytnx_int64 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void assign_selftype(const cytnx_uint32 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void assign_selftype(const cytnx_int32 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void assign_selftype(const cytnx_uint16 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void assign_selftype(const cytnx_int16 &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }
    virtual void assign_selftype(const cytnx_bool &c) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
    }

    virtual Scalar_base *astype(const unsigned int &dtype) {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return nullptr;
    }

    virtual void *get_raw_address() const {
      cytnx_error_msg(true, "[ERROR] Void Type Scalar cannot have operation!!%s", "\n");
      return nullptr;
    }

    virtual void print(std::ostream &os) const {};
    virtual Scalar_base *copy() const {
      Scalar_base *tmp = new Scalar_base();
      return tmp;
    };

    virtual ~Scalar_base(){};
  };

  typedef Scalar_base *(*pScalar_init)();
  ///@endcond

  ///@cond
  class Scalar_init_interface : public Type_class {
   public:
    // std::vector<pScalar_init> UScIInit;
    inline static pScalar_init UScIInit[N_Type];
    inline static bool inited = false;
    Scalar_init_interface();
  };
  extern Scalar_init_interface __ScII;
  ///@endcond

  ///@cond
  class ComplexDoubleScalar : public Scalar_base {
   public:
    cytnx_complex128 _elem;

    ComplexDoubleScalar() : _elem(0) { this->_dtype = Type.ComplexDouble; };
    ComplexDoubleScalar(const cytnx_complex128 &in) : _elem(0) {
      this->_dtype = Type.ComplexDouble;
      this->_elem = in;
    }

    cytnx_float to_cytnx_float() const {
      cytnx_error_msg(true, "[ERROR] cannot cast complex128 to real%s", "\n");
      return 0;
    };
    cytnx_double to_cytnx_double() const {
      cytnx_error_msg(true, "[ERROR] cannot cast complex128 to real%s", "\n");
      return 0;
    };
    cytnx_complex64 to_cytnx_complex64() const { return cytnx_complex64(this->_elem); };
    cytnx_complex128 to_cytnx_complex128() const { return this->_elem; };
    cytnx_int64 to_cytnx_int64() const {
      cytnx_error_msg(true, "[ERROR] cannot cast complex128 to real%s", "\n");
      return 0;
    };
    cytnx_uint64 to_cytnx_uint64() const {
      cytnx_error_msg(true, "[ERROR] cannot cast complex128 to real%s", "\n");
      return 0;
    };
    cytnx_int32 to_cytnx_int32() const {
      cytnx_error_msg(true, "[ERROR] cannot cast complex128 to real%s", "\n");
      return 0;
    };
    cytnx_uint32 to_cytnx_uint32() const {
      cytnx_error_msg(true, "[ERROR] cannot cast complex128 to real%s", "\n");
      return 0;
    };
    cytnx_int16 to_cytnx_int16() const {
      cytnx_error_msg(true, "[ERROR] cannot cast complex128 to real%s", "\n");
      return 0;
    };
    cytnx_uint16 to_cytnx_uint16() const {
      cytnx_error_msg(true, "[ERROR] cannot cast complex128 to real%s", "\n");
      return 0;
    };
    cytnx_bool to_cytnx_bool() const {
      cytnx_error_msg(true, "[ERROR] cannot cast complex128 to real%s", "\n");
      return 0;
    };

    void assign_selftype(const cytnx_complex128 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_complex64 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_double &c) { this->_elem = c; }
    void assign_selftype(const cytnx_float &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint64 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int64 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint32 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int32 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint16 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int16 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_bool &c) { this->_elem = c; }

    void iadd(const Scalar_base *c) { this->_elem += c->to_cytnx_complex128(); }
    void iadd(const cytnx_complex128 &c) { this->_elem += c; }
    void iadd(const cytnx_complex64 &c) { this->_elem += c; }
    void iadd(const cytnx_double &c) { this->_elem += c; }
    void iadd(const cytnx_float &c) { this->_elem += c; }
    void iadd(const cytnx_uint64 &c) { this->_elem += c; }
    void iadd(const cytnx_int64 &c) { this->_elem += c; }
    void iadd(const cytnx_uint32 &c) { this->_elem += c; }
    void iadd(const cytnx_int32 &c) { this->_elem += c; }
    void iadd(const cytnx_uint16 &c) { this->_elem += c; }
    void iadd(const cytnx_int16 &c) { this->_elem += c; }
    void iadd(const cytnx_bool &c) { this->_elem += c; }

    void isub(const Scalar_base *c) { this->_elem -= c->to_cytnx_complex128(); }
    void isub(const cytnx_complex128 &c) { this->_elem -= c; }
    void isub(const cytnx_complex64 &c) { this->_elem -= c; }
    void isub(const cytnx_double &c) { this->_elem -= c; }
    void isub(const cytnx_float &c) { this->_elem -= c; }
    void isub(const cytnx_uint64 &c) { this->_elem -= c; }
    void isub(const cytnx_int64 &c) { this->_elem -= c; }
    void isub(const cytnx_uint32 &c) { this->_elem -= c; }
    void isub(const cytnx_int32 &c) { this->_elem -= c; }
    void isub(const cytnx_uint16 &c) { this->_elem -= c; }
    void isub(const cytnx_int16 &c) { this->_elem -= c; }
    void isub(const cytnx_bool &c) { this->_elem -= c; }

    void imul(const Scalar_base *c) { this->_elem *= c->to_cytnx_complex128(); }
    void imul(const cytnx_complex128 &c) { this->_elem *= c; }
    void imul(const cytnx_complex64 &c) { this->_elem *= c; }
    void imul(const cytnx_double &c) { this->_elem *= c; }
    void imul(const cytnx_float &c) { this->_elem *= c; }
    void imul(const cytnx_uint64 &c) { this->_elem *= c; }
    void imul(const cytnx_int64 &c) { this->_elem *= c; }
    void imul(const cytnx_uint32 &c) { this->_elem *= c; }
    void imul(const cytnx_int32 &c) { this->_elem *= c; }
    void imul(const cytnx_uint16 &c) { this->_elem *= c; }
    void imul(const cytnx_int16 &c) { this->_elem *= c; }
    void imul(const cytnx_bool &c) { this->_elem *= c; }

    void idiv(const Scalar_base *c) { this->_elem /= c->to_cytnx_complex128(); }
    void idiv(const cytnx_complex128 &c) { this->_elem /= c; }
    void idiv(const cytnx_complex64 &c) { this->_elem /= c; }
    void idiv(const cytnx_double &c) { this->_elem /= c; }
    void idiv(const cytnx_float &c) { this->_elem /= c; }
    void idiv(const cytnx_uint64 &c) { this->_elem /= c; }
    void idiv(const cytnx_int64 &c) { this->_elem /= c; }
    void idiv(const cytnx_uint32 &c) { this->_elem /= c; }
    void idiv(const cytnx_int32 &c) { this->_elem /= c; }
    void idiv(const cytnx_uint16 &c) { this->_elem /= c; }
    void idiv(const cytnx_int16 &c) { this->_elem /= c; }
    void idiv(const cytnx_bool &c) { this->_elem /= c; }

    void iabs() { this->_elem = std::abs(cytnx_complex128(this->_elem)); }
    void isqrt() { this->_elem = std::sqrt(this->_elem); }

    bool less(const Scalar_base *c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_double &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_float &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_uint64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_int64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_uint32 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_int32 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_uint16 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_int16 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_bool &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }

    bool greater(const Scalar_base *c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_double &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_float &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_uint64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_int64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_uint32 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_int32 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_uint16 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_int16 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_bool &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }

    bool eq(const Scalar_base *c) { return this->_elem == c->to_cytnx_complex128(); }
    // bool approx_eq(const Scalar_base *c, const cytnx_double &tol = 1e-8) {
    //   return std::abs(this->_elem - c->to_cytnx_complex128()) <= tol;
    // }

    void set_maxval() {
      cytnx_error_msg(true, "[ERROR] maxval not supported for complex type%s", "\n");
    }
    void set_minval() {
      cytnx_error_msg(true, "[ERROR] minval not supported for complex type%s", "\n");
    }

    void conj_() { this->_elem = std::conj(this->_elem); }
    Scalar_base *get_real() {
      Scalar_base *tmp = __ScII.UScIInit[Type.Double]();
      tmp->assign_selftype(this->_elem.real());
      return tmp;
    }
    Scalar_base *get_imag() {
      Scalar_base *tmp = __ScII.UScIInit[Type.Double]();
      tmp->assign_selftype(this->_elem.imag());
      return tmp;
    }

    void *get_raw_address() const { return (void *)(&this->_elem); }

    Scalar_base *astype(const unsigned int &dtype) {
      Scalar_base *tmp = __ScII.UScIInit[dtype]();
      tmp->assign_selftype(this->_elem);
      return tmp;
    }

    Scalar_base *copy() const {
      ComplexDoubleScalar *tmp = new ComplexDoubleScalar(this->_elem);
      return tmp;
    };
    void print(std::ostream &os) const { os << "< " << this->_elem << " >"; };
  };

  class ComplexFloatScalar : public Scalar_base {
   public:
    cytnx_complex64 _elem;

    ComplexFloatScalar() : _elem(0) { this->_dtype = Type.ComplexFloat; };
    ComplexFloatScalar(const cytnx_complex64 &in) : _elem(0) {
      this->_dtype = Type.ComplexFloat;
      this->_elem = in;
    }

    cytnx_float to_cytnx_float() const {
      cytnx_error_msg(true, "[ERROR] cannot cast complex64 to real%s", "\n");
      return 0;
    };
    cytnx_double to_cytnx_double() const {
      cytnx_error_msg(true, "[ERROR] cannot cast complex64 to real%s", "\n");
      return 0;
    };
    cytnx_complex64 to_cytnx_complex64() const { return this->_elem; };
    cytnx_complex128 to_cytnx_complex128() const { return cytnx_complex128(this->_elem); };
    cytnx_int64 to_cytnx_int64() const {
      cytnx_error_msg(true, "[ERROR] cannot cast complex64 to real%s", "\n");
      return 0;
    };
    cytnx_uint64 to_cytnx_uint64() const {
      cytnx_error_msg(true, "[ERROR] cannot cast complex64 to real%s", "\n");
      return 0;
    };
    cytnx_int32 to_cytnx_int32() const {
      cytnx_error_msg(true, "[ERROR] cannot cast complex64 to real%s", "\n");
      return 0;
    };
    cytnx_uint32 to_cytnx_uint32() const {
      cytnx_error_msg(true, "[ERROR] cannot cast complex64 to real%s", "\n");
      return 0;
    };
    cytnx_int16 to_cytnx_int16() const {
      cytnx_error_msg(true, "[ERROR] cannot cast complex64 to real%s", "\n");
      return 0;
    };
    cytnx_uint16 to_cytnx_uint16() const {
      cytnx_error_msg(true, "[ERROR] cannot cast complex64 to real%s", "\n");
      return 0;
    };
    cytnx_bool to_cytnx_bool() const {
      cytnx_error_msg(true, "[ERROR] cannot cast complex64 to real%s", "\n");
      return 0;
    };

    void assign_selftype(const cytnx_complex128 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_complex64 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_double &c) { this->_elem = c; }
    void assign_selftype(const cytnx_float &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint64 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int64 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint32 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int32 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint16 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int16 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_bool &c) { this->_elem = c; }

    void iadd(const Scalar_base *c) { this->_elem += c->to_cytnx_complex64(); }
    void iadd(const cytnx_complex128 &c) { this->_elem += c; }
    void iadd(const cytnx_complex64 &c) { this->_elem += c; }
    void iadd(const cytnx_double &c) { this->_elem += c; }
    void iadd(const cytnx_float &c) { this->_elem += c; }
    void iadd(const cytnx_uint64 &c) { this->_elem += c; }
    void iadd(const cytnx_int64 &c) { this->_elem += c; }
    void iadd(const cytnx_uint32 &c) { this->_elem += c; }
    void iadd(const cytnx_int32 &c) { this->_elem += c; }
    void iadd(const cytnx_uint16 &c) { this->_elem += c; }
    void iadd(const cytnx_int16 &c) { this->_elem += c; }
    void iadd(const cytnx_bool &c) { this->_elem += c; }

    void isub(const Scalar_base *c) { this->_elem -= c->to_cytnx_complex64(); }
    void isub(const cytnx_complex128 &c) { this->_elem -= c; }
    void isub(const cytnx_complex64 &c) { this->_elem -= c; }
    void isub(const cytnx_double &c) { this->_elem -= c; }
    void isub(const cytnx_float &c) { this->_elem -= c; }
    void isub(const cytnx_uint64 &c) { this->_elem -= c; }
    void isub(const cytnx_int64 &c) { this->_elem -= c; }
    void isub(const cytnx_uint32 &c) { this->_elem -= c; }
    void isub(const cytnx_int32 &c) { this->_elem -= c; }
    void isub(const cytnx_uint16 &c) { this->_elem -= c; }
    void isub(const cytnx_int16 &c) { this->_elem -= c; }
    void isub(const cytnx_bool &c) { this->_elem -= c; }

    void imul(const Scalar_base *c) { this->_elem *= c->to_cytnx_complex64(); }
    void imul(const cytnx_complex128 &c) { this->_elem *= c; }
    void imul(const cytnx_complex64 &c) { this->_elem *= c; }
    void imul(const cytnx_double &c) { this->_elem *= c; }
    void imul(const cytnx_float &c) { this->_elem *= c; }
    void imul(const cytnx_uint64 &c) { this->_elem *= c; }
    void imul(const cytnx_int64 &c) { this->_elem *= c; }
    void imul(const cytnx_uint32 &c) { this->_elem *= c; }
    void imul(const cytnx_int32 &c) { this->_elem *= c; }
    void imul(const cytnx_uint16 &c) { this->_elem *= c; }
    void imul(const cytnx_int16 &c) { this->_elem *= c; }
    void imul(const cytnx_bool &c) { this->_elem *= c; }

    void idiv(const Scalar_base *c) { this->_elem /= c->to_cytnx_complex64(); }
    void idiv(const cytnx_complex128 &c) { this->_elem /= c; }
    void idiv(const cytnx_complex64 &c) { this->_elem /= c; }
    void idiv(const cytnx_double &c) { this->_elem /= c; }
    void idiv(const cytnx_float &c) { this->_elem /= c; }
    void idiv(const cytnx_uint64 &c) { this->_elem /= c; }
    void idiv(const cytnx_int64 &c) { this->_elem /= c; }
    void idiv(const cytnx_uint32 &c) { this->_elem /= c; }
    void idiv(const cytnx_int32 &c) { this->_elem /= c; }
    void idiv(const cytnx_uint16 &c) { this->_elem /= c; }
    void idiv(const cytnx_int16 &c) { this->_elem /= c; }
    void idiv(const cytnx_bool &c) { this->_elem /= c; }

    bool less(const Scalar_base *c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_double &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_float &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_uint64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_int64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_uint32 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_int32 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_uint16 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_int16 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_bool &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }

    bool greater(const Scalar_base *c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_double &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_float &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_uint64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_int64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_uint32 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_int32 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_uint16 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_int16 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_bool &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }

    bool eq(const Scalar_base *c) { return this->_elem == c->to_cytnx_complex64(); }
    // bool approx_eq(const Scalar_base *c, const cytnx_double &tol = 1e-8) {
    //   return std::abs(this->_elem - c->to_cytnx_complex64()) <= tol;
    // }

    void set_maxval() {
      cytnx_error_msg(true, "[ERROR] maxval not supported for complex type%s", "\n");
    }
    void set_minval() {
      cytnx_error_msg(true, "[ERROR] minval not supported for complex type%s", "\n");
    }

    void conj_() { this->_elem = std::conj(this->_elem); }
    Scalar_base *get_real() {
      Scalar_base *tmp = __ScII.UScIInit[Type.Float]();
      tmp->assign_selftype(this->_elem.real());
      return tmp;
    }
    Scalar_base *get_imag() {
      Scalar_base *tmp = __ScII.UScIInit[Type.Float]();
      tmp->assign_selftype(this->_elem.imag());
      return tmp;
    }

    void iabs() { this->_elem = std::abs(this->_elem); }
    void isqrt() { this->_elem = std::sqrt(this->_elem); }

    void *get_raw_address() const { return (void *)(&this->_elem); }
    Scalar_base *astype(const unsigned int &dtype) {
      Scalar_base *tmp = __ScII.UScIInit[dtype]();
      tmp->assign_selftype(this->_elem);
      return tmp;
    }

    Scalar_base *copy() const {
      ComplexFloatScalar *tmp = new ComplexFloatScalar(this->_elem);
      return tmp;
    };
    void print(std::ostream &os) const { os << "< " << this->_elem << " >"; };
  };

  class DoubleScalar : public Scalar_base {
   public:
    cytnx_double _elem;

    DoubleScalar() : _elem(0) { this->_dtype = Type.Double; };
    DoubleScalar(const cytnx_double &in) : _elem(0) {
      this->_dtype = Type.Double;
      this->_elem = in;
    }

    cytnx_float to_cytnx_float() const { return this->_elem; };
    cytnx_double to_cytnx_double() const { return this->_elem; };
    cytnx_complex64 to_cytnx_complex64() const { return this->_elem; };
    cytnx_complex128 to_cytnx_complex128() const { return this->_elem; };
    cytnx_int64 to_cytnx_int64() const { return this->_elem; };
    cytnx_uint64 to_cytnx_uint64() const { return this->_elem; };
    cytnx_int32 to_cytnx_int32() const { return this->_elem; };
    cytnx_uint32 to_cytnx_uint32() const { return this->_elem; };
    cytnx_int16 to_cytnx_int16() const { return this->_elem; };
    cytnx_uint16 to_cytnx_uint16() const { return this->_elem; };
    cytnx_bool to_cytnx_bool() const { return this->_elem; };

    void assign_selftype(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot convert complex to real%s", "\n");
    }
    void assign_selftype(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot convert complex to real%s", "\n");
    }
    void assign_selftype(const cytnx_double &c) { this->_elem = c; }
    void assign_selftype(const cytnx_float &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint64 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int64 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint32 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int32 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint16 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int16 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_bool &c) { this->_elem = c; }

    void iadd(const Scalar_base *c) { this->_elem += c->to_cytnx_double(); }
    void iadd(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void iadd(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void iadd(const cytnx_double &c) { this->_elem += c; }
    void iadd(const cytnx_float &c) { this->_elem += c; }
    void iadd(const cytnx_uint64 &c) { this->_elem += c; }
    void iadd(const cytnx_int64 &c) { this->_elem += c; }
    void iadd(const cytnx_uint32 &c) { this->_elem += c; }
    void iadd(const cytnx_int32 &c) { this->_elem += c; }
    void iadd(const cytnx_uint16 &c) { this->_elem += c; }
    void iadd(const cytnx_int16 &c) { this->_elem += c; }
    void iadd(const cytnx_bool &c) { this->_elem += c; }

    void isub(const Scalar_base *c) { this->_elem -= c->to_cytnx_double(); }
    void isub(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void isub(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void isub(const cytnx_double &c) { this->_elem -= c; }
    void isub(const cytnx_float &c) { this->_elem -= c; }
    void isub(const cytnx_uint64 &c) { this->_elem -= c; }
    void isub(const cytnx_int64 &c) { this->_elem -= c; }
    void isub(const cytnx_uint32 &c) { this->_elem -= c; }
    void isub(const cytnx_int32 &c) { this->_elem -= c; }
    void isub(const cytnx_uint16 &c) { this->_elem -= c; }
    void isub(const cytnx_int16 &c) { this->_elem -= c; }
    void isub(const cytnx_bool &c) { this->_elem -= c; }

    void imul(const Scalar_base *c) { this->_elem *= c->to_cytnx_double(); }
    void imul(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void imul(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void imul(const cytnx_double &c) { this->_elem *= c; }
    void imul(const cytnx_float &c) { this->_elem *= c; }
    void imul(const cytnx_uint64 &c) { this->_elem *= c; }
    void imul(const cytnx_int64 &c) { this->_elem *= c; }
    void imul(const cytnx_uint32 &c) { this->_elem *= c; }
    void imul(const cytnx_int32 &c) { this->_elem *= c; }
    void imul(const cytnx_uint16 &c) { this->_elem *= c; }
    void imul(const cytnx_int16 &c) { this->_elem *= c; }
    void imul(const cytnx_bool &c) { this->_elem *= c; }

    void idiv(const Scalar_base *c) { this->_elem /= c->to_cytnx_double(); }
    void idiv(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void idiv(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void idiv(const cytnx_double &c) { this->_elem /= c; }
    void idiv(const cytnx_float &c) { this->_elem /= c; }
    void idiv(const cytnx_uint64 &c) { this->_elem /= c; }
    void idiv(const cytnx_int64 &c) { this->_elem /= c; }
    void idiv(const cytnx_uint32 &c) { this->_elem /= c; }
    void idiv(const cytnx_int32 &c) { this->_elem /= c; }
    void idiv(const cytnx_uint16 &c) { this->_elem /= c; }
    void idiv(const cytnx_int16 &c) { this->_elem /= c; }
    void idiv(const cytnx_bool &c) { this->_elem /= c; }

    void iabs() { this->_elem = std::abs(this->_elem); }
    void isqrt() { this->_elem = std::sqrt(this->_elem); }

    bool less(const Scalar_base *c) { return this->_elem < c->to_cytnx_double(); }
    bool less(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_double &c) { return this->_elem < c; }
    bool less(const cytnx_float &c) { return this->_elem < c; }
    bool less(const cytnx_uint64 &c) { return this->_elem < c; }
    bool less(const cytnx_int64 &c) { return this->_elem < c; }
    bool less(const cytnx_uint32 &c) { return this->_elem < c; }
    bool less(const cytnx_int32 &c) { return this->_elem < c; }
    bool less(const cytnx_uint16 &c) { return this->_elem < c; }
    bool less(const cytnx_int16 &c) { return this->_elem < c; }
    bool less(const cytnx_bool &c) { return this->_elem < c; }

    bool greater(const Scalar_base *c) { return this->_elem > c->to_cytnx_double(); }
    bool greater(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_double &c) { return this->_elem > c; }
    bool greater(const cytnx_float &c) { return this->_elem > c; }
    bool greater(const cytnx_uint64 &c) { return this->_elem > c; }
    bool greater(const cytnx_int64 &c) { return this->_elem > c; }
    bool greater(const cytnx_uint32 &c) { return this->_elem > c; }
    bool greater(const cytnx_int32 &c) { return this->_elem > c; }
    bool greater(const cytnx_uint16 &c) { return this->_elem > c; }
    bool greater(const cytnx_int16 &c) { return this->_elem > c; }
    bool greater(const cytnx_bool &c) { return this->_elem > c; }

    bool eq(const Scalar_base *c) { return this->_elem == c->to_cytnx_double(); }
    // bool approx_eq(const Scalar_base *c, const cytnx_double &tol = 1e-8) {
    //   return std::abs(this->_elem - c->to_cytnx_double()) <= tol;
    // }

    void set_maxval() { this->_elem = std::numeric_limits<double>::max(); }
    void set_minval() { this->_elem = std::numeric_limits<double>::min(); }

    void conj_() { return; }
    Scalar_base *get_real() { return this->copy(); }
    Scalar_base *get_imag() {
      cytnx_error_msg(true, "[ERROR] real type Scalar does not have imag part!%s", "\n");
      return nullptr;
    }

    void *get_raw_address() const { return (void *)(&this->_elem); }
    Scalar_base *astype(const unsigned int &dtype) {
      Scalar_base *tmp = __ScII.UScIInit[dtype]();
      tmp->assign_selftype(this->_elem);
      return tmp;
    }
    Scalar_base *copy() const {
      DoubleScalar *tmp = new DoubleScalar(this->_elem);
      return tmp;
    };
    void print(std::ostream &os) const { os << "< " << this->_elem << " >"; };
  };

  class FloatScalar : public Scalar_base {
   public:
    cytnx_float _elem;

    FloatScalar() : _elem(0) { this->_dtype = Type.Float; };
    FloatScalar(const cytnx_float &in) : _elem(0) {
      this->_dtype = Type.Float;
      this->_elem = in;
    }

    cytnx_float to_cytnx_float() const { return this->_elem; };
    cytnx_double to_cytnx_double() const { return this->_elem; };
    cytnx_complex64 to_cytnx_complex64() const { return this->_elem; };
    cytnx_complex128 to_cytnx_complex128() const { return this->_elem; };
    cytnx_int64 to_cytnx_int64() const { return this->_elem; };
    cytnx_uint64 to_cytnx_uint64() const { return this->_elem; };
    cytnx_int32 to_cytnx_int32() const { return this->_elem; };
    cytnx_uint32 to_cytnx_uint32() const { return this->_elem; };
    cytnx_int16 to_cytnx_int16() const { return this->_elem; };
    cytnx_uint16 to_cytnx_uint16() const { return this->_elem; };
    cytnx_bool to_cytnx_bool() const { return this->_elem; };

    void assign_selftype(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot convert complex to real%s", "\n");
    }
    void assign_selftype(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot convert complex to real%s", "\n");
    }
    void assign_selftype(const cytnx_double &c) { this->_elem = c; }
    void assign_selftype(const cytnx_float &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint64 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int64 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint32 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int32 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint16 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int16 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_bool &c) { this->_elem = c; }

    void iadd(const Scalar_base *c) { this->_elem += c->to_cytnx_float(); }
    void iadd(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void iadd(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void iadd(const cytnx_double &c) { this->_elem += c; }
    void iadd(const cytnx_float &c) { this->_elem += c; }
    void iadd(const cytnx_uint64 &c) { this->_elem += c; }
    void iadd(const cytnx_int64 &c) { this->_elem += c; }
    void iadd(const cytnx_uint32 &c) { this->_elem += c; }
    void iadd(const cytnx_int32 &c) { this->_elem += c; }
    void iadd(const cytnx_uint16 &c) { this->_elem += c; }
    void iadd(const cytnx_int16 &c) { this->_elem += c; }
    void iadd(const cytnx_bool &c) { this->_elem += c; }

    void isub(const Scalar_base *c) { this->_elem -= c->to_cytnx_float(); }
    void isub(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void isub(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void isub(const cytnx_double &c) { this->_elem -= c; }
    void isub(const cytnx_float &c) { this->_elem -= c; }
    void isub(const cytnx_uint64 &c) { this->_elem -= c; }
    void isub(const cytnx_int64 &c) { this->_elem -= c; }
    void isub(const cytnx_uint32 &c) { this->_elem -= c; }
    void isub(const cytnx_int32 &c) { this->_elem -= c; }
    void isub(const cytnx_uint16 &c) { this->_elem -= c; }
    void isub(const cytnx_int16 &c) { this->_elem -= c; }
    void isub(const cytnx_bool &c) { this->_elem -= c; }

    void imul(const Scalar_base *c) { this->_elem *= c->to_cytnx_float(); }
    void imul(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void imul(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void imul(const cytnx_double &c) { this->_elem *= c; }
    void imul(const cytnx_float &c) { this->_elem *= c; }
    void imul(const cytnx_uint64 &c) { this->_elem *= c; }
    void imul(const cytnx_int64 &c) { this->_elem *= c; }
    void imul(const cytnx_uint32 &c) { this->_elem *= c; }
    void imul(const cytnx_int32 &c) { this->_elem *= c; }
    void imul(const cytnx_uint16 &c) { this->_elem *= c; }
    void imul(const cytnx_int16 &c) { this->_elem *= c; }
    void imul(const cytnx_bool &c) { this->_elem *= c; }

    void idiv(const Scalar_base *c) { this->_elem /= c->to_cytnx_float(); }
    void idiv(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void idiv(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void idiv(const cytnx_double &c) { this->_elem /= c; }
    void idiv(const cytnx_float &c) { this->_elem /= c; }
    void idiv(const cytnx_uint64 &c) { this->_elem /= c; }
    void idiv(const cytnx_int64 &c) { this->_elem /= c; }
    void idiv(const cytnx_uint32 &c) { this->_elem /= c; }
    void idiv(const cytnx_int32 &c) { this->_elem /= c; }
    void idiv(const cytnx_uint16 &c) { this->_elem /= c; }
    void idiv(const cytnx_int16 &c) { this->_elem /= c; }
    void idiv(const cytnx_bool &c) { this->_elem /= c; }

    bool less(const Scalar_base *c) { return this->_elem < c->to_cytnx_float(); }
    bool less(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_double &c) { return this->_elem < c; }
    bool less(const cytnx_float &c) { return this->_elem < c; }
    bool less(const cytnx_uint64 &c) { return this->_elem < c; }
    bool less(const cytnx_int64 &c) { return this->_elem < c; }
    bool less(const cytnx_uint32 &c) { return this->_elem < c; }
    bool less(const cytnx_int32 &c) { return this->_elem < c; }
    bool less(const cytnx_uint16 &c) { return this->_elem < c; }
    bool less(const cytnx_int16 &c) { return this->_elem < c; }
    bool less(const cytnx_bool &c) { return this->_elem < c; }

    bool greater(const Scalar_base *c) { return this->_elem > c->to_cytnx_float(); }
    bool greater(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_double &c) { return this->_elem > c; }
    bool greater(const cytnx_float &c) { return this->_elem > c; }
    bool greater(const cytnx_uint64 &c) { return this->_elem > c; }
    bool greater(const cytnx_int64 &c) { return this->_elem > c; }
    bool greater(const cytnx_uint32 &c) { return this->_elem > c; }
    bool greater(const cytnx_int32 &c) { return this->_elem > c; }
    bool greater(const cytnx_uint16 &c) { return this->_elem > c; }
    bool greater(const cytnx_int16 &c) { return this->_elem > c; }
    bool greater(const cytnx_bool &c) { return this->_elem > c; }

    bool eq(const Scalar_base *c) { return this->_elem == c->to_cytnx_float(); }
    // bool approx_eq(const Scalar_base *c, const cytnx_double &tol = 1e-8) {
    //   return std::abs(this->_elem - c->to_cytnx_float()) <= tol;
    // }

    void isqrt() { this->_elem = std::sqrt(this->_elem); }

    void conj_() { return; }
    Scalar_base *get_real() { return this->copy(); }
    Scalar_base *get_imag() {
      cytnx_error_msg(true, "[ERROR] real type Scalar does not have imag part!%s", "\n");
      return nullptr;
    }

    void set_maxval() { this->_elem = std::numeric_limits<float>::max(); }
    void set_minval() { this->_elem = std::numeric_limits<float>::min(); }

    void iabs() { this->_elem = std::abs(this->_elem); }

    void *get_raw_address() const { return (void *)(&this->_elem); }
    Scalar_base *astype(const unsigned int &dtype) {
      Scalar_base *tmp = __ScII.UScIInit[dtype]();
      tmp->assign_selftype(this->_elem);
      return tmp;
    }
    Scalar_base *copy() const {
      FloatScalar *tmp = new FloatScalar(this->_elem);
      return tmp;
    };
    void print(std::ostream &os) const { os << "< " << this->_elem << " >"; };
  };

  class Int64Scalar : public Scalar_base {
   public:
    cytnx_int64 _elem;

    Int64Scalar() : _elem(0) { this->_dtype = Type.Int64; };
    Int64Scalar(const cytnx_int64 &in) : _elem(0) {
      this->_dtype = Type.Int64;
      this->_elem = in;
    }

    cytnx_float to_cytnx_float() const { return this->_elem; };
    cytnx_double to_cytnx_double() const { return this->_elem; };
    cytnx_complex64 to_cytnx_complex64() const { return this->_elem; };
    cytnx_complex128 to_cytnx_complex128() const { return this->_elem; };
    cytnx_int64 to_cytnx_int64() const { return this->_elem; };
    cytnx_uint64 to_cytnx_uint64() const { return this->_elem; };
    cytnx_int32 to_cytnx_int32() const { return this->_elem; };
    cytnx_uint32 to_cytnx_uint32() const { return this->_elem; };
    cytnx_int16 to_cytnx_int16() const { return this->_elem; };
    cytnx_uint16 to_cytnx_uint16() const { return this->_elem; };
    cytnx_bool to_cytnx_bool() const { return this->_elem; };

    void assign_selftype(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot convert complex to real%s", "\n");
    }
    void assign_selftype(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot convert complex to real%s", "\n");
    }
    void assign_selftype(const cytnx_double &c) { this->_elem = c; }
    void assign_selftype(const cytnx_float &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint64 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int64 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint32 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int32 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint16 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int16 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_bool &c) { this->_elem = c; }

    void iadd(const Scalar_base *c) { this->_elem += c->to_cytnx_int64(); }
    void iadd(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void iadd(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void iadd(const cytnx_double &c) { this->_elem += c; }
    void iadd(const cytnx_float &c) { this->_elem += c; }
    void iadd(const cytnx_uint64 &c) { this->_elem += c; }
    void iadd(const cytnx_int64 &c) { this->_elem += c; }
    void iadd(const cytnx_uint32 &c) { this->_elem += c; }
    void iadd(const cytnx_int32 &c) { this->_elem += c; }
    void iadd(const cytnx_uint16 &c) { this->_elem += c; }
    void iadd(const cytnx_int16 &c) { this->_elem += c; }
    void iadd(const cytnx_bool &c) { this->_elem += c; }

    void isub(const Scalar_base *c) { this->_elem -= c->to_cytnx_int64(); }
    void isub(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void isub(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void isub(const cytnx_double &c) { this->_elem -= c; }
    void isub(const cytnx_float &c) { this->_elem -= c; }
    void isub(const cytnx_uint64 &c) { this->_elem -= c; }
    void isub(const cytnx_int64 &c) { this->_elem -= c; }
    void isub(const cytnx_uint32 &c) { this->_elem -= c; }
    void isub(const cytnx_int32 &c) { this->_elem -= c; }
    void isub(const cytnx_uint16 &c) { this->_elem -= c; }
    void isub(const cytnx_int16 &c) { this->_elem -= c; }
    void isub(const cytnx_bool &c) { this->_elem -= c; }

    void imul(const Scalar_base *c) { this->_elem *= c->to_cytnx_int64(); }
    void imul(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void imul(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void imul(const cytnx_double &c) { this->_elem *= c; }
    void imul(const cytnx_float &c) { this->_elem *= c; }
    void imul(const cytnx_uint64 &c) { this->_elem *= c; }
    void imul(const cytnx_int64 &c) { this->_elem *= c; }
    void imul(const cytnx_uint32 &c) { this->_elem *= c; }
    void imul(const cytnx_int32 &c) { this->_elem *= c; }
    void imul(const cytnx_uint16 &c) { this->_elem *= c; }
    void imul(const cytnx_int16 &c) { this->_elem *= c; }
    void imul(const cytnx_bool &c) { this->_elem *= c; }

    void idiv(const Scalar_base *c) { this->_elem /= c->to_cytnx_int64(); }
    void idiv(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void idiv(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void idiv(const cytnx_double &c) { this->_elem /= c; }
    void idiv(const cytnx_float &c) { this->_elem /= c; }
    void idiv(const cytnx_uint64 &c) { this->_elem /= c; }
    void idiv(const cytnx_int64 &c) { this->_elem /= c; }
    void idiv(const cytnx_uint32 &c) { this->_elem /= c; }
    void idiv(const cytnx_int32 &c) { this->_elem /= c; }
    void idiv(const cytnx_uint16 &c) { this->_elem /= c; }
    void idiv(const cytnx_int16 &c) { this->_elem /= c; }
    void idiv(const cytnx_bool &c) { this->_elem /= c; }

    void iabs() { this->_elem = std::abs(this->_elem); }
    void isqrt() { this->_elem = std::sqrt(this->_elem); }

    bool less(const Scalar_base *c) { return this->_elem < c->to_cytnx_int64(); }
    bool less(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_double &c) { return this->_elem < c; }
    bool less(const cytnx_float &c) { return this->_elem < c; }
    bool less(const cytnx_uint64 &c) { return this->_elem < c; }
    bool less(const cytnx_int64 &c) { return this->_elem < c; }
    bool less(const cytnx_uint32 &c) { return this->_elem < c; }
    bool less(const cytnx_int32 &c) { return this->_elem < c; }
    bool less(const cytnx_uint16 &c) { return this->_elem < c; }
    bool less(const cytnx_int16 &c) { return this->_elem < c; }
    bool less(const cytnx_bool &c) { return this->_elem < c; }

    bool greater(const Scalar_base *c) { return this->_elem > c->to_cytnx_int64(); }
    bool greater(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_double &c) { return this->_elem > c; }
    bool greater(const cytnx_float &c) { return this->_elem > c; }
    bool greater(const cytnx_uint64 &c) { return this->_elem > c; }
    bool greater(const cytnx_int64 &c) { return this->_elem > c; }
    bool greater(const cytnx_uint32 &c) { return this->_elem > c; }
    bool greater(const cytnx_int32 &c) { return this->_elem > c; }
    bool greater(const cytnx_uint16 &c) { return this->_elem > c; }
    bool greater(const cytnx_int16 &c) { return this->_elem > c; }
    bool greater(const cytnx_bool &c) { return this->_elem > c; }

    bool eq(const Scalar_base *c) { return this->_elem == c->to_cytnx_int64(); }
    // bool approx_eq(const Scalar_base *c, const cytnx_double &tol = 1e-8) {
    //   return std::abs((cytnx_double)this->_elem - (cytnx_double)c->to_cytnx_int64()) <= tol;
    // }

    void conj_() { return; }
    Scalar_base *get_real() { return this->copy(); }
    Scalar_base *get_imag() {
      cytnx_error_msg(true, "[ERROR] real type Scalar does not have imag part!%s", "\n");
      return nullptr;
    }

    void set_maxval() { this->_elem = std::numeric_limits<cytnx_int64>::max(); }
    void set_minval() { this->_elem = std::numeric_limits<cytnx_int64>::min(); }

    void *get_raw_address() const { return (void *)(&this->_elem); }
    Scalar_base *astype(const unsigned int &dtype) {
      Scalar_base *tmp = __ScII.UScIInit[dtype]();
      tmp->assign_selftype(this->_elem);
      return tmp;
    }
    Scalar_base *copy() const {
      Int64Scalar *tmp = new Int64Scalar(this->_elem);
      return tmp;
    };
    void print(std::ostream &os) const { os << "< " << this->_elem << " >"; };
  };
  class Uint64Scalar : public Scalar_base {
   public:
    cytnx_uint64 _elem;

    Uint64Scalar() : _elem(0) { this->_dtype = Type.Uint64; };
    Uint64Scalar(const cytnx_uint64 &in) : _elem(0) {
      this->_dtype = Type.Uint64;
      this->_elem = in;
    }

    cytnx_float to_cytnx_float() const { return this->_elem; };
    cytnx_double to_cytnx_double() const { return this->_elem; };
    cytnx_complex64 to_cytnx_complex64() const { return this->_elem; };
    cytnx_complex128 to_cytnx_complex128() const { return this->_elem; };
    cytnx_int64 to_cytnx_int64() const { return this->_elem; };
    cytnx_uint64 to_cytnx_uint64() const { return this->_elem; };
    cytnx_int32 to_cytnx_int32() const { return this->_elem; };
    cytnx_uint32 to_cytnx_uint32() const { return this->_elem; };
    cytnx_int16 to_cytnx_int16() const { return this->_elem; };
    cytnx_uint16 to_cytnx_uint16() const { return this->_elem; };
    cytnx_bool to_cytnx_bool() const { return this->_elem; };

    void assign_selftype(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot convert complex to real%s", "\n");
    }
    void assign_selftype(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot convert complex to real%s", "\n");
    }
    void assign_selftype(const cytnx_double &c) { this->_elem = c; }
    void assign_selftype(const cytnx_float &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint64 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int64 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint32 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int32 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint16 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int16 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_bool &c) { this->_elem = c; }

    void iadd(const Scalar_base *c) { this->_elem += c->to_cytnx_uint64(); }
    void iadd(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void iadd(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void iadd(const cytnx_double &c) { this->_elem += c; }
    void iadd(const cytnx_float &c) { this->_elem += c; }
    void iadd(const cytnx_uint64 &c) { this->_elem += c; }
    void iadd(const cytnx_int64 &c) { this->_elem += c; }
    void iadd(const cytnx_uint32 &c) { this->_elem += c; }
    void iadd(const cytnx_int32 &c) { this->_elem += c; }
    void iadd(const cytnx_uint16 &c) { this->_elem += c; }
    void iadd(const cytnx_int16 &c) { this->_elem += c; }
    void iadd(const cytnx_bool &c) { this->_elem += c; }

    void isub(const Scalar_base *c) { this->_elem -= c->to_cytnx_uint64(); }
    void isub(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void isub(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void isub(const cytnx_double &c) { this->_elem -= c; }
    void isub(const cytnx_float &c) { this->_elem -= c; }
    void isub(const cytnx_uint64 &c) { this->_elem -= c; }
    void isub(const cytnx_int64 &c) { this->_elem -= c; }
    void isub(const cytnx_uint32 &c) { this->_elem -= c; }
    void isub(const cytnx_int32 &c) { this->_elem -= c; }
    void isub(const cytnx_uint16 &c) { this->_elem -= c; }
    void isub(const cytnx_int16 &c) { this->_elem -= c; }
    void isub(const cytnx_bool &c) { this->_elem -= c; }

    void imul(const Scalar_base *c) { this->_elem *= c->to_cytnx_uint64(); }
    void imul(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void imul(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void imul(const cytnx_double &c) { this->_elem *= c; }
    void imul(const cytnx_float &c) { this->_elem *= c; }
    void imul(const cytnx_uint64 &c) { this->_elem *= c; }
    void imul(const cytnx_int64 &c) { this->_elem *= c; }
    void imul(const cytnx_uint32 &c) { this->_elem *= c; }
    void imul(const cytnx_int32 &c) { this->_elem *= c; }
    void imul(const cytnx_uint16 &c) { this->_elem *= c; }
    void imul(const cytnx_int16 &c) { this->_elem *= c; }
    void imul(const cytnx_bool &c) { this->_elem *= c; }

    void idiv(const Scalar_base *c) { this->_elem /= c->to_cytnx_uint64(); }
    void idiv(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void idiv(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void idiv(const cytnx_double &c) { this->_elem /= c; }
    void idiv(const cytnx_float &c) { this->_elem /= c; }
    void idiv(const cytnx_uint64 &c) { this->_elem /= c; }
    void idiv(const cytnx_int64 &c) { this->_elem /= c; }
    void idiv(const cytnx_uint32 &c) { this->_elem /= c; }
    void idiv(const cytnx_int32 &c) { this->_elem /= c; }
    void idiv(const cytnx_uint16 &c) { this->_elem /= c; }
    void idiv(const cytnx_int16 &c) { this->_elem /= c; }
    void idiv(const cytnx_bool &c) { this->_elem /= c; }

    void iabs() { this->_elem = std::abs(cytnx_double(this->_elem)); }
    void isqrt() { this->_elem = std::sqrt(this->_elem); }

    bool less(const Scalar_base *c) { return this->_elem < c->to_cytnx_uint64(); }
    bool less(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_double &c) { return this->_elem < c; }
    bool less(const cytnx_float &c) { return this->_elem < c; }
    bool less(const cytnx_uint64 &c) { return this->_elem < c; }
    bool less(const cytnx_int64 &c) { return this->_elem < c; }
    bool less(const cytnx_uint32 &c) { return this->_elem < c; }
    bool less(const cytnx_int32 &c) { return this->_elem < c; }
    bool less(const cytnx_uint16 &c) { return this->_elem < c; }
    bool less(const cytnx_int16 &c) { return this->_elem < c; }
    bool less(const cytnx_bool &c) { return this->_elem < c; }

    bool greater(const Scalar_base *c) { return this->_elem > c->to_cytnx_uint64(); }
    bool greater(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_double &c) { return this->_elem > c; }
    bool greater(const cytnx_float &c) { return this->_elem > c; }
    bool greater(const cytnx_uint64 &c) { return this->_elem > c; }
    bool greater(const cytnx_int64 &c) { return this->_elem > c; }
    bool greater(const cytnx_uint32 &c) { return this->_elem > c; }
    bool greater(const cytnx_int32 &c) { return this->_elem > c; }
    bool greater(const cytnx_uint16 &c) { return this->_elem > c; }
    bool greater(const cytnx_int16 &c) { return this->_elem > c; }
    bool greater(const cytnx_bool &c) { return this->_elem > c; }

    bool eq(const Scalar_base *c) { return this->_elem == c->to_cytnx_uint64(); }
    // bool approx_eq(const Scalar_base *c, const cytnx_double &tol = 1e-8) {
    //   return std::abs((cytnx_double)this->_elem - (cytnx_double)c->to_cytnx_uint64()) <= tol;
    // }

    void conj_() { return; }
    Scalar_base *get_real() { return this->copy(); }
    Scalar_base *get_imag() {
      cytnx_error_msg(true, "[ERROR] real type Scalar does not have imag part!%s", "\n");
      return nullptr;
    }

    void set_maxval() { this->_elem = std::numeric_limits<cytnx_uint64>::max(); }
    void set_minval() { this->_elem = std::numeric_limits<cytnx_uint64>::min(); }

    void *get_raw_address() const { return (void *)(&this->_elem); }
    Scalar_base *astype(const unsigned int &dtype) {
      Scalar_base *tmp = __ScII.UScIInit[dtype]();
      tmp->assign_selftype(this->_elem);
      return tmp;
    }
    Scalar_base *copy() const {
      Uint64Scalar *tmp = new Uint64Scalar(this->_elem);
      return tmp;
    };
    void print(std::ostream &os) const { os << "< " << this->_elem << " >"; };
  };
  class Int32Scalar : public Scalar_base {
   public:
    cytnx_int32 _elem;

    Int32Scalar() : _elem(0) { this->_dtype = Type.Int32; };
    Int32Scalar(const cytnx_int32 &in) : _elem(0) {
      this->_dtype = Type.Int32;
      this->_elem = in;
    }

    cytnx_float to_cytnx_float() const { return this->_elem; };
    cytnx_double to_cytnx_double() const { return this->_elem; };
    cytnx_complex64 to_cytnx_complex64() const { return this->_elem; };
    cytnx_complex128 to_cytnx_complex128() const { return this->_elem; };
    cytnx_int64 to_cytnx_int64() const { return this->_elem; };
    cytnx_uint64 to_cytnx_uint64() const { return this->_elem; };
    cytnx_int32 to_cytnx_int32() const { return this->_elem; };
    cytnx_uint32 to_cytnx_uint32() const { return this->_elem; };
    cytnx_int16 to_cytnx_int16() const { return this->_elem; };
    cytnx_uint16 to_cytnx_uint16() const { return this->_elem; };
    cytnx_bool to_cytnx_bool() const { return this->_elem; };

    void assign_selftype(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot convert complex to real%s", "\n");
    }
    void assign_selftype(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot convert complex to real%s", "\n");
    }
    void assign_selftype(const cytnx_double &c) { this->_elem = c; }
    void assign_selftype(const cytnx_float &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint64 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int64 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint32 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int32 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint16 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int16 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_bool &c) { this->_elem = c; }

    void iadd(const Scalar_base *c) { this->_elem += c->to_cytnx_int32(); }
    void iadd(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void iadd(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void iadd(const cytnx_double &c) { this->_elem += c; }
    void iadd(const cytnx_float &c) { this->_elem += c; }
    void iadd(const cytnx_uint64 &c) { this->_elem += c; }
    void iadd(const cytnx_int64 &c) { this->_elem += c; }
    void iadd(const cytnx_uint32 &c) { this->_elem += c; }
    void iadd(const cytnx_int32 &c) { this->_elem += c; }
    void iadd(const cytnx_uint16 &c) { this->_elem += c; }
    void iadd(const cytnx_int16 &c) { this->_elem += c; }
    void iadd(const cytnx_bool &c) { this->_elem += c; }

    void isub(const Scalar_base *c) { this->_elem -= c->to_cytnx_int32(); }
    void isub(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void isub(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void isub(const cytnx_double &c) { this->_elem -= c; }
    void isub(const cytnx_float &c) { this->_elem -= c; }
    void isub(const cytnx_uint64 &c) { this->_elem -= c; }
    void isub(const cytnx_int64 &c) { this->_elem -= c; }
    void isub(const cytnx_uint32 &c) { this->_elem -= c; }
    void isub(const cytnx_int32 &c) { this->_elem -= c; }
    void isub(const cytnx_uint16 &c) { this->_elem -= c; }
    void isub(const cytnx_int16 &c) { this->_elem -= c; }
    void isub(const cytnx_bool &c) { this->_elem -= c; }

    void imul(const Scalar_base *c) { this->_elem *= c->to_cytnx_int32(); }
    void imul(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void imul(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void imul(const cytnx_double &c) { this->_elem *= c; }
    void imul(const cytnx_float &c) { this->_elem *= c; }
    void imul(const cytnx_uint64 &c) { this->_elem *= c; }
    void imul(const cytnx_int64 &c) { this->_elem *= c; }
    void imul(const cytnx_uint32 &c) { this->_elem *= c; }
    void imul(const cytnx_int32 &c) { this->_elem *= c; }
    void imul(const cytnx_uint16 &c) { this->_elem *= c; }
    void imul(const cytnx_int16 &c) { this->_elem *= c; }
    void imul(const cytnx_bool &c) { this->_elem *= c; }

    void idiv(const Scalar_base *c) { this->_elem /= c->to_cytnx_int32(); }
    void idiv(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void idiv(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void idiv(const cytnx_double &c) { this->_elem /= c; }
    void idiv(const cytnx_float &c) { this->_elem /= c; }
    void idiv(const cytnx_uint64 &c) { this->_elem /= c; }
    void idiv(const cytnx_int64 &c) { this->_elem /= c; }
    void idiv(const cytnx_uint32 &c) { this->_elem /= c; }
    void idiv(const cytnx_int32 &c) { this->_elem /= c; }
    void idiv(const cytnx_uint16 &c) { this->_elem /= c; }
    void idiv(const cytnx_int16 &c) { this->_elem /= c; }
    void idiv(const cytnx_bool &c) { this->_elem /= c; }

    void iabs() { this->_elem = std::abs(this->_elem); }
    void isqrt() { this->_elem = std::sqrt(this->_elem); }

    bool less(const Scalar_base *c) { return this->_elem < c->to_cytnx_int32(); }
    bool less(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_double &c) { return this->_elem < c; }
    bool less(const cytnx_float &c) { return this->_elem < c; }
    bool less(const cytnx_uint64 &c) { return this->_elem < c; }
    bool less(const cytnx_int64 &c) { return this->_elem < c; }
    bool less(const cytnx_uint32 &c) { return this->_elem < c; }
    bool less(const cytnx_int32 &c) { return this->_elem < c; }
    bool less(const cytnx_uint16 &c) { return this->_elem < c; }
    bool less(const cytnx_int16 &c) { return this->_elem < c; }
    bool less(const cytnx_bool &c) { return this->_elem < c; }

    bool greater(const Scalar_base *c) { return this->_elem > c->to_cytnx_int32(); }
    bool greater(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_double &c) { return this->_elem > c; }
    bool greater(const cytnx_float &c) { return this->_elem > c; }
    bool greater(const cytnx_uint64 &c) { return this->_elem > c; }
    bool greater(const cytnx_int64 &c) { return this->_elem > c; }
    bool greater(const cytnx_uint32 &c) { return this->_elem > c; }
    bool greater(const cytnx_int32 &c) { return this->_elem > c; }
    bool greater(const cytnx_uint16 &c) { return this->_elem > c; }
    bool greater(const cytnx_int16 &c) { return this->_elem > c; }
    bool greater(const cytnx_bool &c) { return this->_elem > c; }

    bool eq(const Scalar_base *c) { return this->_elem == c->to_cytnx_int32(); }
    // bool approx_eq(const Scalar_base *c, const cytnx_double &tol = 1e-8) {
    //   return std::abs((cytnx_double)this->_elem - (cytnx_double)c->to_cytnx_int32()) <= tol;
    // }

    void conj_() { return; }
    Scalar_base *get_real() { return this->copy(); }
    Scalar_base *get_imag() {
      cytnx_error_msg(true, "[ERROR] real type Scalar does not have imag part!%s", "\n");
      return nullptr;
    }

    void set_maxval() { this->_elem = std::numeric_limits<cytnx_int32>::max(); }
    void set_minval() { this->_elem = std::numeric_limits<cytnx_int32>::min(); }

    void *get_raw_address() const { return (void *)(&this->_elem); }

    Scalar_base *astype(const unsigned int &dtype) {
      Scalar_base *tmp = __ScII.UScIInit[dtype]();
      tmp->assign_selftype(this->_elem);
      return tmp;
    }
    Scalar_base *copy() const {
      Int32Scalar *tmp = new Int32Scalar(this->_elem);
      return tmp;
    };
    void print(std::ostream &os) const { os << "< " << this->_elem << " >"; };
  };
  class Uint32Scalar : public Scalar_base {
   public:
    cytnx_uint32 _elem;

    Uint32Scalar() : _elem(0) { this->_dtype = Type.Uint32; };
    Uint32Scalar(const cytnx_uint32 &in) : _elem(0) {
      this->_dtype = Type.Uint32;
      this->_elem = in;
    }

    cytnx_float to_cytnx_float() const { return this->_elem; };
    cytnx_double to_cytnx_double() const { return this->_elem; };
    cytnx_complex64 to_cytnx_complex64() const { return this->_elem; };
    cytnx_complex128 to_cytnx_complex128() const { return this->_elem; };
    cytnx_int64 to_cytnx_int64() const { return this->_elem; };
    cytnx_uint64 to_cytnx_uint64() const { return this->_elem; };
    cytnx_int32 to_cytnx_int32() const { return this->_elem; };
    cytnx_uint32 to_cytnx_uint32() const { return this->_elem; };
    cytnx_int16 to_cytnx_int16() const { return this->_elem; };
    cytnx_uint16 to_cytnx_uint16() const { return this->_elem; };
    cytnx_bool to_cytnx_bool() const { return this->_elem; };

    void assign_selftype(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot convert complex to real%s", "\n");
    }
    void assign_selftype(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot convert complex to real%s", "\n");
    }
    void assign_selftype(const cytnx_double &c) { this->_elem = c; }
    void assign_selftype(const cytnx_float &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint64 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int64 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint32 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int32 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint16 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int16 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_bool &c) { this->_elem = c; }

    void iadd(const Scalar_base *c) { this->_elem += c->to_cytnx_uint32(); }
    void iadd(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void iadd(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void iadd(const cytnx_double &c) { this->_elem += c; }
    void iadd(const cytnx_float &c) { this->_elem += c; }
    void iadd(const cytnx_uint64 &c) { this->_elem += c; }
    void iadd(const cytnx_int64 &c) { this->_elem += c; }
    void iadd(const cytnx_uint32 &c) { this->_elem += c; }
    void iadd(const cytnx_int32 &c) { this->_elem += c; }
    void iadd(const cytnx_uint16 &c) { this->_elem += c; }
    void iadd(const cytnx_int16 &c) { this->_elem += c; }
    void iadd(const cytnx_bool &c) { this->_elem += c; }

    void isub(const Scalar_base *c) { this->_elem -= c->to_cytnx_uint32(); }
    void isub(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void isub(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void isub(const cytnx_double &c) { this->_elem -= c; }
    void isub(const cytnx_float &c) { this->_elem -= c; }
    void isub(const cytnx_uint64 &c) { this->_elem -= c; }
    void isub(const cytnx_int64 &c) { this->_elem -= c; }
    void isub(const cytnx_uint32 &c) { this->_elem -= c; }
    void isub(const cytnx_int32 &c) { this->_elem -= c; }
    void isub(const cytnx_uint16 &c) { this->_elem -= c; }
    void isub(const cytnx_int16 &c) { this->_elem -= c; }
    void isub(const cytnx_bool &c) { this->_elem -= c; }

    void imul(const Scalar_base *c) { this->_elem *= c->to_cytnx_uint32(); }
    void imul(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void imul(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void imul(const cytnx_double &c) { this->_elem *= c; }
    void imul(const cytnx_float &c) { this->_elem *= c; }
    void imul(const cytnx_uint64 &c) { this->_elem *= c; }
    void imul(const cytnx_int64 &c) { this->_elem *= c; }
    void imul(const cytnx_uint32 &c) { this->_elem *= c; }
    void imul(const cytnx_int32 &c) { this->_elem *= c; }
    void imul(const cytnx_uint16 &c) { this->_elem *= c; }
    void imul(const cytnx_int16 &c) { this->_elem *= c; }
    void imul(const cytnx_bool &c) { this->_elem *= c; }

    void idiv(const Scalar_base *c) { this->_elem /= c->to_cytnx_uint32(); }
    void idiv(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void idiv(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void idiv(const cytnx_double &c) { this->_elem /= c; }
    void idiv(const cytnx_float &c) { this->_elem /= c; }
    void idiv(const cytnx_uint64 &c) { this->_elem /= c; }
    void idiv(const cytnx_int64 &c) { this->_elem /= c; }
    void idiv(const cytnx_uint32 &c) { this->_elem /= c; }
    void idiv(const cytnx_int32 &c) { this->_elem /= c; }
    void idiv(const cytnx_uint16 &c) { this->_elem /= c; }
    void idiv(const cytnx_int16 &c) { this->_elem /= c; }
    void idiv(const cytnx_bool &c) { this->_elem /= c; }

    void iabs() { this->_elem = std::abs(cytnx_double(this->_elem)); }
    void isqrt() { this->_elem = std::sqrt(this->_elem); }

    bool less(const Scalar_base *c) { return this->_elem < c->to_cytnx_uint32(); }
    bool less(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_double &c) { return this->_elem < c; }
    bool less(const cytnx_float &c) { return this->_elem < c; }
    bool less(const cytnx_uint64 &c) { return this->_elem < c; }
    bool less(const cytnx_int64 &c) { return this->_elem < c; }
    bool less(const cytnx_uint32 &c) { return this->_elem < c; }
    bool less(const cytnx_int32 &c) { return this->_elem < c; }
    bool less(const cytnx_uint16 &c) { return this->_elem < c; }
    bool less(const cytnx_int16 &c) { return this->_elem < c; }
    bool less(const cytnx_bool &c) { return this->_elem < c; }

    bool greater(const Scalar_base *c) { return this->_elem > c->to_cytnx_uint32(); }
    bool greater(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_double &c) { return this->_elem > c; }
    bool greater(const cytnx_float &c) { return this->_elem > c; }
    bool greater(const cytnx_uint64 &c) { return this->_elem > c; }
    bool greater(const cytnx_int64 &c) { return this->_elem > c; }
    bool greater(const cytnx_uint32 &c) { return this->_elem > c; }
    bool greater(const cytnx_int32 &c) { return this->_elem > c; }
    bool greater(const cytnx_uint16 &c) { return this->_elem > c; }
    bool greater(const cytnx_int16 &c) { return this->_elem > c; }
    bool greater(const cytnx_bool &c) { return this->_elem > c; }

    bool eq(const Scalar_base *c) { return this->_elem == c->to_cytnx_uint32(); }
    // bool approx_eq(const Scalar_base *c, const cytnx_double &tol = 1e-8) {
    //   return std::abs((cytnx_double)this->_elem - (cytnx_double)c->to_cytnx_uint32()) <= tol;
    // }

    void conj_() { return; }
    Scalar_base *get_real() { return this->copy(); }
    Scalar_base *get_imag() {
      cytnx_error_msg(true, "[ERROR] real type Scalar does not have imag part!%s", "\n");
      return nullptr;
    }

    void set_maxval() { this->_elem = std::numeric_limits<cytnx_uint32>::max(); }
    void set_minval() { this->_elem = std::numeric_limits<cytnx_uint32>::min(); }

    void *get_raw_address() const { return (void *)(&this->_elem); }
    Scalar_base *astype(const unsigned int &dtype) {
      Scalar_base *tmp = __ScII.UScIInit[dtype]();
      tmp->assign_selftype(this->_elem);
      return tmp;
    }
    Scalar_base *copy() const {
      Uint32Scalar *tmp = new Uint32Scalar(this->_elem);
      return tmp;
    };
    void print(std::ostream &os) const { os << "< " << this->_elem << " >"; };
  };
  class Int16Scalar : public Scalar_base {
   public:
    cytnx_int16 _elem;

    Int16Scalar() : _elem(0) { this->_dtype = Type.Int16; };
    Int16Scalar(const cytnx_int16 &in) : _elem(0) {
      this->_dtype = Type.Int16;
      this->_elem = in;
    }

    cytnx_float to_cytnx_float() const { return this->_elem; };
    cytnx_double to_cytnx_double() const { return this->_elem; };
    cytnx_complex64 to_cytnx_complex64() const { return this->_elem; };
    cytnx_complex128 to_cytnx_complex128() const { return this->_elem; };
    cytnx_int64 to_cytnx_int64() const { return this->_elem; };
    cytnx_uint64 to_cytnx_uint64() const { return this->_elem; };
    cytnx_int32 to_cytnx_int32() const { return this->_elem; };
    cytnx_uint32 to_cytnx_uint32() const { return this->_elem; };
    cytnx_int16 to_cytnx_int16() const { return this->_elem; };
    cytnx_uint16 to_cytnx_uint16() const { return this->_elem; };
    cytnx_bool to_cytnx_bool() const { return this->_elem; };

    void assign_selftype(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot convert complex to real%s", "\n");
    }
    void assign_selftype(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot convert complex to real%s", "\n");
    }
    void assign_selftype(const cytnx_double &c) { this->_elem = c; }
    void assign_selftype(const cytnx_float &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint64 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int64 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint32 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int32 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint16 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int16 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_bool &c) { this->_elem = c; }

    void iadd(const Scalar_base *c) { this->_elem += c->to_cytnx_int16(); }
    void iadd(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void iadd(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void iadd(const cytnx_double &c) { this->_elem += c; }
    void iadd(const cytnx_float &c) { this->_elem += c; }
    void iadd(const cytnx_uint64 &c) { this->_elem += c; }
    void iadd(const cytnx_int64 &c) { this->_elem += c; }
    void iadd(const cytnx_uint32 &c) { this->_elem += c; }
    void iadd(const cytnx_int32 &c) { this->_elem += c; }
    void iadd(const cytnx_uint16 &c) { this->_elem += c; }
    void iadd(const cytnx_int16 &c) { this->_elem += c; }
    void iadd(const cytnx_bool &c) { this->_elem += c; }

    void isub(const Scalar_base *c) { this->_elem -= c->to_cytnx_int16(); }
    void isub(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void isub(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void isub(const cytnx_double &c) { this->_elem -= c; }
    void isub(const cytnx_float &c) { this->_elem -= c; }
    void isub(const cytnx_uint64 &c) { this->_elem -= c; }
    void isub(const cytnx_int64 &c) { this->_elem -= c; }
    void isub(const cytnx_uint32 &c) { this->_elem -= c; }
    void isub(const cytnx_int32 &c) { this->_elem -= c; }
    void isub(const cytnx_uint16 &c) { this->_elem -= c; }
    void isub(const cytnx_int16 &c) { this->_elem -= c; }
    void isub(const cytnx_bool &c) { this->_elem -= c; }

    void imul(const Scalar_base *c) { this->_elem *= c->to_cytnx_int16(); }
    void imul(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void imul(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void imul(const cytnx_double &c) { this->_elem *= c; }
    void imul(const cytnx_float &c) { this->_elem *= c; }
    void imul(const cytnx_uint64 &c) { this->_elem *= c; }
    void imul(const cytnx_int64 &c) { this->_elem *= c; }
    void imul(const cytnx_uint32 &c) { this->_elem *= c; }
    void imul(const cytnx_int32 &c) { this->_elem *= c; }
    void imul(const cytnx_uint16 &c) { this->_elem *= c; }
    void imul(const cytnx_int16 &c) { this->_elem *= c; }
    void imul(const cytnx_bool &c) { this->_elem *= c; }

    void idiv(const Scalar_base *c) { this->_elem /= c->to_cytnx_int16(); }
    void idiv(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void idiv(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void idiv(const cytnx_double &c) { this->_elem /= c; }
    void idiv(const cytnx_float &c) { this->_elem /= c; }
    void idiv(const cytnx_uint64 &c) { this->_elem /= c; }
    void idiv(const cytnx_int64 &c) { this->_elem /= c; }
    void idiv(const cytnx_uint32 &c) { this->_elem /= c; }
    void idiv(const cytnx_int32 &c) { this->_elem /= c; }
    void idiv(const cytnx_uint16 &c) { this->_elem /= c; }
    void idiv(const cytnx_int16 &c) { this->_elem /= c; }
    void idiv(const cytnx_bool &c) { this->_elem /= c; }

    void iabs() { this->_elem = std::abs(this->_elem); }
    void isqrt() { this->_elem = std::sqrt(this->_elem); }

    bool less(const Scalar_base *c) { return this->_elem < c->to_cytnx_int16(); }
    bool less(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_double &c) { return this->_elem < c; }
    bool less(const cytnx_float &c) { return this->_elem < c; }
    bool less(const cytnx_uint64 &c) { return this->_elem < c; }
    bool less(const cytnx_int64 &c) { return this->_elem < c; }
    bool less(const cytnx_uint32 &c) { return this->_elem < c; }
    bool less(const cytnx_int32 &c) { return this->_elem < c; }
    bool less(const cytnx_uint16 &c) { return this->_elem < c; }
    bool less(const cytnx_int16 &c) { return this->_elem < c; }
    bool less(const cytnx_bool &c) { return this->_elem < c; }

    bool greater(const Scalar_base *c) { return this->_elem > c->to_cytnx_int16(); }
    bool greater(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_double &c) { return this->_elem > c; }
    bool greater(const cytnx_float &c) { return this->_elem > c; }
    bool greater(const cytnx_uint64 &c) { return this->_elem > c; }
    bool greater(const cytnx_int64 &c) { return this->_elem > c; }
    bool greater(const cytnx_uint32 &c) { return this->_elem > c; }
    bool greater(const cytnx_int32 &c) { return this->_elem > c; }
    bool greater(const cytnx_uint16 &c) { return this->_elem > c; }
    bool greater(const cytnx_int16 &c) { return this->_elem > c; }
    bool greater(const cytnx_bool &c) { return this->_elem > c; }

    bool eq(const Scalar_base *c) { return this->_elem == c->to_cytnx_int16(); }
    // bool approx_eq(const Scalar_base *c, const cytnx_double &tol = 1e-8) {
    //   return std::abs((cytnx_double)this->_elem - (cytnx_double)c->to_cytnx_int16()) <= tol;
    // }

    void conj_() { return; }
    Scalar_base *get_real() { return this->copy(); }
    Scalar_base *get_imag() {
      cytnx_error_msg(true, "[ERROR] real type Scalar does not have imag part!%s", "\n");
      return nullptr;
    }

    void set_maxval() { this->_elem = std::numeric_limits<cytnx_int16>::max(); }
    void set_minval() { this->_elem = std::numeric_limits<cytnx_int16>::min(); }

    void *get_raw_address() const { return (void *)(&this->_elem); }
    Scalar_base *astype(const unsigned int &dtype) {
      Scalar_base *tmp = __ScII.UScIInit[dtype]();
      tmp->assign_selftype(this->_elem);
      return tmp;
    }
    Scalar_base *copy() const {
      Int16Scalar *tmp = new Int16Scalar(this->_elem);
      return tmp;
    };
    void print(std::ostream &os) const { os << "< " << this->_elem << " >"; };
  };
  class Uint16Scalar : public Scalar_base {
   public:
    cytnx_uint16 _elem;

    Uint16Scalar() : _elem(0) { this->_dtype = Type.Uint16; };
    Uint16Scalar(const cytnx_uint16 &in) : _elem(0) {
      this->_dtype = Type.Uint16;
      this->_elem = in;
    }

    cytnx_float to_cytnx_float() const { return this->_elem; };
    cytnx_double to_cytnx_double() const { return this->_elem; };
    cytnx_complex64 to_cytnx_complex64() const { return this->_elem; };
    cytnx_complex128 to_cytnx_complex128() const { return this->_elem; };
    cytnx_int64 to_cytnx_int64() const { return this->_elem; };
    cytnx_uint64 to_cytnx_uint64() const { return this->_elem; };
    cytnx_int32 to_cytnx_int32() const { return this->_elem; };
    cytnx_uint32 to_cytnx_uint32() const { return this->_elem; };
    cytnx_int16 to_cytnx_int16() const { return this->_elem; };
    cytnx_uint16 to_cytnx_uint16() const { return this->_elem; };
    cytnx_bool to_cytnx_bool() const { return this->_elem; };

    void assign_selftype(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot convert complex to real%s", "\n");
    }
    void assign_selftype(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot convert complex to real%s", "\n");
    }
    void assign_selftype(const cytnx_double &c) { this->_elem = c; }
    void assign_selftype(const cytnx_float &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint64 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int64 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint32 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int32 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint16 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int16 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_bool &c) { this->_elem = c; }

    void iadd(const Scalar_base *c) { this->_elem += c->to_cytnx_uint16(); }
    void iadd(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void iadd(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void iadd(const cytnx_double &c) { this->_elem += c; }
    void iadd(const cytnx_float &c) { this->_elem += c; }
    void iadd(const cytnx_uint64 &c) { this->_elem += c; }
    void iadd(const cytnx_int64 &c) { this->_elem += c; }
    void iadd(const cytnx_uint32 &c) { this->_elem += c; }
    void iadd(const cytnx_int32 &c) { this->_elem += c; }
    void iadd(const cytnx_uint16 &c) { this->_elem += c; }
    void iadd(const cytnx_int16 &c) { this->_elem += c; }
    void iadd(const cytnx_bool &c) { this->_elem += c; }

    void isub(const Scalar_base *c) { this->_elem -= c->to_cytnx_uint16(); }
    void isub(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void isub(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void isub(const cytnx_double &c) { this->_elem -= c; }
    void isub(const cytnx_float &c) { this->_elem -= c; }
    void isub(const cytnx_uint64 &c) { this->_elem -= c; }
    void isub(const cytnx_int64 &c) { this->_elem -= c; }
    void isub(const cytnx_uint32 &c) { this->_elem -= c; }
    void isub(const cytnx_int32 &c) { this->_elem -= c; }
    void isub(const cytnx_uint16 &c) { this->_elem -= c; }
    void isub(const cytnx_int16 &c) { this->_elem -= c; }
    void isub(const cytnx_bool &c) { this->_elem -= c; }

    void imul(const Scalar_base *c) { this->_elem *= c->to_cytnx_uint16(); }
    void imul(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void imul(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void imul(const cytnx_double &c) { this->_elem *= c; }
    void imul(const cytnx_float &c) { this->_elem *= c; }
    void imul(const cytnx_uint64 &c) { this->_elem *= c; }
    void imul(const cytnx_int64 &c) { this->_elem *= c; }
    void imul(const cytnx_uint32 &c) { this->_elem *= c; }
    void imul(const cytnx_int32 &c) { this->_elem *= c; }
    void imul(const cytnx_uint16 &c) { this->_elem *= c; }
    void imul(const cytnx_int16 &c) { this->_elem *= c; }
    void imul(const cytnx_bool &c) { this->_elem *= c; }

    void idiv(const Scalar_base *c) { this->_elem /= c->to_cytnx_uint16(); }
    void idiv(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void idiv(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void idiv(const cytnx_double &c) { this->_elem /= c; }
    void idiv(const cytnx_float &c) { this->_elem /= c; }
    void idiv(const cytnx_uint64 &c) { this->_elem /= c; }
    void idiv(const cytnx_int64 &c) { this->_elem /= c; }
    void idiv(const cytnx_uint32 &c) { this->_elem /= c; }
    void idiv(const cytnx_int32 &c) { this->_elem /= c; }
    void idiv(const cytnx_uint16 &c) { this->_elem /= c; }
    void idiv(const cytnx_int16 &c) { this->_elem /= c; }
    void idiv(const cytnx_bool &c) { this->_elem /= c; }

    void iabs() { this->_elem = std::abs(this->_elem); }
    void isqrt() { this->_elem = std::sqrt(this->_elem); }

    bool less(const Scalar_base *c) { return this->_elem < c->to_cytnx_uint16(); }
    bool less(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_double &c) { return this->_elem < c; }
    bool less(const cytnx_float &c) { return this->_elem < c; }
    bool less(const cytnx_uint64 &c) { return this->_elem < c; }
    bool less(const cytnx_int64 &c) { return this->_elem < c; }
    bool less(const cytnx_uint32 &c) { return this->_elem < c; }
    bool less(const cytnx_int32 &c) { return this->_elem < c; }
    bool less(const cytnx_uint16 &c) { return this->_elem < c; }
    bool less(const cytnx_int16 &c) { return this->_elem < c; }
    bool less(const cytnx_bool &c) { return this->_elem < c; }

    bool greater(const Scalar_base *c) { return this->_elem > c->to_cytnx_uint16(); }
    bool greater(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_double &c) { return this->_elem > c; }
    bool greater(const cytnx_float &c) { return this->_elem > c; }
    bool greater(const cytnx_uint64 &c) { return this->_elem > c; }
    bool greater(const cytnx_int64 &c) { return this->_elem > c; }
    bool greater(const cytnx_uint32 &c) { return this->_elem > c; }
    bool greater(const cytnx_int32 &c) { return this->_elem > c; }
    bool greater(const cytnx_uint16 &c) { return this->_elem > c; }
    bool greater(const cytnx_int16 &c) { return this->_elem > c; }
    bool greater(const cytnx_bool &c) { return this->_elem > c; }

    bool eq(const Scalar_base *c) { return this->_elem == c->to_cytnx_uint16(); }
    // bool approx_eq(const Scalar_base *c, const cytnx_double &tol = 1e-8) {
    //   return std::abs((cytnx_double)this->_elem - (cytnx_double)c->to_cytnx_uint16()) <= tol;
    // }

    void conj_() { return; }
    Scalar_base *get_real() { return this->copy(); }
    Scalar_base *get_imag() {
      cytnx_error_msg(true, "[ERROR] real type Scalar does not have imag part!%s", "\n");
      return nullptr;
    }

    void set_maxval() { this->_elem = std::numeric_limits<cytnx_uint16>::max(); }
    void set_minval() { this->_elem = std::numeric_limits<cytnx_uint16>::min(); }

    void *get_raw_address() const { return (void *)(&this->_elem); }
    Scalar_base *astype(const unsigned int &dtype) {
      Scalar_base *tmp = __ScII.UScIInit[dtype]();
      tmp->assign_selftype(this->_elem);
      return tmp;
    }
    Scalar_base *copy() const {
      Uint16Scalar *tmp = new Uint16Scalar(this->_elem);
      return tmp;
    };
    void print(std::ostream &os) const { os << "< " << this->_elem << " >"; };
  };
  class BoolScalar : public Scalar_base {
   public:
    cytnx_bool _elem;

    BoolScalar() : _elem(0) { this->_dtype = Type.Bool; };
    BoolScalar(const cytnx_bool &in) : _elem(0) {
      this->_dtype = Type.Bool;
      this->_elem = in;
    }

    cytnx_float to_cytnx_float() const { return this->_elem; };
    cytnx_double to_cytnx_double() const { return this->_elem; };
    cytnx_complex64 to_cytnx_complex64() const { return this->_elem; };
    cytnx_complex128 to_cytnx_complex128() const { return this->_elem; };
    cytnx_int64 to_cytnx_int64() const { return this->_elem; };
    cytnx_uint64 to_cytnx_uint64() const { return this->_elem; };
    cytnx_int32 to_cytnx_int32() const { return this->_elem; };
    cytnx_uint32 to_cytnx_uint32() const { return this->_elem; };
    cytnx_int16 to_cytnx_int16() const { return this->_elem; };
    cytnx_uint16 to_cytnx_uint16() const { return this->_elem; };
    cytnx_bool to_cytnx_bool() const { return this->_elem; };

    void assign_selftype(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot convert complex to real%s", "\n");
    }
    void assign_selftype(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot convert complex to real%s", "\n");
    }
    void assign_selftype(const cytnx_double &c) { this->_elem = c; }
    void assign_selftype(const cytnx_float &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint64 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int64 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint32 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int32 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_uint16 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_int16 &c) { this->_elem = c; }
    void assign_selftype(const cytnx_bool &c) { this->_elem = c; }

    void iadd(const Scalar_base *c) { this->_elem += c->to_cytnx_bool(); }
    void iadd(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void iadd(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void iadd(const cytnx_double &c) { this->_elem += c; }
    void iadd(const cytnx_float &c) { this->_elem += c; }
    void iadd(const cytnx_uint64 &c) { this->_elem += c; }
    void iadd(const cytnx_int64 &c) { this->_elem += c; }
    void iadd(const cytnx_uint32 &c) { this->_elem += c; }
    void iadd(const cytnx_int32 &c) { this->_elem += c; }
    void iadd(const cytnx_uint16 &c) { this->_elem += c; }
    void iadd(const cytnx_int16 &c) { this->_elem += c; }
    void iadd(const cytnx_bool &c) { this->_elem += c; }

    void isub(const Scalar_base *c) { this->_elem -= c->to_cytnx_bool(); }
    void isub(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void isub(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void isub(const cytnx_double &c) { this->_elem -= c; }
    void isub(const cytnx_float &c) { this->_elem -= c; }
    void isub(const cytnx_uint64 &c) { this->_elem -= c; }
    void isub(const cytnx_int64 &c) { this->_elem -= c; }
    void isub(const cytnx_uint32 &c) { this->_elem -= c; }
    void isub(const cytnx_int32 &c) { this->_elem -= c; }
    void isub(const cytnx_uint16 &c) { this->_elem -= c; }
    void isub(const cytnx_int16 &c) { this->_elem -= c; }
    void isub(const cytnx_bool &c) { this->_elem -= c; }

    void imul(const Scalar_base *c) { this->_elem *= c->to_cytnx_bool(); }
    void imul(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void imul(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void imul(const cytnx_double &c) { this->_elem *= c; }
    void imul(const cytnx_float &c) { this->_elem *= c; }
    void imul(const cytnx_uint64 &c) { this->_elem *= c; }
    void imul(const cytnx_int64 &c) { this->_elem *= c; }
    void imul(const cytnx_uint32 &c) { this->_elem *= c; }
    void imul(const cytnx_int32 &c) { this->_elem *= c; }
    void imul(const cytnx_uint16 &c) { this->_elem *= c; }
    void imul(const cytnx_int16 &c) { this->_elem *= c; }
    void imul(const cytnx_bool &c) { this->_elem *= c; }

    void idiv(const Scalar_base *c) { this->_elem /= c->to_cytnx_bool(); }
    void idiv(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void idiv(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] cannot operate real and complex values%s", "\n");
    }
    void idiv(const cytnx_double &c) { this->_elem /= c; }
    void idiv(const cytnx_float &c) { this->_elem /= c; }
    void idiv(const cytnx_uint64 &c) { this->_elem /= c; }
    void idiv(const cytnx_int64 &c) { this->_elem /= c; }
    void idiv(const cytnx_uint32 &c) { this->_elem /= c; }
    void idiv(const cytnx_int32 &c) { this->_elem /= c; }
    void idiv(const cytnx_uint16 &c) { this->_elem /= c; }
    void idiv(const cytnx_int16 &c) { this->_elem /= c; }
    void idiv(const cytnx_bool &c) { this->_elem /= c; }

    void iabs() { this->_elem = std::abs(this->_elem); }
    void isqrt() { this->_elem = std::sqrt(this->_elem); }

    bool less(const Scalar_base *c) { return this->_elem < c->to_cytnx_bool(); }
    bool less(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool less(const cytnx_double &c) { return this->_elem < c; }
    bool less(const cytnx_float &c) { return this->_elem < c; }
    bool less(const cytnx_uint64 &c) { return this->_elem < c; }
    bool less(const cytnx_int64 &c) { return this->_elem < c; }
    bool less(const cytnx_uint32 &c) { return this->_elem < c; }
    bool less(const cytnx_int32 &c) { return this->_elem < c; }
    bool less(const cytnx_uint16 &c) { return this->_elem < c; }
    bool less(const cytnx_int16 &c) { return this->_elem < c; }
    bool less(const cytnx_bool &c) { return this->_elem < c; }

    bool greater(const Scalar_base *c) { return this->_elem > c->to_cytnx_bool(); }
    bool greater(const cytnx_complex128 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_complex64 &c) {
      cytnx_error_msg(true, "[ERROR] comparison not supported for complex type%s", "\n");
      return 0;
    }
    bool greater(const cytnx_double &c) { return this->_elem > c; }
    bool greater(const cytnx_float &c) { return this->_elem > c; }
    bool greater(const cytnx_uint64 &c) { return this->_elem > c; }
    bool greater(const cytnx_int64 &c) { return this->_elem > c; }
    bool greater(const cytnx_uint32 &c) { return this->_elem > c; }
    bool greater(const cytnx_int32 &c) { return this->_elem > c; }
    bool greater(const cytnx_uint16 &c) { return this->_elem > c; }
    bool greater(const cytnx_int16 &c) { return this->_elem > c; }
    bool greater(const cytnx_bool &c) { return this->_elem > c; }

    bool eq(const Scalar_base *c) { return this->_elem == c->to_cytnx_bool(); }
    // bool approx_eq(const Scalar_base *c, const cytnx_double &tol = 1e-8) {
    //   return std::abs((cytnx_double)this->_elem - (cytnx_double)c->to_cytnx_bool()) <= tol;
    // }

    void conj_() { return; }
    Scalar_base *get_real() { return this->copy(); }
    Scalar_base *get_imag() {
      cytnx_error_msg(true, "[ERROR] real type Scalar does not have imag part!%s", "\n");
      return nullptr;
    }

    void set_maxval() { this->_elem = true; }
    void set_minval() { this->_elem = false; }

    void *get_raw_address() const { return (void *)(&this->_elem); }
    Scalar_base *astype(const unsigned int &dtype) {
      Scalar_base *tmp = __ScII.UScIInit[dtype]();
      tmp->assign_selftype(this->_elem);
      return tmp;
    }
    Scalar_base *copy() const {
      BoolScalar *tmp = new BoolScalar(this->_elem);
      return tmp;
    };
    void print(std::ostream &os) const { os << "< " << this->_elem << " >"; };
  };

  ///@endcond

  /**
   * @brief A class to represent a scalar.
   * @details This class is used to represent a scalar. You can construct a Scalar by
   * a given value and a dtype (see cytnx::Type for available dtype).
   */
  class Scalar {
   public:
    ///@cond
    struct Sproxy {
      boost::intrusive_ptr<Storage_base> _insimpl;
      cytnx_uint64 _loc;
      Sproxy() {}
      Sproxy(boost::intrusive_ptr<Storage_base> _ptr, const cytnx_uint64 &idx)
          : _insimpl(_ptr), _loc(idx) {}

      Sproxy(const Sproxy &rhs) {
        this->_insimpl = rhs._insimpl;
        this->_loc = rhs._loc;
      }

      // When used to set elems:
      Sproxy &operator=(const Scalar &rc);
      Sproxy &operator=(const cytnx_complex128 &rc);
      Sproxy &operator=(const cytnx_complex64 &rc);
      Sproxy &operator=(const cytnx_double &rc);
      Sproxy &operator=(const cytnx_float &rc);
      Sproxy &operator=(const cytnx_uint64 &rc);
      Sproxy &operator=(const cytnx_int64 &rc);
      Sproxy &operator=(const cytnx_uint32 &rc);
      Sproxy &operator=(const cytnx_int32 &rc);
      Sproxy &operator=(const cytnx_uint16 &rc);
      Sproxy &operator=(const cytnx_int16 &rc);
      Sproxy &operator=(const cytnx_bool &rc);

      Sproxy &operator=(const Sproxy &rc);

      Sproxy copy() const {
        Sproxy out = *this;
        return out;
      }

      Scalar real();
      Scalar imag();
      bool exists() const;

      // When used to get elements:
      // operator Scalar() const;
    };

    Scalar_base *_impl;
    ///@endcond

    /// @brief default constructor
    Scalar() : _impl(new Scalar_base()){};

    // init!!
    /// @brief init a Scalar with a cytnx::cytnx_complex128
    Scalar(const cytnx_complex128 &in) : _impl(new Scalar_base()) { this->Init_by_number(in); }

    /// @brief init a Scalar with a cytnx::cytnx_complex64
    Scalar(const cytnx_complex64 &in) : _impl(new Scalar_base()) { this->Init_by_number(in); }

    /// @brief init a Scalar with a cytnx::cytnx_double
    Scalar(const cytnx_double &in) : _impl(new Scalar_base()) { this->Init_by_number(in); }

    /// @brief init a Scalar with a cytnx::cytnx_float
    Scalar(const cytnx_float &in) : _impl(new Scalar_base()) { this->Init_by_number(in); }

    /// @brief init a Scalar with a cytnx::cytnx_uint64
    Scalar(const cytnx_uint64 &in) : _impl(new Scalar_base()) { this->Init_by_number(in); }

    /// @brief init a Scalar with a cytnx::cytnx_int64
    Scalar(const cytnx_int64 &in) : _impl(new Scalar_base()) { this->Init_by_number(in); }

    /// @brief init a Scalar with a cytnx::cytnx_uint32
    Scalar(const cytnx_uint32 &in) : _impl(new Scalar_base()) { this->Init_by_number(in); }

    /// @brief init a Scalar with a cytnx::cytnx_int32
    Scalar(const cytnx_int32 &in) : _impl(new Scalar_base()) { this->Init_by_number(in); }

    /// @brief init a Scalar with a cytnx::cytnx_uint16
    Scalar(const cytnx_uint16 &in) : _impl(new Scalar_base()) { this->Init_by_number(in); }

    /// @brief init a Scalar with a cytnx::cytnx_int16
    Scalar(const cytnx_int16 &in) : _impl(new Scalar_base()) { this->Init_by_number(in); }

    /// @brief init a Scalar with a cytnx::cytnx_bool
    Scalar(const cytnx_bool &in) : _impl(new Scalar_base()) { this->Init_by_number(in); }

    /**
     * @brief Get the max value of the Scalar with the given \p dtype.
     * @details This function is used to get the max value of the Scalar with the given \p dtype.
     * That is, for example, if you want to get the max value of a Scalar with
     * \p dtype = cytnx::Type.Int16, then you will get the max value of a 16-bit integer 32767.
     * @param[in] dtype The data type of the Scalar.
     * @return The max value of the Scalar with the given \p dtype.
     */
    static Scalar maxval(const unsigned int &dtype) {
      Scalar out(0, dtype);
      out._impl->set_maxval();
      return out;
    }

    /**
     * @brief Get the min value of the Scalar with the given \p dtype.
     * @details This function is used to get the min value of the Scalar with the given \p dtype.
     * That is, for example, if you want to get the min value of a Scalar with
     * \p dtype = cytnx::Type.Int16, then you will get the min value of a 16-bit integer -32768.
     * @param[in] dtype The data type of the Scalar.
     * @return The min value of the Scalar with the given \p dtype.
     */
    static Scalar minval(const unsigned int &dtype) {
      Scalar out(0, dtype);
      out._impl->set_minval();
      return out;
    }

    /**
     * @brief The constructor of the Scalar class.
     * @details This constructor is used to init a Scalar with a given template value
     *  \p in and \p dtype (see cytnx::Type).
     * @param[in] in The value of the Scalar.
     * @param[in] dtype The data type of the Scalar.
     * @return A Scalar object.
     * @note The \p dtype can be any of the cytnx::Type.
     */
    template <class T>
    Scalar(const T &in, const unsigned int &dtype) : _impl(new Scalar_base()) {
      if (this->_impl != nullptr) delete this->_impl;
      this->_impl = __ScII.UScIInit[dtype]();
      this->_impl->assign_selftype(in);
    };

    /// @cond
    // move sproxy when use to get elements here.
    Scalar(const Sproxy &prox);

    //[Internal!!]
    Scalar(Scalar_base *in) { this->_impl = in; }
    /// @endcond

    // specialization of init:
    ///@cond
    void Init_by_number(const cytnx_complex128 &in) {
      if (this->_impl != nullptr) delete this->_impl;
      this->_impl = new ComplexDoubleScalar(in);
    };
    void Init_by_number(const cytnx_complex64 &in) {
      if (this->_impl != nullptr) delete this->_impl;
      this->_impl = new ComplexFloatScalar(in);
    };
    void Init_by_number(const cytnx_double &in) {
      if (this->_impl != nullptr) delete this->_impl;
      this->_impl = new DoubleScalar(in);
    }
    void Init_by_number(const cytnx_float &in) {
      if (this->_impl != nullptr) delete this->_impl;
      this->_impl = new FloatScalar(in);
    }
    void Init_by_number(const cytnx_int64 &in) {
      if (this->_impl != nullptr) delete this->_impl;
      this->_impl = new Int64Scalar(in);
    }
    void Init_by_number(const cytnx_uint64 &in) {
      if (this->_impl != nullptr) delete this->_impl;
      this->_impl = new Uint64Scalar(in);
    }
    void Init_by_number(const cytnx_int32 &in) {
      if (this->_impl != nullptr) delete this->_impl;
      this->_impl = new Int32Scalar(in);
    }
    void Init_by_number(const cytnx_uint32 &in) {
      if (this->_impl != nullptr) delete this->_impl;
      this->_impl = new Uint32Scalar(in);
    }
    void Init_by_number(const cytnx_int16 &in) {
      if (this->_impl != nullptr) delete this->_impl;
      this->_impl = new Int16Scalar(in);
    }
    void Init_by_number(const cytnx_uint16 &in) {
      if (this->_impl != nullptr) delete this->_impl;
      this->_impl = new Uint16Scalar(in);
    }
    void Init_by_number(const cytnx_bool &in) {
      if (this->_impl != nullptr) delete this->_impl;
      this->_impl = new BoolScalar(in);
    }

    // The copy constructor
    Scalar(const Scalar &rhs) : _impl(new Scalar_base()) {
      if (this->_impl != nullptr) delete this->_impl;

      this->_impl = rhs._impl->copy();
    }
    /// @endcond

    /// @brief The copy assignment of the Scalar class.
    Scalar &operator=(const Scalar &rhs) {
      if (this->_impl != nullptr) delete this->_impl;

      this->_impl = rhs._impl->copy();
      return *this;
    };

    // copy assignment [Number]:
    /** @brief The copy assignment operator of the Scalar class with a given number
     * cytnx::cytnx_complex128 \p rhs.
     */
    Scalar &operator=(const cytnx_complex128 &rhs) {
      this->Init_by_number(rhs);
      return *this;
    }

    /** @brief The copy assignment operator of the Scalar class with a given number
     * cytnx::cytnx_complex64 \p rhs.
     */
    Scalar &operator=(const cytnx_complex64 &rhs) {
      this->Init_by_number(rhs);
      return *this;
    }

    /** @brief The copy assignment operator of the Scalar class with a given number
     * cytnx::cytnx_double \p rhs.
     */
    Scalar &operator=(const cytnx_double &rhs) {
      this->Init_by_number(rhs);
      return *this;
    }

    /** @brief The copy assignment operator of the Scalar class with a given number
     * cytnx::cytnx_float \p rhs.
     */
    Scalar &operator=(const cytnx_float &rhs) {
      this->Init_by_number(rhs);
      return *this;
    }

    /** @brief The copy assignment operator of the Scalar class with a given number
     * cytnx::cytnx_uint64 \p rhs.
     */
    Scalar &operator=(const cytnx_uint64 &rhs) {
      this->Init_by_number(rhs);
      return *this;
    }

    /** @brief The copy assignment operator of the Scalar class with a given number
     * cytnx::cytnx_int64 \p rhs.
     */
    Scalar &operator=(const cytnx_int64 &rhs) {
      this->Init_by_number(rhs);
      return *this;
    }

    /** @brief The copy assignment operator of the Scalar class with a given number
     * cytnx::cytnx_uint32 \p rhs.
     */
    Scalar &operator=(const cytnx_uint32 &rhs) {
      this->Init_by_number(rhs);
      return *this;
    }

    /** @brief The copy assignment operator of the Scalar class with a given number
     * cytnx::cytnx_int32 \p rhs.
     */
    Scalar &operator=(const cytnx_int32 &rhs) {
      this->Init_by_number(rhs);
      return *this;
    }

    /** @brief The copy assignment operator of the Scalar class with a given number
     * cytnx::cytnx_uint16 \p rhs.
     */
    Scalar &operator=(const cytnx_uint16 &rhs) {
      this->Init_by_number(rhs);
      return *this;
    }

    /** @brief The copy assignment operator of the Scalar class with a given number
     * cytnx::cytnx_int16 \p rhs.
     */
    Scalar &operator=(const cytnx_int16 &rhs) {
      this->Init_by_number(rhs);
      return *this;
    }

    /** @brief The copy assignment operator of the Scalar class with a given number
     * cytnx::cytnx_bool \p rhs.
     */
    Scalar &operator=(const cytnx_bool &rhs) {
      this->Init_by_number(rhs);
      return *this;
    }

    /**
     * @brief Type conversion function.
     * @param[in] dtype The type of the output Scalar (see cytnx::Type for more details).
     * @return The converted Scalar.
     * @attention The function cannot convert from complex to real, please use
     * cytnx::Scalar::real() or cytnx::Scalar::imag() to get the real or imaginary
     * part of the Scalar instead.
     */
    Scalar astype(const unsigned int &dtype) const {
      Scalar out(this->_impl->astype(dtype));
      return out;
    }

    /**
     * @brief Get the conjugate of the Scalar. That means return \f$ c^* \f$ if
     * the Scalar is \f$ c \f$.
     * @return The conjugate of the Scalar.
     */
    Scalar conj() const {
      Scalar out = *this;
      out._impl->conj_();
      return out;
    }

    /**
     * @brief Get the imaginary part of the Scalar. That means return \f$ \Im(c) \f$ if
     * the Scalar is \f$ c \f$.
     * @return The imaginary part of the Scalar.
     */
    Scalar imag() const { return Scalar(this->_impl->get_imag()); }

    /**
     * @brief Get the real part of the Scalar. That means return \f$ \Re(c) \f$ if
     * the Scalar is \f$ c \f$.
     * @return The real part of the Scalar.
     */
    Scalar real() const { return Scalar(this->_impl->get_real()); }
    // Scalar& set_imag(const Scalar &in){   return *this;}
    // Scalar& set_real(const Scalar &in){   return *this;}

    /**
     * @brief Get the dtype of the Scalar (see cytnx::Type for more details).
     */
    int dtype() const { return this->_impl->_dtype; }

    // print()
    /**
     * @brief Print the Scalar to the standard output.
     */
    void print() const {
      this->_impl->print(std::cout);
      std::cout << std::string(" Scalar dtype: [") << Type.getname(this->_impl->_dtype)
                << std::string("]") << std::endl;
    }

    // casting
    /// @brief The explicit casting operator of the Scalar class to cytnx::cytnx_double.
    explicit operator cytnx_double() const { return this->_impl->to_cytnx_double(); }

    /// @brief The explicit casting operator of the Scalar class to cytnx::cytnx_float
    explicit operator cytnx_float() const { return this->_impl->to_cytnx_float(); }

    /// @brief The explicit casting operator of the Scalar class to cytnx::cytnx_uint64.
    explicit operator cytnx_uint64() const { return this->_impl->to_cytnx_uint64(); }

    /// @brief The explicit casting operator of the Scalar class to cytnx::cytnx_int64.
    explicit operator cytnx_int64() const { return this->_impl->to_cytnx_int64(); }

    /// @brief The explicit casting operator of the Scalar class to cytnx::cytnx_uint32.
    explicit operator cytnx_uint32() const { return this->_impl->to_cytnx_uint32(); }

    /// @brief The explicit casting operator of the Scalar class to cytnx::cytnx_int32.
    explicit operator cytnx_int32() const { return this->_impl->to_cytnx_int32(); }

    /// @brief The explicit casting operator of the Scalar class to cytnx::cytnx_uint16.
    explicit operator cytnx_uint16() const { return this->_impl->to_cytnx_uint16(); }

    /// @brief The explicit casting operator of the Scalar class to cytnx::cytnx_int16.
    explicit operator cytnx_int16() const { return this->_impl->to_cytnx_int16(); }

    /// @brief The explicit casting operator of the Scalar class to cytnx::cytnx_bool.
    explicit operator cytnx_bool() const { return this->_impl->to_cytnx_bool(); }

    /// @cond
    // destructor
    ~Scalar() {
      if (this->_impl != nullptr) delete this->_impl;
    };
    /// @endcond

    // arithmetic:
    ///@brief The addition assignment operator of the Scalar class with a given number (template).
    template <class T>
    void operator+=(const T &rc) {
      this->_impl->iadd(rc);
    }

    ///@brief The addition assignment operator of the Scalar class with a given Scalar.
    void operator+=(const Scalar &rhs) { this->_impl->iadd(rhs._impl); }

    ///@brief The subtraction assignment operator of the Scalar class with a given number
    ///(template).
    template <class T>
    void operator-=(const T &rc) {
      this->_impl->isub(rc);
    }

    ///@brief The subtraction assignment operator of the Scalar class with a given Scalar.
    void operator-=(const Scalar &rhs) { this->_impl->isub(rhs._impl); }
    template <class T>

    ///@brief The multiplication assignment operator of the Scalar class with a given number
    ///(template).
    void operator*=(const T &rc) {
      this->_impl->imul(rc);
    }

    ///@brief The multiplication assignment operator of the Scalar class with a given Scalar.
    void operator*=(const Scalar &rhs) { this->_impl->imul(rhs._impl); }
    template <class T>

    /**
     * @brief The division assignment operator of the Scalar class with a given number (template).
     */
    void operator/=(const T &rc) {
      this->_impl->idiv(rc);
    }

    /**
     * @brief The division assignment operator of the Scalar class with a given Scalar.
     */
    void operator/=(const Scalar &rhs) { this->_impl->idiv(rhs._impl); }

    /// @brief Set the Scalar to absolute value. (inplace)
    void iabs() { this->_impl->iabs(); }

    /// @brief Set the Scalar to square root. (inplace)
    void isqrt() { this->_impl->isqrt(); }

    /**
     * @brief The member function to get the absolute value of the Scalar.
     * @note Compare to the iabs() function, this function will return a new Scalar object.
     * @return The absolute value of the Scalar.
     * @see iabs()
     */
    Scalar abs() const {
      Scalar out = *this;
      out._impl->iabs();
      return out.real();
    }

    /**
     * @brief The member function to get the square root of the Scalar.
     * @note Compare to the isqrt() function, this function will return a new Scalar object.
     * @return The square root of the Scalar.
     * @see isqrt()
     */
    Scalar sqrt() const {
      Scalar out = *this;
      out._impl->isqrt();
      return out;
    }

    // comparison <
    /**
     * @brief Return whether the current Scalar is less than a given template number \p rc.
     * @details That is, whether \f$ s < r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given number \p rc.
     * @see operator<(const Scalar &lhs, const Scalar &rhs)
     */
    template <class T>
    bool less(const T &rc) const {
      Scalar tmp;
      int rid = Type.cy_typeid(rc);
      if (rid < this->dtype()) {
        tmp = this->astype(rid);
        return tmp._impl->less(rc);
      } else {
        return this->_impl->less(rc);
      }
    }

    /**
     * @brief Return whether the current Scalar is less than a given Scalar \p rhs.
     * @details That is, whether \f$ s < r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given Scalar \p rhs.
     * @see operator<(const Scalar &lhs, const Scalar &rhs)
     */
    bool less(const Scalar &rhs) const {
      Scalar tmp;
      if (rhs.dtype() < this->dtype()) {
        tmp = this->astype(rhs.dtype());
        return tmp._impl->less(rhs._impl);
      } else {
        return this->_impl->less(rhs._impl);
      }
    }

    // comparison <=

    /**
     * @brief Return whether the current Scalar is less than or equal to a given template number \p
     * rc.
     * @details That is, whether \f$ s \leq r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given number \p rc.
     * @see operator<=(const Scalar &lhs, const Scalar &rhs)
     */
    template <class T>
    bool leq(const T &rc) const {
      Scalar tmp;
      int rid = Type.cy_typeid(rc);
      if (rid < this->dtype()) {
        tmp = this->astype(rid);
        return !(tmp._impl->greater(rc));
      } else {
        return !(this->_impl->greater(rc));
      }
    }

    /**
     * @brief Return whether the current Scalar is less than or equal to a given Scalar \p rhs.
     * @details That is, whether \f$ s \leq r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given Scalar \p rhs.
     * @see operator<=(const Scalar &lhs, const Scalar &rhs)
     */
    bool leq(const Scalar &rhs) const {
      Scalar tmp;
      if (rhs.dtype() < this->dtype()) {
        tmp = this->astype(rhs.dtype());
        return !(tmp._impl->greater(rhs._impl));
      } else {
        return !(this->_impl->greater(rhs._impl));
      }
    }

    // comparison >
    /**
     * @brief Return whether the current Scalar is greater than a given template number \p rc.
     * @details That is, whether \f$ s > r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given number \p rc.
     * @see operator>(const Scalar &lhs, const Scalar &rhs)
     */
    template <class T>
    bool greater(const T &rc) const {
      Scalar tmp;
      int rid = Type.cy_typeid(rc);
      if (rid < this->dtype()) {
        tmp = this->astype(rid);
        return tmp._impl->greater(rc);
      } else {
        return this->_impl->greater(rc);
      }
    }

    /**
     * @brief Return whether the current Scalar is greater than a given Scalar \p rhs.
     * @details That is, whether \f$ s > r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given Scalar \p rhs.
     * @see operator>(const Scalar &lhs, const Scalar &rhs)
     */
    bool greater(const Scalar &rhs) const {
      Scalar tmp;
      if (rhs.dtype() < this->dtype()) {
        tmp = this->astype(rhs.dtype());
        return tmp._impl->greater(rhs._impl);
      } else {
        return this->_impl->greater(rhs._impl);
      }
    }

    // comparison >=

    /**
     * @brief Return whether the current Scalar is greater than or equal to a given template number
     * \p rc.
     * @details That is, whether \f$ s \geq r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given number \p rc.
     * @see operator>=(const Scalar &lhs, const Scalar &rhs)
     */
    template <class T>
    bool geq(const T &rc) const {
      Scalar tmp;
      int rid = Type.cy_typeid(rc);
      if (rid < this->dtype()) {
        tmp = this->astype(rid);
        return !(tmp._impl->less(rc));
      } else {
        return !(this->_impl->less(rc));
      }
    }

    /**
     * @brief Return whether the current Scalar is greater than or equal to a given Scalar \p rhs.
     * @details That is, whether \f$ s \geq r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given Scalar \p rhs.
     * @see operator>=(const Scalar &lhs, const Scalar &rhs)
     */
    bool geq(const Scalar &rhs) const {
      Scalar tmp;
      if (rhs.dtype() < this->dtype()) {
        tmp = this->astype(rhs.dtype());
        return !(tmp._impl->less(rhs._impl));
      } else {
        return !(this->_impl->less(rhs._impl));
      }
    }

    // comparison ==

    /**
     * @brief Return whether the current Scalar is equal to a given template number \p rc.
     * @details That is, whether \f$ s = r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given number \p rc.
     * @see operator==(const Scalar &lhs, const Scalar &rhs)
     */
    template <class T>
    bool eq(const T &rc) const {
      Scalar tmp;
      int rid = Type.cy_typeid(rc);
      if (rid < this->dtype()) {
        tmp = this->astype(rid);
        return tmp._impl->eq(rc);
      } else {
        return this->_impl->eq(rc);
      }
    }
    // /**
    //  * @brief Return whether the current Scalar is approximately equal to a given template number
    //  \p
    //  * rc.
    //  * @details That is, whether \f$ abs(s-r)<tol \f$, where \f$ s \f$ is the current Scalar
    //  * itself, \f$ r \f$ is the given number \p rc and \p tol is the tolerance value.
    //  */
    // template <class T>
    // bool approx_eq(const T &rc, cytnx_double tol = 1e-8) const {
    //   Scalar tmp;
    //   int rid = Type.cy_typeid(rc);
    //   if (rid < this->dtype()) {
    //     tmp = this->astype(rid);
    //     return tmp._impl->approx_eq(rc, tol);
    //   } else {
    //     return this->_impl->approx_eq(rc, tol);
    //   }
    // }

    /**
     * @brief Return whether the current Scalar is equal to a given Scalar \p rhs.
     * @details That is, whether \f$ s = r \f$, where \f$ s \f$ is the current Scalar
     * itself and \f$ r \f$ is the given Scalar \p rhs.
     * @see operator==(const Scalar &lhs, const Scalar &rhs)
     */
    bool eq(const Scalar &rhs) const {
      Scalar tmp;
      if (rhs.dtype() < this->dtype()) {
        tmp = this->astype(rhs.dtype());
        return tmp._impl->eq(rhs._impl);
      } else {
        return this->_impl->eq(rhs._impl);
      }
    }
    // /**
    //  * @brief Return whether the current Scalar is approximately equal to a given Scalar \p rhs.
    //  * @details That is, whether \f$ abs(s-r)<tol \f$, where \f$ s \f$ is the current Scalar
    //  * itself, \f$ r \f$ is the given Scalar \p rhs and \p tol is the tolerance value.
    //  */
    // bool approx_eq(const Scalar &rhs, cytnx_double tol = 1e-8) const {
    //   Scalar tmp;
    //   if (rhs.dtype() < this->dtype()) {
    //     tmp = this->astype(rhs.dtype());
    //     return tmp._impl->approx_eq(rhs._impl, tol);
    //   } else {
    //     return this->_impl->approx_eq(rhs._impl, tol);
    //   }
    // }

    // radd: Scalar + c

    /**
     * @brief Return the addition of the current Scalar and a given template number \p rc.
     * @see operator+(const Scalar &lhs, const Scalar &rhs)
     */
    template <class T>
    Scalar radd(const T &rc) const {
      Scalar out;
      int rid = Type.cy_typeid(rc);
      if (this->dtype() < rid) {
        out = *this;
      } else {
        out = this->astype(rid);
      }
      out._impl->iadd(rc);
      return out;
    }

    /**
     * @brief Return the addition of the current Scalar and a given Scalar \p rhs.
     * @see operator+(const Scalar &lhs, const Scalar &rhs)
     */
    Scalar radd(const Scalar &rhs) const {
      Scalar out;
      if (this->dtype() < rhs.dtype()) {
        out = *this;
      } else {
        out = this->astype(rhs.dtype());
      }
      out._impl->iadd(rhs._impl);
      return out;
    }

    // rmul: Scalar * c

    /**
     * @brief Return the multiplication of the current Scalar and a given template number \p rc.
     * @see operator*(const Scalar &lhs, const Scalar &rhs)
     */
    template <class T>
    Scalar rmul(const T &rc) const {
      Scalar out;
      int rid = Type.cy_typeid(rc);
      if (this->dtype() < rid) {
        out = *this;
      } else {
        out = this->astype(rid);
      }
      out._impl->imul(rc);
      return out;
    }

    /**
     * @brief Return the multiplication of the current Scalar and a given Scalar \p rhs.
     * @see operator*(const Scalar &lhs, const Scalar &rhs)
     */
    Scalar rmul(const Scalar &rhs) const {
      Scalar out;
      if (this->dtype() < rhs.dtype()) {
        out = *this;
      } else {
        out = this->astype(rhs.dtype());
      }
      out._impl->imul(rhs._impl);
      return out;
    }

    // rsub: Scalar - c

    /**
     * @brief Return the subtraction of the current Scalar and a given template number \p rc.
     * @see operator-(const Scalar &lhs, const Scalar &rhs)
     */
    template <class T>
    Scalar rsub(const T &rc) const {
      Scalar out;
      int rid = Type.cy_typeid(rc);
      if (this->dtype() < rid) {
        out = *this;
      } else {
        out = this->astype(rid);
      }
      out._impl->isub(rc);
      return out;
    }

    /**
     * @brief Return the subtraction of the current Scalar and a given Scalar \p rhs.
     * @see operator-(const Scalar &lhs, const Scalar &rhs)
     */
    Scalar rsub(const Scalar &rhs) const {
      Scalar out;
      if (this->dtype() < rhs.dtype()) {
        out = *this;
      } else {
        out = this->astype(rhs.dtype());
      }
      out._impl->isub(rhs._impl);
      return out;
    }

    // rdiv: Scalar / c

    /**
     * @brief Return the division of the current Scalar and a given template number \p rc.
     * @see operator/(const Scalar &lhs, const Scalar &rhs)
     */
    template <class T>
    Scalar rdiv(const T &rc) const {
      Scalar out;
      int rid = Type.cy_typeid(rc);
      if (this->dtype() < rid) {
        out = *this;
      } else {
        out = this->astype(rid);
      }
      out._impl->idiv(rc);
      return out;
    }

    /**
     * @brief Return the division of the current Scalar and a given Scalar \p rhs.
     * @see operator/(const Scalar &lhs, const Scalar &rhs)
     */
    Scalar rdiv(const Scalar &rhs) const {
      Scalar out;
      if (this->dtype() < rhs.dtype()) {
        out = *this;
      } else {
        out = this->astype(rhs.dtype());
      }
      out._impl->idiv(rhs._impl);
      return out;
    }

    /*
    //operator:
    template<class T>
    Scalar operator+(const T &rc){
        return this->radd(rc);
    }
    template<class T>
    Scalar operator*(const T &rc){
        return this->rmul(rc);
    }
    template<class T>
    Scalar operator-(const T &rc){
        return this->rsub(rc);
    }
    template<class T>
    Scalar operator/(const T &rc){
        return this->rdiv(rc);
    }

    template<class T>
    bool operator<(const T &rc){
        return this->less(rc);
    }

    template<class T>
    bool operator>(const T &rc){
        return this->greater(rc);
    }


    template<class T>
    bool operator<=(const T &rc){
        return this->leq(rc);
    }

    template<class T>
    bool operator>=(const T &rc){
        return this->geq(rc);
    }
    */
  };

  // ladd: c + Scalar:

  /**
   * @brief The addition operator between two Scalar objects.
   * @details Return
   * \f[ l+r \f],
   * where \f$ l \f$ is the left Scalar \p lc and \f$ r \f$ is the right Scalar \p rs .
   */
  Scalar operator+(const Scalar &lc, const Scalar &rs);  //{return rs.radd(lc);};

  // lmul c * Scalar;
  /**
   * @brief The multiplication operator between two Scalar objects.
   * @details Return
   * \f[ l \cdot r \f],
   * where \f$ l \f$ is the left Scalar \p lc and \f$ r \f$ is the right Scalar \p rs .
   */
  Scalar operator*(const Scalar &lc, const Scalar &rs);  //{return rs.rmul(lc);};

  // lsub c * Scalar;
  /**
   * @brief The subtraction operator between two Scalar objects.
   * @details Return
   * \f[ l-r \f],
   * where \f$ l \f$ is the left Scalar \p lc and \f$ r \f$ is the right Scalar \p rs .
   */
  Scalar operator-(const Scalar &lc, const Scalar &rs);  //{return Scalar(lc).rsub(rs);};

  // ldiv c / Scalar;
  /**
   * @brief The division operator between two Scalar objects.
   * @details Return
   * \f[ l/r \f],
   * where \f$ l \f$ is the left Scalar \p lc and \f$ r \f$ is the right Scalar \p rs .
   */
  Scalar operator/(const Scalar &lc, const Scalar &rs);  //{return Scalar(lc).rdiv(rs);};

  // lless c < Scalar;
  /**
   * @brief The less-than operator between two Scalar objects.
   * @details Return
   * \f[ l<r \f],
   * where \f$ l \f$ is the left Scalar \p lc and \f$ r \f$ is the right Scalar \p rs .
   */
  bool operator<(const Scalar &lc, const Scalar &rs);

  // lgreater c > Scalar;
  /**
   * @brief The greater-than operator between two Scalar objects.
   * @details Return
   * \f[ l>r \f],
   * where \f$ l \f$ is the left Scalar \p lc and \f$ r \f$ is the right Scalar \p rs .
   */
  bool operator>(const Scalar &lc, const Scalar &rs);

  // lless c <= Scalar;
  /**
   * @brief The less-than-or-equal operator between two Scalar objects.
   * @details Return
   * \f[ l\leq r \f],
   * where \f$ l \f$ is the left Scalar \p lc and \f$ r \f$ is the right Scalar \p rs .
   */
  bool operator<=(const Scalar &lc, const Scalar &rs);

  // lgreater c >= Scalar;
  /**
   * @brief The greater-than-or-equal operator between two Scalar objects.
   * @details Return
   * \f[ l\geq r \f],
   * where \f$ l \f$ is the left Scalar \p lc and \f$ r \f$ is the right Scalar \p rs .
   */
  bool operator>=(const Scalar &lc, const Scalar &rs);

  // eq c == Scalar;
  /**
   * @brief The equal operator between two Scalar objects.
   * @details Return
   * \f[ l==r \f],
   * where \f$ l \f$ is the left Scalar \p lc and \f$ r \f$ is the right Scalar \p rs .
   */
  bool operator==(const Scalar &lc, const Scalar &rs);

  // abs:

  /**
   * @brief Return the absolute value of a Scalar object.
   * @details Return
   * \f[ \left|c\right| \f],
   * where \f$ c \f$ is the input Scalar \p c .
   */
  Scalar abs(const Scalar &c);

  // sqrt:
  /**
   * @brief Return the square root of a Scalar object.
   * @details Return
   * \f[ \sqrt{c} \f],
   * where \f$ c \f$ is the input Scalar \p c .
   */
  Scalar sqrt(const Scalar &c);

  // complex conversion:
  /// @brief Convert a Scalar object to a cytnx::complex128.
  cytnx_complex128 complex128(const Scalar &in);

  /// @brief Convert a Scalar object to a cytnx::complex64.
  cytnx_complex64 complex64(const Scalar &in);

  /// @cond
  std::ostream &operator<<(std::ostream &os, const Scalar &in);
  /// @endcond

}  // namespace cytnx

#endif
#endif
