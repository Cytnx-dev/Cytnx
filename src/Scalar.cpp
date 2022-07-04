#include "Scalar.hpp"
#include "Storage.hpp"

namespace cytnx {

  cytnx_complex128 complex128(const Scalar& in) { return in._impl->to_cytnx_complex128(); }

  cytnx_complex64 complex64(const Scalar& in) { return in._impl->to_cytnx_complex64(); }

  std::ostream& operator<<(std::ostream& os, const Scalar& in) {
    in._impl->print(os);
    os << std::string(" dtype: [") << Type.getname(in._impl->_dtype) << std::string("]");
    return os;
  }

  // ladd: c + Scalar:
  Scalar operator+(const Scalar& lc, const Scalar& rs) { return rs.radd(lc); };

  // lmul c * Scalar;
  Scalar operator*(const Scalar& lc, const Scalar& rs) { return rs.rmul(lc); };

  // lsub c - Scalar;
  Scalar operator-(const Scalar& lc, const Scalar& rs) { return lc.rsub(rs); };

  // ldiv c / Scalar;
  Scalar operator/(const Scalar& lc, const Scalar& rs) { return lc.rdiv(rs); };

  // lless c < Scalar;
  bool operator<(const Scalar& lc, const Scalar& rs) { return lc.less(rs); };

  // lless c > Scalar;
  bool operator>(const Scalar& lc, const Scalar& rs) { return lc.greater(rs); };

  // lless c <= Scalar;
  bool operator<=(const Scalar& lc, const Scalar& rs) { return lc.leq(rs); };

  // lless c >= Scalar;
  bool operator>=(const Scalar& lc, const Scalar& rs) { return lc.geq(rs); };

  // eq c == Scalar;
  bool operator==(const Scalar& lc, const Scalar& rs) {
    if (lc.dtype() < rs.dtype())
      return lc.geq(rs);
    else
      return rs.geq(lc);
  };

  Scalar abs(const Scalar& c) { return c.abs(); };

  Scalar sqrt(const Scalar& c) { return c.sqrt(); };

  // Scalar proxy:
  // Sproxy
  const Scalar::Sproxy& Scalar::Sproxy::operator=(const Scalar::Sproxy& rc) {
    Scalar tmp = rc._insimpl->get_item(rc._loc);
    this->_insimpl->set_item(this->_loc, tmp);
    return rc;
  }
  const Scalar::Sproxy& Scalar::Sproxy::operator=(const Scalar& rc) {
    this->_insimpl->set_item(this->_loc, rc);
    return *this;
  }
  const Scalar::Sproxy& Scalar::Sproxy::operator=(const cytnx_complex128& rc) {
    this->_insimpl->set_item(this->_loc, rc);
    return *this;
  }
  const Scalar::Sproxy& Scalar::Sproxy::operator=(const cytnx_complex64& rc) {
    this->_insimpl->set_item(this->_loc, rc);
    return *this;
  }
  const Scalar::Sproxy& Scalar::Sproxy::operator=(const cytnx_double& rc) {
    this->_insimpl->set_item(this->_loc, rc);
    return *this;
  }
  const Scalar::Sproxy& Scalar::Sproxy::operator=(const cytnx_float& rc) {
    this->_insimpl->set_item(this->_loc, rc);
    return *this;
  }
  const Scalar::Sproxy& Scalar::Sproxy::operator=(const cytnx_uint64& rc) {
    this->_insimpl->set_item(this->_loc, rc);
    return *this;
  }
  const Scalar::Sproxy& Scalar::Sproxy::operator=(const cytnx_int64& rc) {
    this->_insimpl->set_item(this->_loc, rc);
    return *this;
  }
  const Scalar::Sproxy& Scalar::Sproxy::operator=(const cytnx_uint32& rc) {
    this->_insimpl->set_item(this->_loc, rc);
    return *this;
  }
  const Scalar::Sproxy& Scalar::Sproxy::operator=(const cytnx_int32& rc) {
    this->_insimpl->set_item(this->_loc, rc);
    return *this;
  }
  const Scalar::Sproxy& Scalar::Sproxy::operator=(const cytnx_uint16& rc) {
    this->_insimpl->set_item(this->_loc, rc);
    return *this;
  }
  const Scalar::Sproxy& Scalar::Sproxy::operator=(const cytnx_int16& rc) {
    this->_insimpl->set_item(this->_loc, rc);
    return *this;
  }
  const Scalar::Sproxy& Scalar::Sproxy::operator=(const cytnx_bool& rc) {
    this->_insimpl->set_item(this->_loc, rc);
    return *this;
  }

  Scalar Scalar::Sproxy::real() { return Scalar(*this).real(); }
  Scalar Scalar::Sproxy::imag() { return Scalar(*this).imag(); }

  Scalar::Scalar(const Sproxy& prox) : _impl(new Scalar_base()) {
    if (this->_impl != nullptr) {
      delete this->_impl;
    }
    this->_impl = prox._insimpl->get_item(prox._loc)._impl->copy();
  }

  // Storage Init interface.
  //=============================
  Scalar_base* ScIInit_cd() {
    Scalar_base* out = new ComplexDoubleScalar();
    return out;
  }
  Scalar_base* ScIInit_cf() {
    Scalar_base* out = new ComplexFloatScalar();
    return out;
  }
  Scalar_base* ScIInit_d() {
    Scalar_base* out = new DoubleScalar();
    return out;
  }
  Scalar_base* ScIInit_f() {
    Scalar_base* out = new FloatScalar();
    return out;
  }
  Scalar_base* ScIInit_u64() {
    Scalar_base* out = new Uint64Scalar();
    return out;
  }
  Scalar_base* ScIInit_i64() {
    Scalar_base* out = new Int64Scalar();
    return out;
  }
  Scalar_base* ScIInit_u32() {
    Scalar_base* out = new Uint32Scalar();
    return out;
  }
  Scalar_base* ScIInit_i32() {
    Scalar_base* out = new Int32Scalar();
    return out;
  }
  Scalar_base* ScIInit_u16() {
    Scalar_base* out = new Uint16Scalar();
    return out;
  }
  Scalar_base* ScIInit_i16() {
    Scalar_base* out = new Int16Scalar();
    return out;
  }
  Scalar_base* ScIInit_b() {
    Scalar_base* out = new BoolScalar();
    return out;
  }
  Scalar_init_interface::Scalar_init_interface() {
    UScIInit.resize(N_Type);
    UScIInit[this->Double] = ScIInit_d;
    UScIInit[this->Float] = ScIInit_f;
    UScIInit[this->ComplexDouble] = ScIInit_cd;
    UScIInit[this->ComplexFloat] = ScIInit_cf;
    UScIInit[this->Uint64] = ScIInit_u64;
    UScIInit[this->Int64] = ScIInit_i64;
    UScIInit[this->Uint32] = ScIInit_u32;
    UScIInit[this->Int32] = ScIInit_i32;
    UScIInit[this->Uint16] = ScIInit_u16;
    UScIInit[this->Int16] = ScIInit_i16;
    UScIInit[this->Bool] = ScIInit_b;
  }

  Scalar_init_interface __ScII;

}  // namespace cytnx
