#include "Type.hpp"
#include "utils/complex_arithmetic.hpp"
namespace cytnx {

#ifdef UNI_ICPC

#else
  cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_complex64 &rn) {
    return cytnx_complex128(ln.real() + rn.real(), ln.imag() + rn.imag());
  }
  cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_double &rn) {
    return cytnx_complex128(ln.real() + rn, ln.imag());
  }
  cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_float &rn) {
    return cytnx_complex128(ln.real() + rn, ln.imag());
  }
  cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_uint64 &rn) {
    return cytnx_complex128(ln.real() + rn, ln.imag());
  }
  cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_uint32 &rn) {
    return cytnx_complex128(ln.real() + rn, ln.imag());
  }
  cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_int64 &rn) {
    return cytnx_complex128(ln.real() + rn, ln.imag());
  }
  cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_int32 &rn) {
    return cytnx_complex128(ln.real() + rn, ln.imag());
  }
  cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_uint16 &rn) {
    return cytnx_complex128(ln.real() + rn, ln.imag());
  }
  cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_int16 &rn) {
    return cytnx_complex128(ln.real() + rn, ln.imag());
  }
  cytnx_complex128 operator+(const cytnx_complex128 &ln, const cytnx_bool &rn) {
    return cytnx_complex128(ln.real() + cytnx_double(rn), ln.imag());
  }

  //-----------------------------
  cytnx_complex128 operator+(const cytnx_complex64 &ln, const cytnx_complex128 &rn) {
    return cytnx_complex128(ln.real() + rn.real(), ln.imag() + rn.imag());
  }

  cytnx_complex64 operator+(const cytnx_complex64 &ln, const cytnx_double &rn) {
    return cytnx_complex64(ln.real() + rn, ln.imag());
  }
  cytnx_complex64 operator+(const cytnx_complex64 &ln, const cytnx_float &rn) {
    return cytnx_complex64(ln.real() + rn, ln.imag());
  }
  cytnx_complex64 operator+(const cytnx_complex64 &ln, const cytnx_uint64 &rn) {
    return cytnx_complex64(ln.real() + rn, ln.imag());
  }
  cytnx_complex64 operator+(const cytnx_complex64 &ln, const cytnx_uint32 &rn) {
    return cytnx_complex64(ln.real() + rn, ln.imag());
  }
  cytnx_complex64 operator+(const cytnx_complex64 &ln, const cytnx_int64 &rn) {
    return cytnx_complex64(ln.real() + rn, ln.imag());
  }
  cytnx_complex64 operator+(const cytnx_complex64 &ln, const cytnx_int32 &rn) {
    return cytnx_complex64(ln.real() + rn, ln.imag());
  }
  cytnx_complex64 operator+(const cytnx_complex64 &ln, const cytnx_uint16 &rn) {
    return cytnx_complex64(ln.real() + rn, ln.imag());
  }
  cytnx_complex64 operator+(const cytnx_complex64 &ln, const cytnx_int16 &rn) {
    return cytnx_complex64(ln.real() + rn, ln.imag());
  }
  cytnx_complex64 operator+(const cytnx_complex64 &ln, const cytnx_bool &rn) {
    return cytnx_complex64(ln.real() + cytnx_float(rn), ln.imag());
  }
  //-----------------------
  // cytnx_complex128 operator+(const cytnx_complex128 &rn,const cytnx_complex128 &ln);
  // cytnx_complex128 operator+(const cytnx_complex64 &rn,const cytnx_complex128 &ln);
  cytnx_complex128 operator+(const cytnx_double &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(ln.real() + rn, ln.imag());
  }
  cytnx_complex128 operator+(const cytnx_float &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(ln.real() + rn, ln.imag());
  }
  cytnx_complex128 operator+(const cytnx_uint64 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(ln.real() + rn, ln.imag());
  }
  cytnx_complex128 operator+(const cytnx_uint32 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(ln.real() + rn, ln.imag());
  }
  cytnx_complex128 operator+(const cytnx_int64 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(ln.real() + rn, ln.imag());
  }
  cytnx_complex128 operator+(const cytnx_int32 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(ln.real() + rn, ln.imag());
  }
  cytnx_complex128 operator+(const cytnx_uint16 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(ln.real() + rn, ln.imag());
  }
  cytnx_complex128 operator+(const cytnx_int16 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(ln.real() + rn, ln.imag());
  }
  cytnx_complex128 operator+(const cytnx_bool &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(ln.real() + cytnx_double(rn), ln.imag());
  }

  //----------------------

  // cytnx_complex128 operator+(const cytnx_complex128 &rn,const cytnx_complex64 &ln);
  // cytnx_complex64 operator+(const cytnx_complex64 &rn,const cytnx_complex64 &ln);
  cytnx_complex64 operator+(const cytnx_double &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(ln.real() + rn, ln.imag());
  }
  cytnx_complex64 operator+(const cytnx_float &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(ln.real() + rn, ln.imag());
  }
  cytnx_complex64 operator+(const cytnx_uint64 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(ln.real() + rn, ln.imag());
  }
  cytnx_complex64 operator+(const cytnx_uint32 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(ln.real() + rn, ln.imag());
  }
  cytnx_complex64 operator+(const cytnx_int64 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(ln.real() + rn, ln.imag());
  }
  cytnx_complex64 operator+(const cytnx_int32 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(ln.real() + rn, ln.imag());
  }
  cytnx_complex64 operator+(const cytnx_uint16 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(ln.real() + rn, ln.imag());
  }
  cytnx_complex64 operator+(const cytnx_int16 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(ln.real() + rn, ln.imag());
  }
  cytnx_complex64 operator+(const cytnx_bool &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(ln.real() + cytnx_float(rn), ln.imag());
  }
  //===================================

  // cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_complex128 &rn);
  cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_complex64 &rn) {
    return cytnx_complex128(ln.real() - rn.real(), ln.imag() - rn.imag());
  }
  cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_double &rn) {
    return cytnx_complex128(ln.real() - rn, ln.imag());
  }
  cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_float &rn) {
    return cytnx_complex128(ln.real() - rn, ln.imag());
  }
  cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_uint64 &rn) {
    return cytnx_complex128(ln.real() - rn, ln.imag());
  }
  cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_uint32 &rn) {
    return cytnx_complex128(ln.real() - rn, ln.imag());
  }
  cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_int64 &rn) {
    return cytnx_complex128(ln.real() - rn, ln.imag());
  }
  cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_int32 &rn) {
    return cytnx_complex128(ln.real() - rn, ln.imag());
  }
  cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_uint16 &rn) {
    return cytnx_complex128(ln.real() - rn, ln.imag());
  }
  cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_int16 &rn) {
    return cytnx_complex128(ln.real() - rn, ln.imag());
  }
  cytnx_complex128 operator-(const cytnx_complex128 &ln, const cytnx_bool &rn) {
    return cytnx_complex128(ln.real() - cytnx_double(rn), ln.imag());
  }

  cytnx_complex128 operator-(const cytnx_complex64 &ln, const cytnx_complex128 &rn) {
    return cytnx_complex128(ln.real() - rn.real(), ln.imag() - rn.imag());
  }
  // cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_complex64 &rn);
  cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_double &rn) {
    return cytnx_complex64(ln.real() - rn, ln.imag());
  }
  cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_float &rn) {
    return cytnx_complex64(ln.real() - rn, ln.imag());
  }
  cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_uint64 &rn) {
    return cytnx_complex64(ln.real() - rn, ln.imag());
  }
  cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_uint32 &rn) {
    return cytnx_complex64(ln.real() - rn, ln.imag());
  }
  cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_int64 &rn) {
    return cytnx_complex64(ln.real() - rn, ln.imag());
  }
  cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_int32 &rn) {
    return cytnx_complex64(ln.real() - rn, ln.imag());
  }
  cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_uint16 &rn) {
    return cytnx_complex64(ln.real() - rn, ln.imag());
  }
  cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_int16 &rn) {
    return cytnx_complex64(ln.real() - rn, ln.imag());
  }
  cytnx_complex64 operator-(const cytnx_complex64 &ln, const cytnx_bool &rn) {
    return cytnx_complex64(ln.real() - cytnx_float(rn), ln.imag());
  }

  //------------

  // cytnx_complex128 operator-(const cytnx_complex128 &rn,const cytnx_complex128 &ln);
  // cytnx_complex128 operator-(const cytnx_complex64 &rn,const cytnx_complex128 &ln);
  cytnx_complex128 operator-(const cytnx_double &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(rn - ln.real(), -ln.imag());
  }
  cytnx_complex128 operator-(const cytnx_float &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(rn - ln.real(), -ln.imag());
  }
  cytnx_complex128 operator-(const cytnx_uint64 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(rn - ln.real(), -ln.imag());
  }
  cytnx_complex128 operator-(const cytnx_uint32 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(rn - ln.real(), -ln.imag());
  }
  cytnx_complex128 operator-(const cytnx_int64 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(rn - ln.real(), -ln.imag());
  }
  cytnx_complex128 operator-(const cytnx_int32 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(rn - ln.real(), -ln.imag());
  }
  cytnx_complex128 operator-(const cytnx_uint16 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(rn - ln.real(), -ln.imag());
  }
  cytnx_complex128 operator-(const cytnx_int16 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(rn - ln.real(), -ln.imag());
  }
  cytnx_complex128 operator-(const cytnx_bool &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(cytnx_double(rn) - ln.real(), -ln.imag());
  }

  //----------------

  // cytnx_complex128 operator-(const cytnx_complex128 &rn,const cytnx_complex64 &ln);
  // cytnx_complex64 operator-(const cytnx_complex64 &rn,const cytnx_complex64 &ln);
  cytnx_complex64 operator-(const cytnx_double &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(rn - ln.real(), -ln.imag());
  }
  cytnx_complex64 operator-(const cytnx_float &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(rn - ln.real(), -ln.imag());
  }
  cytnx_complex64 operator-(const cytnx_uint64 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(rn - ln.real(), -ln.imag());
  }
  cytnx_complex64 operator-(const cytnx_uint32 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(rn - ln.real(), -ln.imag());
  }
  cytnx_complex64 operator-(const cytnx_int64 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(rn - ln.real(), -ln.imag());
  }
  cytnx_complex64 operator-(const cytnx_int32 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(rn - ln.real(), -ln.imag());
  }
  cytnx_complex64 operator-(const cytnx_uint16 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(rn - ln.real(), -ln.imag());
  }
  cytnx_complex64 operator-(const cytnx_int16 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(rn - ln.real(), -ln.imag());
  }
  cytnx_complex64 operator-(const cytnx_bool &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(cytnx_float(rn) - ln.real(), -ln.imag());
  }

  //=============================

  // cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_complex128 &rn);
  cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_complex64 &rn) {
    return ln * cytnx_complex128(rn.real(), rn.imag());
  }
  cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_double &rn) {
    return cytnx_complex128(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_float &rn) {
    return cytnx_complex128(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_uint64 &rn) {
    return cytnx_complex128(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_uint32 &rn) {
    return cytnx_complex128(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_int64 &rn) {
    return cytnx_complex128(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_int32 &rn) {
    return cytnx_complex128(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_uint16 &rn) {
    return cytnx_complex128(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_int16 &rn) {
    return cytnx_complex128(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex128 operator*(const cytnx_complex128 &ln, const cytnx_bool &rn) {
    return cytnx_complex128(ln.real() * cytnx_double(rn), ln.imag() * cytnx_double(rn));
  }

  cytnx_complex128 operator*(const cytnx_complex64 &ln, const cytnx_complex128 &rn) {
    return rn * cytnx_complex128(ln.real(), ln.imag());
  }
  // cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_complex64 &rn);
  cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_double &rn) {
    return cytnx_complex64(ln.real() * rn, ln.imag() * rn);
  }

  cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_float &rn) {
    return cytnx_complex64(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_uint64 &rn) {
    return cytnx_complex64(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_uint32 &rn) {
    return cytnx_complex64(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_int64 &rn) {
    return cytnx_complex64(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_int32 &rn) {
    return cytnx_complex64(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_uint16 &rn) {
    return cytnx_complex64(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_int16 &rn) {
    return cytnx_complex64(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex64 operator*(const cytnx_complex64 &ln, const cytnx_bool &rn) {
    return cytnx_complex64(ln.real() * cytnx_float(rn), ln.imag() * cytnx_float(rn));
  }

  // cytnx_complex128 operator*(const cytnx_complex128 &rn,const cytnx_complex128 &ln);
  // cytnx_complex128 operator*(const cytnx_complex64 &rn,const cytnx_complex128 &ln);
  cytnx_complex128 operator*(const cytnx_double &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex128 operator*(const cytnx_float &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex128 operator*(const cytnx_uint64 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex128 operator*(const cytnx_uint32 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex128 operator*(const cytnx_int64 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex128 operator*(const cytnx_int32 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex128 operator*(const cytnx_uint16 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex128 operator*(const cytnx_int16 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex128 operator*(const cytnx_bool &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(ln.real() * cytnx_double(rn), ln.imag() * cytnx_double(rn));
  }

  // cytnx_complex128 operator*(const cytnx_complex128 &rn,const cytnx_complex64 &ln);
  // cytnx_complex64 operator*(const cytnx_complex64 &rn,const cytnx_complex64 &ln);
  cytnx_complex64 operator*(const cytnx_double &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex64 operator*(const cytnx_float &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex64 operator*(const cytnx_uint64 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex64 operator*(const cytnx_uint32 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex64 operator*(const cytnx_int64 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex64 operator*(const cytnx_int32 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex64 operator*(const cytnx_uint16 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex64 operator*(const cytnx_int16 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(ln.real() * rn, ln.imag() * rn);
  }
  cytnx_complex64 operator*(const cytnx_bool &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(ln.real() * cytnx_float(rn), ln.imag() * cytnx_float(rn));
  }
  //-------------------------------

  // cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_complex128 &rn);
  cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_complex64 &rn) {
    return ln / cytnx_complex128(rn);
  }
  cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_double &rn) {
    // std::cout << "[arithmetic call] cd/d" << std::endl;
    return cytnx_complex128(ln.real() / rn, ln.imag() / rn);
  }
  cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_float &rn) {
    return cytnx_complex128(ln.real() / rn, ln.imag() / rn);
  }
  cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_uint64 &rn) {
    return cytnx_complex128(ln.real() / rn, ln.imag() / rn);
  }
  cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_uint32 &rn) {
    return cytnx_complex128(ln.real() / rn, ln.imag() / rn);
  }
  cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_int64 &rn) {
    return cytnx_complex128(ln.real() / rn, ln.imag() / rn);
  }
  cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_int32 &rn) {
    return cytnx_complex128(ln.real() / rn, ln.imag() / rn);
  }
  cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_uint16 &rn) {
    return cytnx_complex128(ln.real() / rn, ln.imag() / rn);
  }
  cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_int16 &rn) {
    return cytnx_complex128(ln.real() / rn, ln.imag() / rn);
  }
  cytnx_complex128 operator/(const cytnx_complex128 &ln, const cytnx_bool &rn) {
    return cytnx_complex128(ln.real() / cytnx_double(rn), ln.imag() / cytnx_double(rn));
  }

  cytnx_complex128 operator/(const cytnx_complex64 &ln, const cytnx_complex128 &rn) {
    return cytnx_complex128(ln) / rn;
  }
  // cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_complex64 &rn);
  cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_double &rn) {
    return cytnx_complex64(ln.real() / rn, ln.imag() / rn);
  }
  cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_float &rn) {
    return cytnx_complex64(ln.real() / rn, ln.imag() / rn);
  }
  cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_uint64 &rn) {
    return cytnx_complex64(ln.real() / rn, ln.imag() / rn);
  }
  cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_uint32 &rn) {
    return cytnx_complex64(ln.real() / rn, ln.imag() / rn);
  }
  cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_int64 &rn) {
    return cytnx_complex64(ln.real() / rn, ln.imag() / rn);
  }
  cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_int32 &rn) {
    return cytnx_complex64(ln.real() / rn, ln.imag() / rn);
  }
  cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_uint16 &rn) {
    return cytnx_complex64(ln.real() / rn, ln.imag() / rn);
  }
  cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_int16 &rn) {
    return cytnx_complex64(ln.real() / rn, ln.imag() / rn);
  }
  cytnx_complex64 operator/(const cytnx_complex64 &ln, const cytnx_bool &rn) {
    return cytnx_complex64(ln.real() / cytnx_float(rn), ln.imag() / cytnx_float(rn));
  }

  // cytnx_complex128 operator/(const cytnx_complex128 &rn,const cytnx_complex128 &ln);
  // cytnx_complex128 operator/(const cytnx_complex64 &rn,const cytnx_complex128 &ln);
  cytnx_complex128 operator/(const cytnx_double &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(rn, 0) / ln;
  }
  cytnx_complex128 operator/(const cytnx_float &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(rn, 0) / ln;
  }
  cytnx_complex128 operator/(const cytnx_uint64 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(rn, 0) / ln;
  }
  cytnx_complex128 operator/(const cytnx_uint32 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(rn, 0) / ln;
  }
  cytnx_complex128 operator/(const cytnx_int64 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(rn, 0) / ln;
  }
  cytnx_complex128 operator/(const cytnx_int32 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(rn, 0) / ln;
  }
  cytnx_complex128 operator/(const cytnx_uint16 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(rn, 0) / ln;
  }
  cytnx_complex128 operator/(const cytnx_int16 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(rn, 0) / ln;
  }
  cytnx_complex128 operator/(const cytnx_bool &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(rn, 0) / ln;
  }

  // cytnx_complex128 operator/(const cytnx_complex128 &rn,const cytnx_complex64 &ln);
  // cytnx_complex64 operator/(const cytnx_complex64 &rn,const cytnx_complex64 &ln);
  cytnx_complex64 operator/(const cytnx_double &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(rn, 0) / ln;
  }
  cytnx_complex64 operator/(const cytnx_float &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(rn, 0) / ln;
  }
  cytnx_complex64 operator/(const cytnx_uint64 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(rn, 0) / ln;
  }
  cytnx_complex64 operator/(const cytnx_uint32 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(rn, 0) / ln;
  }
  cytnx_complex64 operator/(const cytnx_int64 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(rn, 0) / ln;
  }
  cytnx_complex64 operator/(const cytnx_int32 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(rn, 0) / ln;
  }
  cytnx_complex64 operator/(const cytnx_uint16 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(rn, 0) / ln;
  }
  cytnx_complex64 operator/(const cytnx_int16 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(rn, 0) / ln;
  }
  cytnx_complex64 operator/(const cytnx_bool &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(rn, 0) / ln;
  }

  //-----------------------------
  bool operator==(const cytnx_complex128 &ln, const cytnx_complex64 &rn) {
    return cytnx_complex128(rn) == ln;
  }
  bool operator==(const cytnx_complex128 &ln, const cytnx_double &rn) {
    return cytnx_complex128(rn, 0) == ln;
  }
  bool operator==(const cytnx_complex128 &ln, const cytnx_float &rn) {
    return cytnx_complex128(rn, 0) == ln;
  }
  bool operator==(const cytnx_complex128 &ln, const cytnx_uint64 &rn) {
    return cytnx_complex128(rn, 0) == ln;
  }
  bool operator==(const cytnx_complex128 &ln, const cytnx_uint32 &rn) {
    return cytnx_complex128(rn, 0) == ln;
  }
  bool operator==(const cytnx_complex128 &ln, const cytnx_int64 &rn) {
    return cytnx_complex128(rn, 0) == ln;
  }
  bool operator==(const cytnx_complex128 &ln, const cytnx_int32 &rn) {
    return cytnx_complex128(rn, 0) == ln;
  }
  bool operator==(const cytnx_complex128 &ln, const cytnx_uint16 &rn) {
    return cytnx_complex128(rn, 0) == ln;
  }
  bool operator==(const cytnx_complex128 &ln, const cytnx_int16 &rn) {
    return cytnx_complex128(rn, 0) == ln;
  }
  bool operator==(const cytnx_complex128 &ln, const cytnx_bool &rn) {
    return cytnx_complex128(rn, 0) == ln;
  }

  //-----------------------------
  bool operator==(const cytnx_complex64 &ln, const cytnx_complex128 &rn) {
    return rn == cytnx_complex128(ln);
  }

  bool operator==(const cytnx_complex64 &ln, const cytnx_double &rn) {
    return cytnx_complex64(rn, 0) == ln;
  }
  bool operator==(const cytnx_complex64 &ln, const cytnx_float &rn) {
    return cytnx_complex64(rn, 0) == ln;
  }
  bool operator==(const cytnx_complex64 &ln, const cytnx_uint64 &rn) {
    return cytnx_complex64(rn, 0) == ln;
  }
  bool operator==(const cytnx_complex64 &ln, const cytnx_uint32 &rn) {
    return cytnx_complex64(rn, 0) == ln;
  }
  bool operator==(const cytnx_complex64 &ln, const cytnx_int64 &rn) {
    return cytnx_complex64(rn, 0) == ln;
  }
  bool operator==(const cytnx_complex64 &ln, const cytnx_int32 &rn) {
    return cytnx_complex64(rn, 0) == ln;
  }
  bool operator==(const cytnx_complex64 &ln, const cytnx_uint16 &rn) {
    return cytnx_complex64(rn, 0) == ln;
  }
  bool operator==(const cytnx_complex64 &ln, const cytnx_int16 &rn) {
    return cytnx_complex64(rn, 0) == ln;
  }
  bool operator==(const cytnx_complex64 &ln, const cytnx_bool &rn) {
    return cytnx_complex64(rn, 0) == ln;
  }
  //-----------------------
  // bool operator==(const cytnx_complex128 &rn,const cytnx_complex128 &ln);
  // bool operator==(const cytnx_complex64 &rn,const cytnx_complex128 &ln);
  bool operator==(const cytnx_double &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(rn, 0) == ln;
  }
  bool operator==(const cytnx_float &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(rn, 0) == ln;
  }
  bool operator==(const cytnx_uint64 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(rn, 0) == ln;
  }
  bool operator==(const cytnx_uint32 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(rn, 0) == ln;
  }
  bool operator==(const cytnx_int64 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(rn, 0) == ln;
  }
  bool operator==(const cytnx_int32 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(rn, 0) == ln;
  }
  bool operator==(const cytnx_uint16 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(rn, 0) == ln;
  }
  bool operator==(const cytnx_int16 &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(rn, 0) == ln;
  }
  bool operator==(const cytnx_bool &rn, const cytnx_complex128 &ln) {
    return cytnx_complex128(rn, 0) == ln;
  }
  //----------------------

  // bool operator==(const cytnx_complex128 &rn,const cytnx_complex64 &ln);
  // bool operator==(const cytnx_complex64 &rn,const cytnx_complex64 &ln);
  bool operator==(const cytnx_double &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(rn, 0) == ln;
  }
  bool operator==(const cytnx_float &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(rn, 0) == ln;
  }
  bool operator==(const cytnx_uint64 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(rn, 0) == ln;
  }
  bool operator==(const cytnx_uint32 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(rn, 0) == ln;
  }
  bool operator==(const cytnx_int64 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(rn, 0) == ln;
  }
  bool operator==(const cytnx_int32 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(rn, 0) == ln;
  }
  bool operator==(const cytnx_uint16 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(rn, 0) == ln;
  }
  bool operator==(const cytnx_int16 &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(rn, 0) == ln;
  }
  bool operator==(const cytnx_bool &rn, const cytnx_complex64 &ln) {
    return cytnx_complex64(rn, 0) == ln;
  }
#endif

}  // namespace cytnx
