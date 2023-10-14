#include "backend_torch/Scalar.hpp"

using namespace std;
namespace cytnx {
  cytnx_complex128 complex128(const Scalar& in) { return complex<double>(in.toComplexDouble()); }

  cytnx_complex64 complex64(const Scalar& in) { return complex<float>(in.toComplexFloat()); }

  std::ostream& operator<<(std::ostream& os, const Scalar& in) {
    in.print_elem(os);
    os << std::string(" dtype: [") << Type.getname(in._dtype) << std::string("]");
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
    // if (lc.dtype() < rs.dtype())
    //   return lc.geq(rs);
    // else
    //   return rs.geq(lc);
    return lc.eq(rs);
  };

  Scalar abs(const Scalar& c) { return c.abs(); };

  Scalar sqrt(const Scalar& c) { return c.sqrt(); };

  at::Tensor StorageImpl2Tensor(const c10::intrusive_ptr<c10::StorageImpl>& impl, int dtype) {
    c10::Storage sto = c10::Storage(impl);

    if (dtype == Type.Bool) {
      at::TensorImpl tnimpl =
        at::TensorImpl(std::move(sto), c10::DispatchKeySet::FULL, caffe2::TypeMeta::Make<bool>());
      return at::Tensor(
        c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::reclaim(&tnimpl));
    } else if (dtype == Type.ComplexDouble) {
      at::TensorImpl tnimpl = at::TensorImpl(std::move(sto), c10::DispatchKeySet::FULL,
                                             caffe2::TypeMeta::Make<c10::complex<double>>());
      return at::Tensor(
        c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::reclaim(&tnimpl));
    } else if (dtype == Type.ComplexFloat) {
      at::TensorImpl tnimpl = at::TensorImpl(std::move(sto), c10::DispatchKeySet::FULL,
                                             caffe2::TypeMeta::Make<c10::complex<float>>());
      return at::Tensor(
        c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::reclaim(&tnimpl));
    } else if (dtype == Type.Int64) {
      at::TensorImpl tnimpl = at::TensorImpl(std::move(sto), c10::DispatchKeySet::FULL,
                                             caffe2::TypeMeta::Make<int64_t>());
      return at::Tensor(
        c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::reclaim(&tnimpl));
    } else if (dtype == Type.Int32) {
      at::TensorImpl tnimpl = at::TensorImpl(std::move(sto), c10::DispatchKeySet::FULL,
                                             caffe2::TypeMeta::Make<int32_t>());
      return at::Tensor(
        c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::reclaim(&tnimpl));
    } else if (dtype == Type.Int16) {
      at::TensorImpl tnimpl = at::TensorImpl(std::move(sto), c10::DispatchKeySet::FULL,
                                             caffe2::TypeMeta::Make<int16_t>());
      return at::Tensor(
        c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::reclaim(&tnimpl));
    } else if (dtype == Type.Double) {
      at::TensorImpl tnimpl =
        at::TensorImpl(std::move(sto), c10::DispatchKeySet::FULL, caffe2::TypeMeta::Make<double>());
      return at::Tensor(
        c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::reclaim(&tnimpl));
    } else if (dtype == Type.Float) {
      at::TensorImpl tnimpl =
        at::TensorImpl(std::move(sto), c10::DispatchKeySet::FULL, caffe2::TypeMeta::Make<float>());
      return at::Tensor(
        c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::reclaim(&tnimpl));
    } else if (dtype == Type.Float) {
      at::TensorImpl tnimpl =
        at::TensorImpl(std::move(sto), c10::DispatchKeySet::FULL, caffe2::TypeMeta::Make<float>());
      return at::Tensor(
        c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::reclaim(&tnimpl));
    }
    cytnx_error_msg(true, "[ERROR] StorageImpl2Tensor: unsupported dtype%s", "\n");
    return at::Tensor();
  }

  // Scalar proxy:
  // Sproxy
  Scalar::Sproxy& Scalar::Sproxy::operator=(const Scalar::Sproxy& rc) {
    if (this->_insimpl.get() == 0) {
      this->_insimpl = rc._insimpl;
      this->_loc = rc._loc;
      return *this;
    } else {
      if ((rc._insimpl == this->_insimpl) && (rc._loc == this->_loc)) {
        return *this;
      } else {
        at::Tensor tnthis = StorageImpl2Tensor(this->_insimpl, this->_dtype);
        at::Tensor tnrc = StorageImpl2Tensor(rc._insimpl, rc._dtype);
        tnthis[this->_loc] = tnrc[rc._loc];
        return *this;
      }
    }
  }
  Scalar::Sproxy& Scalar::Sproxy::operator=(const Scalar& rc) {
    at::Tensor tnthis = StorageImpl2Tensor(this->_insimpl, this->_dtype);
    tnthis[this->_loc] = rc;
    return *this;
  }
  Scalar::Sproxy& Scalar::Sproxy::operator=(const cytnx_complex128& rc) {
    at::Tensor tnthis = StorageImpl2Tensor(this->_insimpl, this->_dtype);
    tnthis[this->_loc] = c10::complex<double>(rc);
    return *this;
  }
  Scalar::Sproxy& Scalar::Sproxy::operator=(const cytnx_complex64& rc) {
    at::Tensor tnthis = StorageImpl2Tensor(this->_insimpl, this->_dtype);
    tnthis[this->_loc] = c10::complex<float>(rc);
    return *this;
  }
  Scalar::Sproxy& Scalar::Sproxy::operator=(const cytnx_double& rc) {
    at::Tensor tnthis = StorageImpl2Tensor(this->_insimpl, this->_dtype);
    tnthis[this->_loc] = rc;
    return *this;
  }
  Scalar::Sproxy& Scalar::Sproxy::operator=(const cytnx_float& rc) {
    at::Tensor tnthis = StorageImpl2Tensor(this->_insimpl, this->_dtype);
    tnthis[this->_loc] = rc;
    return *this;
  }
  Scalar::Sproxy& Scalar::Sproxy::operator=(const cytnx_uint64& rc) {
    cytnx_error_msg(true,
                    "[ERROR] invalid dtype for scalar operator=, pytorch backend doesn't support "
                    "unsigned type %s",
                    "\n");
    return *this;
  }
  Scalar::Sproxy& Scalar::Sproxy::operator=(const cytnx_int64& rc) {
    at::Tensor tnthis = StorageImpl2Tensor(this->_insimpl, this->_dtype);
    tnthis[this->_loc] = rc;
    return *this;
  }
  Scalar::Sproxy& Scalar::Sproxy::operator=(const cytnx_uint32& rc) {
    cytnx_error_msg(true,
                    "[ERROR] invalid dtype for scalar operator=, pytorch backend doesn't support "
                    "unsigned type %s",
                    "\n");
    return *this;
  }
  Scalar::Sproxy& Scalar::Sproxy::operator=(const cytnx_int32& rc) {
    at::Tensor tnthis = StorageImpl2Tensor(this->_insimpl, this->_dtype);
    tnthis[this->_loc] = rc;
    return *this;
  }
  Scalar::Sproxy& Scalar::Sproxy::operator=(const cytnx_uint16& rc) {
    cytnx_error_msg(true,
                    "[ERROR] invalid dtype for scalar operator=, pytorch backend doesn't support "
                    "unsigned type %s",
                    "\n");
    return *this;
  }
  Scalar::Sproxy& Scalar::Sproxy::operator=(const cytnx_int16& rc) {
    at::Tensor tnthis = StorageImpl2Tensor(this->_insimpl, this->_dtype);
    tnthis[this->_loc] = rc;
    return *this;
  }
  Scalar::Sproxy& Scalar::Sproxy::operator=(const cytnx_bool& rc) {
    at::Tensor tnthis = StorageImpl2Tensor(this->_insimpl, this->_dtype);
    tnthis[this->_loc] = rc;
    return *this;
  }

  bool Scalar::Sproxy::exists() const { return this->_dtype != Type.Void; };

  Scalar Scalar::Sproxy::real() { return Scalar(*this).real(); }
  Scalar Scalar::Sproxy::imag() { return Scalar(*this).imag(); }

  Scalar::Scalar(const Sproxy& prox) {
    switch (prox._dtype) {
      case Type.ComplexDouble:
        *this = StorageImpl2Tensor(prox._insimpl, prox._dtype)[prox._loc].item();
        break;
      case Type.ComplexFloat:
        *this = StorageImpl2Tensor(prox._insimpl, prox._dtype)[prox._loc].item();
        break;
      case Type.Double:
        *this = StorageImpl2Tensor(prox._insimpl, prox._dtype)[prox._loc].item();
        break;
      case Type.Float:
        *this = StorageImpl2Tensor(prox._insimpl, prox._dtype)[prox._loc].item();
        break;
      case Type.Int64:
        *this = StorageImpl2Tensor(prox._insimpl, prox._dtype)[prox._loc].item();
        break;
      case Type.Uint64:
        cytnx_error_msg(true, "[ERROR] no support for unsigned dtype for torch backend %s", "\n");
        break;
      case Type.Int32:
        *this = StorageImpl2Tensor(prox._insimpl, prox._dtype)[prox._loc].item();
        break;
      case Type.Uint32:
        cytnx_error_msg(true, "[ERROR] no support for unsigned dtype for torch backend %s", "\n");
        break;
      case Type.Int16:
        *this = StorageImpl2Tensor(prox._insimpl, prox._dtype)[prox._loc].item();
        break;
      case Type.Uint16:
        *this = StorageImpl2Tensor(prox._insimpl, prox._dtype)[prox._loc].item();
        break;
      case Type.Bool:
        *this = StorageImpl2Tensor(prox._insimpl, prox._dtype)[prox._loc].item();
        break;
      default:
        cytnx_error_msg(true, "[ERROR] invalid dtype for torch backend %s", "\n");
        break;
    }
  }

  //   // Storage Init interface.
  //   //=============================
  //   inline Scalar_base* ScIInit_cd() {
  //     Scalar_base* out = new ComplexDoubleScalar();
  //     return out;
  //   }
  //   inline Scalar_base* ScIInit_cf() {
  //     Scalar_base* out = new ComplexFloatScalar();
  //     return out;
  //   }
  //   inline Scalar_base* ScIInit_d() {
  //     Scalar_base* out = new DoubleScalar();
  //     return out;
  //   }
  //   inline Scalar_base* ScIInit_f() {
  //     Scalar_base* out = new FloatScalar();
  //     return out;
  //   }
  //   inline Scalar_base* ScIInit_u64() {
  //     Scalar_base* out = new Uint64Scalar();
  //     return out;
  //   }
  //   inline Scalar_base* ScIInit_i64() {
  //     Scalar_base* out = new Int64Scalar();
  //     return out;
  //   }
  //   inline Scalar_base* ScIInit_u32() {
  //     Scalar_base* out = new Uint32Scalar();
  //     return out;
  //   }
  //   inline Scalar_base* ScIInit_i32() {
  //     Scalar_base* out = new Int32Scalar();
  //     return out;
  //   }
  //   inline Scalar_base* ScIInit_u16() {
  //     Scalar_base* out = new Uint16Scalar();
  //     return out;
  //   }
  //   inline Scalar_base* ScIInit_i16() {
  //     Scalar_base* out = new Int16Scalar();
  //     return out;
  //   }
  //   inline Scalar_base* ScIInit_b() {
  //     Scalar_base* out = new BoolScalar();
  //     return out;
  //   }
  //   Scalar_init_interface::Scalar_init_interface() {
  //     if (!inited) {
  //       UScIInit[this->Double] = ScIInit_d;
  //       UScIInit[this->Float] = ScIInit_f;
  //       UScIInit[this->ComplexDouble] = ScIInit_cd;
  //       UScIInit[this->ComplexFloat] = ScIInit_cf;
  //       UScIInit[this->Uint64] = ScIInit_u64;
  //       UScIInit[this->Int64] = ScIInit_i64;
  //       UScIInit[this->Uint32] = ScIInit_u32;
  //       UScIInit[this->Int32] = ScIInit_i32;
  //       UScIInit[this->Uint16] = ScIInit_u16;
  //       UScIInit[this->Int16] = ScIInit_i16;
  //       UScIInit[this->Bool] = ScIInit_b;
  //       inited = true;
  //     }
  //   }

  //   Scalar_init_interface __ScII;
}  // namespace cytnx
