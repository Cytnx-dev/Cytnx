#include "backend/Scalar.hpp"
#include "backend/Storage.hpp"

#include <complex>
#include <limits>
#include <type_traits>
#include <variant>

namespace cytnx {

  namespace {

    // ---- dtype / promotion helpers -----------------------------------------
    //
    // Every promotion decision below routes through Type_class::type_promote
    // (Type.hpp), which is the single source of truth for "what dtype should
    // the result of combining typeL and typeR have". Before this refactor,
    // Scalar re-derived a promotion order by hand (comparing raw enum values),
    // which silently disagreed with type_promote for signed/unsigned integer
    // mixes and gave the wrong (lossy) answer for ComplexFloat vs Double (see
    // #935). Centralizing on type_promote means Scalar automatically inherits
    // any future fix to the promotion table.

    void ensure_not_void(const Scalar &s, const char *what) {
      cytnx_error_msg(s.dtype() == (int)Type.Void, "[ERROR] %s: Scalar is Void (uninitialized).%s",
                      what, "\n");
    }

    // Prints "< value >" exactly as the old per-dtype Scalar_base::print()
    // implementations did (the Void base class printed nothing).
    void print_value(std::ostream &os, const Scalar &in) {
      std::visit(
        [&](auto &&v) {
          using T = std::decay_t<decltype(v)>;
          if constexpr (!std::is_same_v<T, std::monostate>) {
            os << "< " << v << " >";
          }
        },
        in._val);
    }

    // ---- generic per-alternative helpers ------------------------------------

    // Turn a runtime dtype id into a value-initialized ScalarVariant whose
    // active alternative is the one at index `dtype`. This is the inverse of
    // Scalar::dtype() (itself just _val.index()): ScalarVariant is
    // index-aligned with Type_list -- pinned by the static_asserts in
    // Scalar.hpp -- so alternative `dtype` holds exactly the C++ type that
    // dtype names. std::visit-ing the returned prototype recovers that type at
    // compile time, so astype() dispatches by *visiting the variant* instead
    // of a hand-written dtype->type switch that could drift out of sync with
    // Type_list. The fold emplaces the single alternative whose index matches
    // `dtype` (short-circuiting on the first match); an out-of-range dtype
    // matches nothing and errors.
    Scalar::ScalarVariant variant_prototype(unsigned int dtype) {
      Scalar::ScalarVariant proto;  // defaults to monostate (index 0 == Void)
      const bool matched = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return ((Is == dtype ? (proto.template emplace<Is>(), true) : false) || ...);
      }
      (std::make_index_sequence<std::variant_size_v<Scalar::ScalarVariant>>{});
      cytnx_error_msg(!matched,
                      "[ERROR] invalid dtype id %u for Scalar (no matching Type_list entry).%s",
                      dtype, "\n");
      return proto;
    }

    // ---- arithmetic -----------------------------------------------------

    // Perform the requested binary arithmetic op between the (already
    // type_promote-selected) representations of lhs and rhs, returning the
    // promoted-dtype result. `Op` is one of +,-,*,/ passed as a lambda.
    template <typename BinOp>
    Scalar binary_arith(const Scalar &lhs, const Scalar &rhs, BinOp &&op, const char *opname) {
      ensure_not_void(lhs, opname);
      ensure_not_void(rhs, opname);
      unsigned int out_dtype = Type_class::type_promote(lhs.dtype(), rhs.dtype());
      Scalar L = lhs.astype(out_dtype);
      Scalar R = rhs.astype(out_dtype);
      Scalar out;
      std::visit(
        [&](auto &&lv, auto &&rv) {
          using TL = std::decay_t<decltype(lv)>;
          using TR = std::decay_t<decltype(rv)>;
          // TL and TR are guaranteed identical here since both operands were
          // just astype()'d to the same promoted dtype; the mismatched-type
          // branches below are unreachable at runtime but must still be
          // well-formed for std::visit's exhaustiveness requirement.
          if constexpr (std::is_same_v<TL, TR> && !std::is_same_v<TL, std::monostate>) {
            out._val = static_cast<TL>(op(lv, rv));
          } else {
            cytnx_error_msg(true, "[ERROR] Scalar %s: unexpected Void operand.%s", opname, "\n");
          }
        },
        L._val, R._val);
      return out;
    }

    // `forbid_complex` is a template parameter, not a runtime flag, so the
    // complex-ordering branch below is discarded with `if constexpr`. That
    // matters: with a runtime bool the compiler must still type-check
    // `op(complex, complex)` for the ordering lambdas (< <= > >=), which has no
    // built-in operator and would only compile by accidentally routing through
    // operator<(Scalar, Scalar) via the implicit Scalar constructors -- a
    // silent infinite-recursion hazard. Making it compile-time removes that
    // branch entirely for ordering ops, so it is never instantiated.
    template <bool forbid_complex, typename CmpOp>
    bool binary_cmp(const Scalar &lhs, const Scalar &rhs, CmpOp &&op, const char *opname) {
      ensure_not_void(lhs, opname);
      ensure_not_void(rhs, opname);
      if constexpr (forbid_complex) {
        cytnx_error_msg(Type.is_complex(lhs.dtype()) || Type.is_complex(rhs.dtype()),
                        "[ERROR] Scalar %s: comparison not supported for complex type%s", opname,
                        "\n");
      }
      unsigned int out_dtype = Type_class::type_promote(lhs.dtype(), rhs.dtype());
      Scalar L = lhs.astype(out_dtype);
      Scalar R = rhs.astype(out_dtype);
      bool result = false;
      std::visit(
        [&](auto &&lv, auto &&rv) {
          using TL = std::decay_t<decltype(lv)>;
          using TR = std::decay_t<decltype(rv)>;
          // See binary_arith(): TL and TR are guaranteed identical at
          // runtime since both operands were astype()'d to the promoted
          // dtype; other branches exist only to satisfy std::visit.
          if constexpr (!std::is_same_v<TL, TR>) {
            cytnx_error_msg(true, "[ERROR] Scalar %s: unexpected Void operand.%s", opname, "\n");
          } else if constexpr (std::is_same_v<TL, std::monostate>) {
            cytnx_error_msg(true, "[ERROR] Scalar %s: unexpected Void operand.%s", opname, "\n");
          } else if constexpr (is_complex_v<TL>) {
            // Ordering comparisons (< <= > >=) forbid complex entirely (also
            // checked eagerly above); equality (forbid_complex == false) is
            // well-defined for complex via operator== and must not throw.
            // `if constexpr` here means `op(complex, complex)` is only
            // instantiated for equality, never for the ordering lambdas.
            if constexpr (forbid_complex) {
              cytnx_error_msg(true,
                              "[ERROR] Scalar %s: comparison not supported for complex type%s",
                              opname, "\n");
            } else {
              result = op(lv, rv);
            }
          } else {
            result = op(lv, rv);
          }
        },
        L._val, R._val);
      return result;
    }

  }  // namespace

  // ---- Scalar member functions --------------------------------------------

  Scalar Scalar::maxval(const unsigned int &dtype) {
    Scalar out(0, dtype);
    std::visit(
      [](auto &&v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          cytnx_error_msg(true, "[ERROR] maxval not supported for Void type%s", "\n");
        } else if constexpr (is_complex_v<T>) {
          cytnx_error_msg(true, "[ERROR] maxval not supported for complex type%s", "\n");
        } else {
          v = std::numeric_limits<T>::max();
        }
      },
      out._val);
    return out;
  }

  Scalar Scalar::minval(const unsigned int &dtype) {
    Scalar out(0, dtype);
    std::visit(
      [](auto &&v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          cytnx_error_msg(true, "[ERROR] minval not supported for Void type%s", "\n");
        } else if constexpr (is_complex_v<T>) {
          cytnx_error_msg(true, "[ERROR] minval not supported for complex type%s", "\n");
        } else if constexpr (std::is_floating_point_v<T>) {
          // #935: floating-point minval must be the most negative finite
          // value (lowest()), not numeric_limits<T>::min() (smallest
          // positive normal value).
          v = std::numeric_limits<T>::lowest();
        } else {
          v = std::numeric_limits<T>::min();
        }
      },
      out._val);
    return out;
  }

  Scalar::Scalar(const Sproxy &prox) : _val(prox._insimpl->get_item(prox._loc)._val) {}

  Scalar Scalar::astype(const unsigned int &dtype) const {
    // Dispatch by visiting the variant: materialize the *destination*
    // alternative as a prototype at index `dtype`, then std::visit the
    // (source, destination) pair to recover both compile-time types and
    // perform the conversion. No dtype->type switch to keep in sync.
    Scalar out;
    ScalarVariant dst_proto = variant_prototype(dtype);
    std::visit(
      [&](auto &&src, auto &&dst) {
        using TSrc = std::decay_t<decltype(src)>;
        using TDst = std::decay_t<decltype(dst)>;
        if constexpr (std::is_same_v<TSrc, std::monostate>) {
          // An explicit conversion of an uninitialized Scalar is a caller
          // bug; error instead of silently staying Void (review thread on
          // the earlier no-op behavior).
          cytnx_error_msg(true,
                          "[ERROR] Cannot astype a Void (uninitialized) Scalar. Assign a value "
                          "first.%s",
                          "\n");
        } else if constexpr (std::is_same_v<TDst, std::monostate>) {
          // Void is not a valid conversion target for a non-Void Scalar.
          cytnx_error_msg(true, "[ERROR] invalid target dtype for Scalar::astype: Void%s", "\n");
        } else if constexpr (requires { static_cast<TDst>(src); }) {
          out._val = static_cast<TDst>(src);
        } else {
          // The only cytnx pair with no static_cast is complex -> real.
          cytnx_error_msg(true,
                          "[ERROR] Cannot convert Scalar from dtype [%s] to dtype [%s]. Use "
                          "real() or imag() to extract a component first.%s",
                          Type_enum_name<TSrc>, Type_enum_name<TDst>, "\n");
        }
      },
      this->_val, dst_proto);
    return out;
  }

  Scalar Scalar::conj() const {
    // The old PIMPL's Void base threw here; keep that contract instead of
    // silently returning another Void (codex review finding).
    ensure_not_void(*this, "conj");
    Scalar out = *this;
    std::visit(
      [](auto &&v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (is_complex_v<T>) v = std::conj(v);
      },
      out._val);
    return out;
  }

  Scalar Scalar::imag() const {
    ensure_not_void(*this, "imag");
    Scalar out;
    std::visit(
      [&](auto &&v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, cytnx_complex128>) {
          out._val = v.imag();
        } else if constexpr (std::is_same_v<T, cytnx_complex64>) {
          out._val = v.imag();
        } else if constexpr (std::is_same_v<T, std::monostate>) {
          // Unreachable after the ensure_not_void guard; kept for
          // std::visit exhaustiveness with a consistent message.
          cytnx_error_msg(true, "[ERROR] imag(): Scalar is Void (uninitialized).%s", "\n");
        } else {
          cytnx_error_msg(true, "[ERROR] real type Scalar does not have imag part!%s", "\n");
        }
      },
      this->_val);
    return out;
  }

  Scalar Scalar::real() const {
    ensure_not_void(*this, "real");
    Scalar out;
    std::visit(
      [&](auto &&v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, cytnx_complex128>) {
          out._val = v.real();
        } else if constexpr (std::is_same_v<T, cytnx_complex64>) {
          out._val = v.real();
        } else if constexpr (std::is_same_v<T, std::monostate>) {
          // Unreachable after the ensure_not_void guard; kept for
          // std::visit exhaustiveness with a consistent message.
          cytnx_error_msg(true, "[ERROR] real(): Scalar is Void (uninitialized).%s", "\n");
        } else {
          out._val = v;
        }
      },
      this->_val);
    return out;
  }

  // ScalarVariant is index-aligned with Type_list (static_asserts in
  // Scalar.hpp pin this), so the active alternative's index IS the dtype id.
  int Scalar::dtype() const { return (int)this->_val.index(); }

  void Scalar::print() const {
    print_value(std::cout, *this);
    std::cout << std::string(" Scalar dtype: [") << Type.getname(this->dtype()) << std::string("]")
              << std::endl;
  }

  // The per-type cast operators and to_complex128()/to_complex64() are now
  // defined inline in Scalar.hpp as one constrained operator template (see
  // the review thread); the explicit_cast/convert_value helpers they used
  // are gone with them.

  // ---- in-place arithmetic ------------------------------------------------
  //
  // In-place binary arithmetic follows Python's value-type semantics: `a op= b`
  // is exactly `a = a op b`, promoting the destination dtype via
  // Type.type_promote just like the out-of-place operators (maintainer ruling
  // 2026-07-08, adopting @ianmccul's point on #1011). It does NOT preserve the
  // LHS dtype: e.g. `Int64 += Double` yields a Double, `Double *= ComplexDouble`
  // yields a ComplexDouble. This is well-defined precisely because the variant
  // rewrite routes every combination through type_promote (the dtype bugs that
  // motivated #937's emergency disabling of these paths are structurally gone).
  // Delegating to radd/rsub/rmul/rdiv keeps in-place and out-of-place bit-for-bit
  // consistent, and the out-of-place result is a fresh Scalar, so `a += a` is
  // safe (no aliasing during the visit).

  void Scalar::operator+=(const Scalar &rhs) { *this = this->radd(rhs); }

  void Scalar::operator-=(const Scalar &rhs) { *this = this->rsub(rhs); }

  void Scalar::operator*=(const Scalar &rhs) { *this = this->rmul(rhs); }

  void Scalar::operator/=(const Scalar &rhs) { *this = this->rdiv(rhs); }

  void Scalar::iabs() {
    ensure_not_void(*this, "iabs");
    std::visit(
      [](auto &&v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          // unreachable, guarded above.
        } else if constexpr (is_complex_v<T>) {
          // iabs()/isqrt() are *unary* in-place transforms of the stored value
          // and are dtype-preserving by design -- unlike in-place binary
          // arithmetic (+= etc.), which promotes, there is no second operand to
          // promote against. So iabs() on a complex Scalar keeps the complex
          // dtype and stores the magnitude as the real part with a zero
          // imaginary part. Its *value* (the magnitude) matches the
          // out-of-place abs() exactly; abs() additionally narrows the result
          // to the real dtype (#935's abs()/iabs() consistency is thus by
          // value, documented here and pinned in Scalar_test.cpp).
          v = T(std::abs(v), 0);
        } else if constexpr (std::is_unsigned_v<T>) {
          // Unsigned integer / bool: abs() is a no-op (already non-negative);
          // std::abs has no unsigned overload and is ambiguous if called
          // directly on these types.
          // (leave v unchanged)
        } else {
          v = std::abs(v);
        }
      },
      this->_val);
  }

  void Scalar::isqrt() {
    ensure_not_void(*this, "isqrt");
    std::visit(
      [](auto &&v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          // unreachable, guarded above.
        } else if constexpr (std::is_same_v<T, cytnx_bool>) {
          cytnx_error_msg(true, "[ERROR] isqrt not supported for Bool type%s", "\n");
        } else {
          // For integer T, std::sqrt returns double; isqrt is dtype-preserving
          // (unary in-place), so narrow back to T explicitly.
          v = static_cast<T>(std::sqrt(v));
        }
      },
      this->_val);
  }

  Scalar Scalar::abs() const {
    ensure_not_void(*this, "abs");
    Scalar out;
    std::visit(
      [&](auto &&v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          // unreachable, guarded above.
        } else if constexpr (is_complex_v<T>) {
          // abs() of a complex Scalar always returns the (real-dtype)
          // magnitude -- fixes #935's abs()/iabs() dtype inconsistency.
          out._val = std::abs(v);
        } else if constexpr (std::is_unsigned_v<T>) {
          // Already non-negative; std::abs has no unsigned overload.
          out._val = v;
        } else {
          out._val = static_cast<T>(std::abs(v));
        }
      },
      this->_val);
    return out;
  }

  Scalar Scalar::sqrt() const {
    Scalar out = *this;
    out.isqrt();
    return out;
  }

  bool Scalar::less(const Scalar &rhs) const {
    return binary_cmp<true>(
      *this, rhs, [](auto &&a, auto &&b) { return a < b; }, "less-than (<)");
  }
  bool Scalar::leq(const Scalar &rhs) const {
    return binary_cmp<true>(
      *this, rhs, [](auto &&a, auto &&b) { return a <= b; }, "less-equal (<=)");
  }
  bool Scalar::greater(const Scalar &rhs) const {
    return binary_cmp<true>(
      *this, rhs, [](auto &&a, auto &&b) { return a > b; }, "greater-than (>)");
  }
  bool Scalar::geq(const Scalar &rhs) const {
    return binary_cmp<true>(
      *this, rhs, [](auto &&a, auto &&b) { return a >= b; }, "greater-equal (>=)");
  }
  bool Scalar::eq(const Scalar &rhs) const {
    return binary_cmp<false>(
      *this, rhs, [](auto &&a, auto &&b) { return a == b; }, "equal (==)");
  }

  Scalar Scalar::radd(const Scalar &rhs) const {
    return binary_arith(
      *this, rhs, [](auto &&a, auto &&b) { return a + b; }, "addition (+)");
  }
  Scalar Scalar::rmul(const Scalar &rhs) const {
    return binary_arith(
      *this, rhs, [](auto &&a, auto &&b) { return a * b; }, "multiplication (*)");
  }
  Scalar Scalar::rsub(const Scalar &rhs) const {
    return binary_arith(
      *this, rhs, [](auto &&a, auto &&b) { return a - b; }, "subtraction (-)");
  }
  Scalar Scalar::rdiv(const Scalar &rhs) const {
    return binary_arith(
      *this, rhs, [](auto &&a, auto &&b) { return a / b; }, "division (/)");
  }

  // ---- free functions -------------------------------------------------

  cytnx_complex128 complex128(const Scalar &in) { return in.to_complex128(); }

  cytnx_complex64 complex64(const Scalar &in) { return in.to_complex64(); }

  std::ostream &operator<<(std::ostream &os, const Scalar &in) {
    print_value(os, in);
    os << std::string(" dtype: [") << Type.getname(in.dtype()) << std::string("]");
    return os;
  }

  // ladd: c + Scalar:
  Scalar operator+(const Scalar &lc, const Scalar &rs) { return rs.radd(lc); }

  // lmul c * Scalar;
  Scalar operator*(const Scalar &lc, const Scalar &rs) { return rs.rmul(lc); }

  // lsub c - Scalar;
  Scalar operator-(const Scalar &lc, const Scalar &rs) { return lc.rsub(rs); }

  // ldiv c / Scalar;
  Scalar operator/(const Scalar &lc, const Scalar &rs) { return lc.rdiv(rs); }

  // lless c < Scalar;
  bool operator<(const Scalar &lc, const Scalar &rs) { return lc.less(rs); }

  // lless c > Scalar;
  bool operator>(const Scalar &lc, const Scalar &rs) { return lc.greater(rs); }

  // lless c <= Scalar;
  bool operator<=(const Scalar &lc, const Scalar &rs) { return lc.leq(rs); }

  // lless c >= Scalar;
  bool operator>=(const Scalar &lc, const Scalar &rs) { return lc.geq(rs); }

  // eq c == Scalar;
  bool operator==(const Scalar &lc, const Scalar &rs) { return lc.eq(rs); }

  Scalar abs(const Scalar &c) { return c.abs(); }

  Scalar sqrt(const Scalar &c) { return c.sqrt(); }

  // Scalar proxy:
  // Sproxy
  Scalar::Sproxy &Scalar::Sproxy::operator=(const Scalar::Sproxy &rc) {
    if (this->_insimpl.get() == 0) {
      //  not init:
      this->_insimpl = rc._insimpl;
      this->_loc = rc._loc;
      return *this;
    } else {
      if ((rc._insimpl == this->_insimpl) && (rc._loc == this->_loc)) {
        return *this;
      } else {
        Scalar tmp = rc._insimpl->get_item(rc._loc);
        this->_insimpl->set_item(this->_loc, tmp);
        return *this;
      }
    }
  }
  Scalar::Sproxy &Scalar::Sproxy::operator=(const Scalar &rc) {
    this->_insimpl->set_item(this->_loc, rc);
    return *this;
  }

  bool Scalar::Sproxy::exists() const { return this->_insimpl->dtype() != Type.Void; }

  Scalar Scalar::Sproxy::real() { return Scalar(*this).real(); }
  Scalar Scalar::Sproxy::imag() { return Scalar(*this).imag(); }

}  // namespace cytnx
