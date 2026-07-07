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

    // Ruling 1 (issues #935/#937): in-place arithmetic (+= -= *= /=) throws
    // unless combining the two dtypes under type_promote would stay at the
    // LHS dtype. This is exactly the "does the in-place op need to silently
    // change dtype" test:
    //   - same dtype: promote(L,L) == L -> always allowed.
    //   - Int64 += Int32, Double += Float, ComplexDouble += Double, etc.:
    //     promote(L,R) == L -> allowed: the conversion of R into L is
    //     value-preserving per the sanctioned type_promote table (see
    //     include/Type.hpp type_promote; maintainer sign-off 2026-07-07).
    //     Note this includes same-width unsigned->signed RHS conversions
    //     (e.g. Int64 += Uint64), which the table resolves to the signed
    //     LHS by explicit maintainer ruling: in-place and out-of-place stay
    //     consistent, and any future tightening happens in the table.
    //   - Uint64 += Int64, Int64 += Double, Double += ComplexDouble, Bool +=
    //     anything-but-Bool: promote(L,R) != L -> would require changing the
    //     LHS's dtype (or silently truncating, e.g. old integer-preserving
    //     in-place division) -> throw, telling the caller to use astype().
    void ensure_inplace_dtype_unchanged_by_promote(unsigned int lhs_dtype, unsigned int rhs_dtype,
                                                   const char *op, const char *op_symbol) {
      if (rhs_dtype == Type.Void || lhs_dtype == Type.Void) {
        cytnx_error_msg(true, "[ERROR] Scalar %s: cannot operate on a Void Scalar.%s", op, "\n");
      }
      unsigned int promoted = Type_class::type_promote(lhs_dtype, rhs_dtype);
      cytnx_error_msg(promoted != lhs_dtype,
                      "[ERROR] Scalar in-place %s between dtype [%s] and [%s] would change or "
                      "truncate the destination dtype (result dtype would be [%s]). Use "
                      "out-of-place arithmetic (e.g. `a = a %s b`) or `astype()` to convert "
                      "explicitly.%s",
                      op, Type.getname(lhs_dtype).c_str(), Type.getname(rhs_dtype).c_str(),
                      Type.getname(promoted).c_str(), op_symbol, "\n");
    }

    // ---- generic per-alternative helpers ------------------------------------

    // Cast a source value of type TSrc to the destination type TDst,
    // following the pre-existing Scalar conversion rules:
    //  - real <- complex is disallowed (use real()/imag() instead).
    //  - anything else is a plain static_cast (matches the old
    //    assign_selftype()/to_cytnx_XXX() behavior).
    template <typename TDst, typename TSrc>
    TDst convert_value(const TSrc &src) {
      if constexpr (is_complex_v<TDst> || !is_complex_v<TSrc>) {
        return static_cast<TDst>(src);
      } else {
        cytnx_error_msg(true,
                        "[ERROR] Cannot convert a complex Scalar to a real dtype. Use real() or "
                        "imag() to extract a component first.%s",
                        "\n");
        return TDst{};
      }
    }

    // Visitor: convert whatever is active into dtype `dtype` on the Type_list,
    // returning a new ScalarVariant.
    struct AstypeVisitor {
      unsigned int dtype;

      Scalar::ScalarVariant operator()(std::monostate) const { return std::monostate{}; }

      template <typename TSrc>
      Scalar::ScalarVariant operator()(const TSrc &src) const {
        Scalar::ScalarVariant out;
        VisitByDtype(dtype, [&](auto tag) {
          using TDst = typename decltype(tag)::type;
          out = convert_value<TDst>(src);
        });
        return out;
      }

      // Dispatch a runtime dtype id to a compile-time type tag, calling
      // `fn(type_identity<T>{})`. Kept local to this visitor: it is the
      // inverse operation of Scalar::dtype() and is only needed for astype().
      template <typename Fn>
      static void VisitByDtype(unsigned int dtype, Fn &&fn) {
        switch (dtype) {
          case Type.ComplexDouble:
            fn(std::type_identity<cytnx_complex128>{});
            return;
          case Type.ComplexFloat:
            fn(std::type_identity<cytnx_complex64>{});
            return;
          case Type.Double:
            fn(std::type_identity<cytnx_double>{});
            return;
          case Type.Float:
            fn(std::type_identity<cytnx_float>{});
            return;
          case Type.Int64:
            fn(std::type_identity<cytnx_int64>{});
            return;
          case Type.Uint64:
            fn(std::type_identity<cytnx_uint64>{});
            return;
          case Type.Int32:
            fn(std::type_identity<cytnx_int32>{});
            return;
          case Type.Uint32:
            fn(std::type_identity<cytnx_uint32>{});
            return;
          case Type.Int16:
            fn(std::type_identity<cytnx_int16>{});
            return;
          case Type.Uint16:
            fn(std::type_identity<cytnx_uint16>{});
            return;
          case Type.Bool:
            fn(std::type_identity<cytnx_bool>{});
            return;
          default:
            cytnx_error_msg(true, "[ERROR] invalid target dtype for Scalar::astype: %d%s", dtype,
                            "\n");
        }
      }
    };

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

    template <typename CmpOp>
    bool binary_cmp(const Scalar &lhs, const Scalar &rhs, CmpOp &&op, const char *opname,
                    bool forbid_complex) {
      ensure_not_void(lhs, opname);
      ensure_not_void(rhs, opname);
      if (forbid_complex) {
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
            // Ordering comparisons (< <= > >=) forbid complex entirely
            // (checked eagerly above via `forbid_complex`, so this is
            // unreachable in that case); equality (`forbid_complex == false`)
            // is well-defined for complex via operator== and must not throw.
            if (forbid_complex) {
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
    Scalar out;
    out._val = std::visit(AstypeVisitor{dtype}, this->_val);
    return out;
  }

  Scalar Scalar::conj() const {
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

  namespace {
    template <typename TDst>
    TDst explicit_cast(const Scalar &s) {
      TDst out{};
      std::visit(
        [&](auto &&v) {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, std::monostate>) {
            cytnx_error_msg(
              true, "[ERROR] Cannot cast a Void (uninitialized) Scalar to any type.%s", "\n");
          } else {
            out = convert_value<TDst>(v);
          }
        },
        s._val);
      return out;
    }
  }  // namespace

  Scalar::operator cytnx_double() const { return explicit_cast<cytnx_double>(*this); }
  Scalar::operator cytnx_float() const { return explicit_cast<cytnx_float>(*this); }
  Scalar::operator cytnx_uint64() const { return explicit_cast<cytnx_uint64>(*this); }
  Scalar::operator cytnx_int64() const { return explicit_cast<cytnx_int64>(*this); }
  Scalar::operator cytnx_uint32() const { return explicit_cast<cytnx_uint32>(*this); }
  Scalar::operator cytnx_int32() const { return explicit_cast<cytnx_int32>(*this); }
  Scalar::operator cytnx_uint16() const { return explicit_cast<cytnx_uint16>(*this); }
  Scalar::operator cytnx_int16() const { return explicit_cast<cytnx_int16>(*this); }
  Scalar::operator cytnx_bool() const { return explicit_cast<cytnx_bool>(*this); }
  cytnx_complex128 Scalar::to_complex128() const { return explicit_cast<cytnx_complex128>(*this); }
  cytnx_complex64 Scalar::to_complex64() const { return explicit_cast<cytnx_complex64>(*this); }

  // ---- in-place arithmetic (Ruling 1: throw on lossy mixed-dtype) --------

  namespace {
    // Shared body of the four in-place operators, mirroring binary_arith();
    // a future in-place op (e.g. %=) only needs a new one-line call. After
    // `rhs.astype(lhs.dtype())`, the two variants are guaranteed to hold the
    // *same* alternative (both equal to lhs.dtype()), so the std::visit
    // callback only ever needs the TL==TR branch to be live at runtime. It
    // must still be well-formed (if constexpr-guarded) for every (TL, TR)
    // combination the compiler instantiates, including monostate and
    // mismatched-type pairs, since std::visit requires the visitor to be
    // callable for the full cross product of alternatives.
    template <typename InplaceOp>
    void inplace_arith(Scalar &lhs, const Scalar &rhs, InplaceOp &&op, const char *opname,
                       const char *op_symbol) {
      ensure_inplace_dtype_unchanged_by_promote(lhs.dtype(), rhs.dtype(), opname, op_symbol);
      Scalar rhs_conv = rhs.astype(lhs.dtype());
      std::visit(
        [&](auto &&lv, auto &&rv) {
          using TL = std::decay_t<decltype(lv)>;
          using TR = std::decay_t<decltype(rv)>;
          if constexpr (std::is_same_v<TL, TR> && !std::is_same_v<TL, std::monostate>) {
            op(lv, rv);
          }
        },
        lhs._val, rhs_conv._val);
    }
  }  // namespace

  void Scalar::operator+=(const Scalar &rhs) {
    inplace_arith(
      *this, rhs, [](auto &lv, const auto &rv) { lv += rv; }, "addition (+=)", "+");
  }

  void Scalar::operator-=(const Scalar &rhs) {
    inplace_arith(
      *this, rhs, [](auto &lv, const auto &rv) { lv -= rv; }, "subtraction (-=)", "-");
  }

  void Scalar::operator*=(const Scalar &rhs) {
    inplace_arith(
      *this, rhs, [](auto &lv, const auto &rv) { lv *= rv; }, "multiplication (*=)", "*");
  }

  void Scalar::operator/=(const Scalar &rhs) {
    inplace_arith(
      *this, rhs, [](auto &lv, const auto &rv) { lv /= rv; }, "division (/=)", "/");
  }

  void Scalar::iabs() {
    ensure_not_void(*this, "iabs");
    std::visit(
      [](auto &&v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          // unreachable, guarded above.
        } else if constexpr (is_complex_v<T>) {
          // #935: iabs() on a complex Scalar used to keep the dtype complex
          // and store the (real) magnitude as the real part with a zero
          // imaginary part -- inconsistent with the non-inplace abs(), which
          // always returns a real dtype. We keep iabs() dtype-preserving
          // (as every other in-place op is), so the chosen, documented
          // semantics is: iabs() stores the magnitude as the real part and
          // zeros the imaginary part. This matches abs()'s *value* (the
          // magnitude), differing only in that iabs() cannot change dtype
          // (no in-place op does), while abs() (out-of-place) additionally
          // narrows the result to the real dtype.
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
          v = std::sqrt(v);
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
    return binary_cmp(
      *this, rhs, [](auto &&a, auto &&b) { return a < b; }, "less-than (<)", true);
  }
  bool Scalar::leq(const Scalar &rhs) const {
    return binary_cmp(
      *this, rhs, [](auto &&a, auto &&b) { return a <= b; }, "less-equal (<=)", true);
  }
  bool Scalar::greater(const Scalar &rhs) const {
    return binary_cmp(
      *this, rhs, [](auto &&a, auto &&b) { return a > b; }, "greater-than (>)", true);
  }
  bool Scalar::geq(const Scalar &rhs) const {
    return binary_cmp(
      *this, rhs, [](auto &&a, auto &&b) { return a >= b; }, "greater-equal (>=)", true);
  }
  bool Scalar::eq(const Scalar &rhs) const {
    return binary_cmp(
      *this, rhs, [](auto &&a, auto &&b) { return a == b; }, "equal (==)", false);
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
  Scalar::Sproxy &Scalar::Sproxy::operator=(const cytnx_complex128 &rc) {
    this->_insimpl->set_item(this->_loc, rc);
    return *this;
  }
  Scalar::Sproxy &Scalar::Sproxy::operator=(const cytnx_complex64 &rc) {
    this->_insimpl->set_item(this->_loc, rc);
    return *this;
  }
  Scalar::Sproxy &Scalar::Sproxy::operator=(const cytnx_double &rc) {
    this->_insimpl->set_item(this->_loc, rc);
    return *this;
  }
  Scalar::Sproxy &Scalar::Sproxy::operator=(const cytnx_float &rc) {
    this->_insimpl->set_item(this->_loc, rc);
    return *this;
  }
  Scalar::Sproxy &Scalar::Sproxy::operator=(const cytnx_uint64 &rc) {
    this->_insimpl->set_item(this->_loc, rc);
    return *this;
  }
  Scalar::Sproxy &Scalar::Sproxy::operator=(const cytnx_int64 &rc) {
    this->_insimpl->set_item(this->_loc, rc);
    return *this;
  }
  Scalar::Sproxy &Scalar::Sproxy::operator=(const cytnx_uint32 &rc) {
    this->_insimpl->set_item(this->_loc, rc);
    return *this;
  }
  Scalar::Sproxy &Scalar::Sproxy::operator=(const cytnx_int32 &rc) {
    this->_insimpl->set_item(this->_loc, rc);
    return *this;
  }
  Scalar::Sproxy &Scalar::Sproxy::operator=(const cytnx_uint16 &rc) {
    this->_insimpl->set_item(this->_loc, rc);
    return *this;
  }
  Scalar::Sproxy &Scalar::Sproxy::operator=(const cytnx_int16 &rc) {
    this->_insimpl->set_item(this->_loc, rc);
    return *this;
  }
  Scalar::Sproxy &Scalar::Sproxy::operator=(const cytnx_bool &rc) {
    this->_insimpl->set_item(this->_loc, rc);
    return *this;
  }

  bool Scalar::Sproxy::exists() const { return this->_insimpl->dtype() != Type.Void; }

  Scalar Scalar::Sproxy::real() { return Scalar(*this).real(); }
  Scalar Scalar::Sproxy::imag() { return Scalar(*this).imag(); }

}  // namespace cytnx
