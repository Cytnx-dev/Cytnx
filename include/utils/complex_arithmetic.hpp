#ifndef CYTNX_UTILS_COMPLEX_ARITHMETIC_H_
#define CYTNX_UTILS_COMPLEX_ARITHMETIC_H_

#include <complex>
#include <type_traits>

#include "Type.hpp"

// Mixed complex/scalar arithmetic for the cytnx dtypes.
//
// std::complex only defines its heterogeneous operators as
//
//   template <class T> complex<T> operator*(const complex<T>&, const T&);
//
// so both operands must deduce the *same* T. That leaves ordinary expressions such as
// `cytnx_complex128{} * 2`, `z == 0`, `z + cytnx_float{1}` and `cytnx_complex64{} * 2.0`
// without a viable operator. This header fills exactly those gaps.
//
// It used to do so with ~300 hand-written namespace-scope overloads against every builtin
// scalar. Because cytnx_complex64/128 are aliases for std::complex<float/double> and builtins
// convert into them, those declarations entered ordinary overload resolution for completely
// unrelated code under `using namespace cytnx` -- amplified by C++20's reversed operator==
// rewrite, which made e.g. `std::vector<bool>::reference == bool` ambiguous (#1003).
//
// The templates below are constrained so that a candidate only exists when at least one operand
// really is a complex cytnx dtype, so unrelated types never reach overload resolution.
// tests/overload_hygiene_test.cpp pins that.
namespace cytnx {

  namespace internal {

    // The real type underlying a complex dtype; T itself for a non-complex T. Used only to ask
    // "would std::complex's own operator already handle this pair?".
    template <class T>
    struct complex_value_type {
      using type = T;
    };

    template <class T>
    struct complex_value_type<std::complex<T>> {
      using type = T;
    };

    template <class T>
    using complex_value_type_t = typename complex_value_type<T>::type;

    // True when <L, R> is a pair std::complex already provides an operator for: complex<T> with
    // complex<T>, complex<T> with T, or T with complex<T>. Those must stay with std, otherwise the
    // cytnx template would be a second equally-good candidate and make them ambiguous.
    template <class L, class R>
    constexpr bool std_complex_handles_v = std::is_same_v<L, R> ||
                                           (is_complex_floating_point_v<L> &&
                                            std::is_same_v<complex_value_type_t<L>, R>) ||
                                           (is_complex_floating_point_v<R> &&
                                            std::is_same_v<L, complex_value_type_t<R>>);

  }  // namespace internal

  // The operand pairs the operators below apply to: both are cytnx dtypes, at least one is a
  // complex dtype, and std::complex does not already handle the pair. Requiring CytnxType on both
  // sides is what keeps unrelated types (std::vector<bool>::reference, user-defined classes, char,
  // long double, ...) from ever forming a candidate, and it also guarantees that
  // Type_class::type_promote_t below is well-formed.
  template <class L, class R>
  concept ComplexMixedOperands = CytnxType<L> && CytnxType<R> &&
    (is_complex_floating_point_v<L> ||
     is_complex_floating_point_v<R>)&&!internal::std_complex_handles_v<L, R>;

  // The result dtype is Cytnx's own promotion (Type.type_promote), not C++'s. Note this differs
  // from the overloads it replaces: those returned complex64 for `complex64 op double`, silently
  // dropping the double's precision. type_promote crosses the real/complex boundary by precision
  // (#858, #982), so that pair now yields complex128.
  //
  // Operands are taken by value: complex<double> is two registers wide, and every other
  // participating type is a builtin scalar.
  template <class L, class R>
  requires ComplexMixedOperands<L, R>
  constexpr auto operator+(L lhs, R rhs) {
    using TO = Type_class::type_promote_t<L, R>;
    return static_cast<TO>(lhs) + static_cast<TO>(rhs);
  }

  template <class L, class R>
  requires ComplexMixedOperands<L, R>
  constexpr auto operator-(L lhs, R rhs) {
    using TO = Type_class::type_promote_t<L, R>;
    return static_cast<TO>(lhs) - static_cast<TO>(rhs);
  }

  template <class L, class R>
  requires ComplexMixedOperands<L, R>
  constexpr auto operator*(L lhs, R rhs) {
    using TO = Type_class::type_promote_t<L, R>;
    return static_cast<TO>(lhs) * static_cast<TO>(rhs);
  }

  template <class L, class R>
  requires ComplexMixedOperands<L, R>
  constexpr auto operator/(L lhs, R rhs) {
    using TO = Type_class::type_promote_t<L, R>;
    return static_cast<TO>(lhs) / static_cast<TO>(rhs);
  }

  // One operator== suffices: C++20 synthesizes `scalar == complex` from the reversed candidate and
  // `!=` from the negation. Comparing in the promoted type matches the arithmetic above.
  template <class L, class R>
  requires ComplexMixedOperands<L, R>
  constexpr bool operator==(L lhs, R rhs) {
    using TO = Type_class::type_promote_t<L, R>;
    return static_cast<TO>(lhs) == static_cast<TO>(rhs);
  }

}  // namespace cytnx

#endif  // CYTNX_UTILS_COMPLEX_ARITHMETIC_H_
