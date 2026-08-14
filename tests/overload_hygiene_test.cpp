#include <complex>
#include <string>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"

#include "cytnx.hpp"

// Two things are pinned here, both from the #1003 operator-hygiene fold-in.
//
// 1. HYGIENE. utils/complex_arithmetic.hpp used to declare ~300 namespace-scope operators between
//    cytnx_complex64/128 and every builtin scalar. Because cytnx_complex64/128 are aliases for
//    std::complex<float/double> and builtins convert into them, those declarations entered
//    ordinary overload resolution for unrelated code -- amplified by C++20's reversed operator==
//    rewrite, which made `std::vector<bool>::reference == bool` ambiguous under
//    `using namespace cytnx`. The replacement templates are constrained to require a complex cytnx
//    dtype operand, so unrelated types cannot form a candidate. This is mostly a COMPILE-TIME
//    guard: if unconstrained operators come back, this translation unit stops compiling.
//
// 2. POLICY. The mixed complex/scalar expressions std::complex itself does not provide (its
//    heterogeneous operators deduce a single T from both operands) stay available, and their
//    result dtype follows Cytnx's own Type.type_promote rather than C++'s native promotion. The
//    static_asserts below are the contract; the EXPECT_* calls pin the values.
using namespace cytnx;

namespace {

  // The exact expression from the original report, kept as a function so the ambiguity would be a
  // compile error even if the test body were optimized away.
  //
  // The vector MUST be non-const: only the non-const operator[] returns the proxy
  // std::vector<bool>::reference, and the proxy is what pulled the cytnx complex operators into
  // the candidate set. A const vector<bool> yields a plain bool and guards nothing.
  static_assert(!std::is_same_v<std::vector<bool>::reference, bool>,
                "std::vector<bool>::operator[] must return a proxy for this guard to mean "
                "anything");

  bool CompareVectorBoolElement(std::vector<bool>& values, bool rhs) { return values[0] == rhs; }

}  // namespace

TEST(OverloadHygiene, VectorBoolReferenceComparesWithBool) {
  std::vector<bool> values(2);
  values[0] = true;
  values[1] = false;
  bool rhs = true;

  EXPECT_TRUE(CompareVectorBoolElement(values, rhs));
  EXPECT_TRUE(values[0] == rhs);
  EXPECT_FALSE(values[1] == rhs);

  // Reversed operand order (also went through the C++20 reversed-candidate path).
  EXPECT_TRUE(rhs == values[0]);
  EXPECT_FALSE(rhs == values[1]);
}

TEST(OverloadHygiene, BuiltinEqualityIsUnambiguous) {
  // Plain builtin comparisons must not be hijacked by the cytnx complex operators either.
  int i = 3;
  double d = 3.0;
  bool b = true;
  EXPECT_TRUE(i == 3);
  EXPECT_TRUE(d == 3.0);
  EXPECT_TRUE(b == true);
}

TEST(OverloadHygiene, ConceptExcludesNonComplexOperands) {
  // The constraint is what keeps unrelated types out of overload resolution: a candidate exists
  // only when at least one operand is a complex cytnx dtype.
  static_assert(!ComplexMixedOperands<cytnx_double, cytnx_int32>);
  static_assert(!ComplexMixedOperands<bool, bool>);
  static_assert(!ComplexMixedOperands<std::vector<bool>::reference, bool>);
  static_assert(!ComplexMixedOperands<std::string, cytnx_complex128>);
  static_assert(!ComplexMixedOperands<long double, cytnx_complex128>);

  // ... and only when std::complex does not already handle the pair, so the cytnx template can
  // never become a second equally-good candidate for those.
  static_assert(!ComplexMixedOperands<cytnx_complex128, cytnx_complex128>);
  static_assert(!ComplexMixedOperands<cytnx_complex128, cytnx_double>);
  static_assert(!ComplexMixedOperands<cytnx_double, cytnx_complex128>);
  static_assert(!ComplexMixedOperands<cytnx_complex64, cytnx_float>);

  // The gaps this header fills.
  static_assert(ComplexMixedOperands<cytnx_complex128, cytnx_complex64>);
  static_assert(ComplexMixedOperands<cytnx_complex128, cytnx_int32>);
  static_assert(ComplexMixedOperands<cytnx_complex64, cytnx_double>);
  static_assert(ComplexMixedOperands<cytnx_bool, cytnx_complex64>);
}

TEST(ComplexScalarArithmetic, ResultDtypeFollowsTypePromote) {
  // Result types are Cytnx's promotion, not C++'s. In particular complex64 op double widens to
  // complex128: the retired hand-written overloads returned complex64 here, silently discarding
  // the double's precision (contrary to Type.type_promote, #858/#982).
  static_assert(std::is_same_v<decltype(cytnx_complex64{} * cytnx_double{}), cytnx_complex128>);
  static_assert(std::is_same_v<decltype(cytnx_complex64{} + cytnx_double{}), cytnx_complex128>);
  static_assert(std::is_same_v<decltype(cytnx_complex128{} * cytnx_complex64{}), cytnx_complex128>);
  static_assert(std::is_same_v<decltype(cytnx_complex128{} * cytnx_int32{}), cytnx_complex128>);
  static_assert(std::is_same_v<decltype(cytnx_complex64{} * cytnx_int64{}), cytnx_complex64>);
  static_assert(std::is_same_v<decltype(cytnx_int16{} + cytnx_complex64{}), cytnx_complex64>);

  // Pairs std::complex owns keep their own (unchanged) result types.
  static_assert(std::is_same_v<decltype(cytnx_complex128{} * cytnx_double{}), cytnx_complex128>);
  static_assert(std::is_same_v<decltype(cytnx_complex64{} * cytnx_float{}), cytnx_complex64>);
}

TEST(ComplexScalarArithmetic, MixedPrecisionValuesAreExact) {
  // complex128 x complex64, the one case std::complex genuinely cannot express. Values are chosen
  // to be exact in binary floating point so the expectation is independent of the code path:
  // (1.5 - 2.25i) * (0.5 + 0.25i) = 0.75 + 0.375i - 1.125i - 0.5625 i^2 = 1.3125 - 0.75i.
  cytnx_complex128 z(1.5, -2.25);
  cytnx_complex64 w(0.5f, 0.25f);

  cytnx_complex128 product = z * w;
  EXPECT_DOUBLE_EQ(product.real(), 1.3125);
  EXPECT_DOUBLE_EQ(product.imag(), -0.75);

  cytnx_complex128 sum = z + w;
  EXPECT_DOUBLE_EQ(sum.real(), 2.0);
  EXPECT_DOUBLE_EQ(sum.imag(), -2.0);

  cytnx_complex128 difference = z - w;
  EXPECT_DOUBLE_EQ(difference.real(), 1.0);
  EXPECT_DOUBLE_EQ(difference.imag(), -2.5);

  // (1.5 - 2.25i) / (0.5 + 0.25i): multiply through by conj(w) over |w|^2 = 0.3125.
  // z * conj(w) = (0.75 - 0.5625) + (-0.375 - 1.125)i = 0.1875 - 1.5i
  // -> (0.1875 - 1.5i) / 0.3125 = 0.6 - 4.8i.
  cytnx_complex128 quotient = z / w;
  EXPECT_NEAR(quotient.real(), 0.6, 1e-14);
  EXPECT_NEAR(quotient.imag(), -4.8, 1e-14);
}

TEST(ComplexScalarArithmetic, IntegerAndNegativeScalarOperands) {
  cytnx_complex128 z(2.5, -1.25);

  // Integer literals (deduce as int == cytnx_int32) have no std::complex operator.
  cytnx_complex128 scaled = z * 2;
  EXPECT_DOUBLE_EQ(scaled.real(), 5.0);
  EXPECT_DOUBLE_EQ(scaled.imag(), -2.5);

  cytnx_complex128 shifted = z + cytnx_int64{-3};
  EXPECT_DOUBLE_EQ(shifted.real(), -0.5);
  EXPECT_DOUBLE_EQ(shifted.imag(), -1.25);

  // Signed/unsigned mixing: the scalar converts into the promoted complex type, so the negative
  // imaginary part survives rather than wrapping through an unsigned intermediate.
  cytnx_complex128 unsigned_scaled = z * cytnx_uint32{4};
  EXPECT_DOUBLE_EQ(unsigned_scaled.real(), 10.0);
  EXPECT_DOUBLE_EQ(unsigned_scaled.imag(), -5.0);

  cytnx_complex128 halved = z / cytnx_int16{2};
  EXPECT_DOUBLE_EQ(halved.real(), 1.25);
  EXPECT_DOUBLE_EQ(halved.imag(), -0.625);

  // bool participates as 0/1.
  cytnx_complex64 w(1.5f, 0.5f);
  cytnx_complex64 kept = w * true;
  EXPECT_FLOAT_EQ(kept.real(), 1.5f);
  EXPECT_FLOAT_EQ(kept.imag(), 0.5f);
}

TEST(ComplexScalarArithmetic, EqualityAgainstScalars) {
  cytnx_complex128 zero(0, 0);
  cytnx_complex128 real_only(3.0, 0);
  cytnx_complex128 with_imag(3.0, 1.0);

  // `z == 0` compares against an int; std::complex cannot deduce a single T for it.
  EXPECT_TRUE(zero == 0);
  EXPECT_FALSE(real_only == 0);
  EXPECT_TRUE(real_only == 3);
  // A nonzero imaginary part must never compare equal to a real scalar.
  EXPECT_FALSE(with_imag == 3);

  // Reversed (synthesized) and negated (rewritten) forms.
  EXPECT_TRUE(0 == zero);
  EXPECT_TRUE(3 == real_only);
  EXPECT_TRUE(with_imag != 3);
  EXPECT_FALSE(zero != 0);

  // Across complex precisions, and against a scalar of a different precision than the complex.
  cytnx_complex64 w(3.0f, 0.0f);
  EXPECT_TRUE(cytnx_complex128(3.0, 0.0) == w);
  EXPECT_TRUE(w == cytnx_double{3.0});
  EXPECT_FALSE(w == cytnx_double{3.5});
}
