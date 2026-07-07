#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/Scalar.hpp"
#include "backend/Storage.hpp"
#include "gtest/gtest.h"

// This suite absorbs and extends PR #937's tests/Scalar_test.cpp (credit to
// @ianmccul for the original guard semantics and test coverage). #937 disabled
// a subset of lossy Scalar arithmetic by throwing std::logic_error; this
// refactor (#847/#935) replaces the PIMPL+virtual-dispatch Scalar with
// std::variant, and implements the maintainer's Ruling 1: in-place ops throw
// cytnx::error (a std::logic_error) whenever the RHS dtype does not
// losslessly convert into the LHS dtype (checked via
// Type.type_promote(lhs, rhs) == lhs.dtype()). Same-dtype and lossless
// widening in-place ops (Int64 += Int32, Double += Float, ComplexDouble +=
// Double, ...) remain valid -- this is strictly more permissive than #937's
// original "any differing integer dtype throws" guard, while still rejecting
// every case #937 targeted (signed/unsigned mixes, int-with-float,
// real-with-complex).

namespace {

  using cytnx::cytnx_bool;
  using cytnx::cytnx_complex128;
  using cytnx::cytnx_complex64;
  using cytnx::cytnx_double;
  using cytnx::cytnx_float;
  using cytnx::cytnx_int16;
  using cytnx::cytnx_int32;
  using cytnx::cytnx_int64;
  using cytnx::cytnx_uint16;
  using cytnx::cytnx_uint32;
  using cytnx::cytnx_uint64;
  using cytnx::Scalar;
  using cytnx::Type;
  using cytnx::Type_class;

  void ExpectDoubleScalarEq(const Scalar &value, double expected) {
    EXPECT_EQ(value.dtype(), Type.Double);
    EXPECT_DOUBLE_EQ(static_cast<cytnx_double>(value), expected);
  }

  void ExpectInt64ScalarEq(const Scalar &value, cytnx_int64 expected) {
    EXPECT_EQ(value.dtype(), Type.Int64);
    EXPECT_EQ(static_cast<cytnx_int64>(value), expected);
  }

  void ExpectUint64ScalarEq(const Scalar &value, cytnx_uint64 expected) {
    EXPECT_EQ(value.dtype(), Type.Uint64);
    EXPECT_EQ(static_cast<cytnx_uint64>(value), expected);
  }

  void ExpectComplexDoubleScalarEq(const Scalar &value, const cytnx_complex128 &expected) {
    EXPECT_EQ(value.dtype(), Type.ComplexDouble);
    const auto got = cytnx::complex128(value);
    EXPECT_DOUBLE_EQ(got.real(), expected.real());
    EXPECT_DOUBLE_EQ(got.imag(), expected.imag());
  }

  template <typename Fn>
  void ExpectThrows(Fn &&fn) {
    EXPECT_THROW({ fn(); }, std::logic_error);
  }

  // The full list of non-Void dtypes, in the same order the 11x11 promotion
  // matrix below iterates them.
  const std::vector<unsigned int> kAllDtypes = {
    Type.ComplexDouble, Type.ComplexFloat, Type.Double, Type.Float,  Type.Int64, Type.Uint64,
    Type.Int32,         Type.Uint32,       Type.Int16,  Type.Uint16, Type.Bool,
  };

  // A representative nonzero value for each dtype, chosen so that +,-,*,/ all
  // produce exact (no-rounding-ambiguity) results when both operands are
  // small integers, and so that division never hits zero.
  Scalar RepresentativeValue(unsigned int dtype) {
    if (dtype == Type.ComplexDouble) return Scalar(cytnx_complex128(6.0, 2.0));
    if (dtype == Type.ComplexFloat) return Scalar(cytnx_complex64(6.0, 2.0));
    if (dtype == Type.Double) return Scalar(cytnx_double(6.0));
    if (dtype == Type.Float) return Scalar(cytnx_float(6.0));
    if (dtype == Type.Int64) return Scalar(cytnx_int64(6));
    if (dtype == Type.Uint64) return Scalar(cytnx_uint64(6));
    if (dtype == Type.Int32) return Scalar(cytnx_int32(6));
    if (dtype == Type.Uint32) return Scalar(cytnx_uint32(6));
    if (dtype == Type.Int16) return Scalar(cytnx_int16(6));
    if (dtype == Type.Uint16) return Scalar(cytnx_uint16(6));
    if (dtype == Type.Bool) return Scalar(cytnx_bool(true));
    cytnx_error_msg(true, "unreachable dtype in test helper%s", "\n");
    return Scalar();
  }

  Scalar RepresentativeSecondValue(unsigned int dtype) {
    if (dtype == Type.ComplexDouble) return Scalar(cytnx_complex128(3.0, 1.0));
    if (dtype == Type.ComplexFloat) return Scalar(cytnx_complex64(3.0, 1.0));
    if (dtype == Type.Double) return Scalar(cytnx_double(3.0));
    if (dtype == Type.Float) return Scalar(cytnx_float(3.0));
    if (dtype == Type.Int64) return Scalar(cytnx_int64(3));
    if (dtype == Type.Uint64) return Scalar(cytnx_uint64(3));
    if (dtype == Type.Int32) return Scalar(cytnx_int32(3));
    if (dtype == Type.Uint32) return Scalar(cytnx_uint32(3));
    if (dtype == Type.Int16) return Scalar(cytnx_int16(3));
    if (dtype == Type.Uint16) return Scalar(cytnx_uint16(3));
    if (dtype == Type.Bool) return Scalar(cytnx_bool(true));
    cytnx_error_msg(true, "unreachable dtype in test helper%s", "\n");
    return Scalar();
  }

  // Reference value as a complex128, used to check numeric correctness of the
  // promoted result regardless of which dtype it ended up in.
  cytnx_complex128 AsComplex(const Scalar &s) {
    return cytnx::complex128(s.astype(Type.ComplexDouble));
  }

}  // namespace

// ===========================================================================
// #937-derived core suite (credit: @ianmccul). Adjusted for the variant
// rewrite: `abs`/`sqrt`/bool-arithmetic are no longer blanket-disabled --
// only genuinely lossy in-place combinations throw.
// ===========================================================================

TEST(ScalarTest, FloatingBinaryArithmeticReturnsCorrectValues) {
  const Scalar a(cytnx_double(3.0));
  const Scalar b(cytnx_double(2.0));

  ExpectDoubleScalarEq(a + b, 5.0);
  ExpectDoubleScalarEq(a - b, 1.0);
  ExpectDoubleScalarEq(a * b, 6.0);
  ExpectDoubleScalarEq(a / b, 1.5);
}

TEST(ScalarTest, ComplexBinaryArithmeticReturnsCorrectValues) {
  const Scalar a(cytnx_complex128(3.0, 4.0));
  const Scalar b(cytnx_complex128(1.0, -2.0));

  ExpectComplexDoubleScalarEq(a + b, cytnx_complex128(4.0, 2.0));
  ExpectComplexDoubleScalarEq(a - b, cytnx_complex128(2.0, 6.0));
  ExpectComplexDoubleScalarEq(a * b, cytnx_complex128(11.0, -2.0));
  ExpectComplexDoubleScalarEq(a / b, cytnx_complex128(-1.0, 2.0));
}

TEST(ScalarTest, FloatingInPlaceArithmeticReturnsCorrectValues) {
  Scalar value(cytnx_double(3.0));
  value += Scalar(cytnx_double(2.0));
  ExpectDoubleScalarEq(value, 5.0);
  value -= Scalar(cytnx_double(1.0));
  ExpectDoubleScalarEq(value, 4.0);
  value *= Scalar(cytnx_double(3.0));
  ExpectDoubleScalarEq(value, 12.0);
  value /= Scalar(cytnx_double(4.0));
  ExpectDoubleScalarEq(value, 3.0);
}

TEST(ScalarTest, IntegerConstructionAndConversionStillWork) {
  EXPECT_EQ(Scalar(cytnx_int64(-3)).dtype(), Type.Int64);
  EXPECT_EQ(static_cast<cytnx_int64>(Scalar(cytnx_int64(-3))), cytnx_int64(-3));
  EXPECT_EQ(Scalar(cytnx_uint64(3)).dtype(), Type.Uint64);
  EXPECT_EQ(static_cast<cytnx_uint64>(Scalar(cytnx_uint64(3))), cytnx_uint64(3));
  EXPECT_EQ(Scalar(cytnx_bool(true)).dtype(), Type.Bool);
  EXPECT_EQ(static_cast<cytnx_bool>(Scalar(cytnx_bool(true))), true);
}

TEST(ScalarTest, CytnxTypeAssignmentPreservesInputDtype) {
  Scalar value(cytnx_double(3.0));

  value = cytnx_float(2.0);
  EXPECT_EQ(value.dtype(), Type.Float);
  EXPECT_FLOAT_EQ(static_cast<cytnx_float>(value), cytnx_float(2.0));

  value = cytnx_int32(-4);
  EXPECT_EQ(value.dtype(), Type.Int32);
  EXPECT_EQ(static_cast<cytnx_int32>(value), cytnx_int32(-4));
}

TEST(ScalarTest, IntegerBinaryArithmeticReturnsCorrectValues) {
  // Out-of-place arithmetic always promotes via Type.type_promote and never
  // throws (unlike #937's guard, which conservatively disabled all
  // integer-result arithmetic; now that promotion is centralized on
  // type_promote, these are simply correct).
  ExpectDoubleScalarEq(Scalar(cytnx_int64(3)) + Scalar(cytnx_double(2.0)), 5.0);
  ExpectDoubleScalarEq(Scalar(cytnx_int64(3)) - Scalar(cytnx_double(2.0)), 1.0);
  ExpectDoubleScalarEq(Scalar(cytnx_int64(3)) * Scalar(cytnx_double(2.0)), 6.0);
  ExpectDoubleScalarEq(Scalar(cytnx_int64(3)) / Scalar(cytnx_double(2.0)), 1.5);
}

TEST(ScalarTest, SameIntegerDtypeBinaryArithmeticReturnsCorrectValues) {
  ExpectInt64ScalarEq(Scalar(cytnx_int64(3)) + Scalar(cytnx_int64(2)), cytnx_int64(5));
  ExpectUint64ScalarEq(Scalar(cytnx_uint64(3)) - Scalar(cytnx_uint64(2)), cytnx_uint64(1));
  EXPECT_EQ(static_cast<cytnx_int32>(Scalar(cytnx_int32(3)) * Scalar(cytnx_int32(2))),
            cytnx_int32(6));
  EXPECT_EQ(static_cast<cytnx_uint32>(Scalar(cytnx_uint32(3)) / Scalar(cytnx_uint32(2))),
            cytnx_uint32(1));
  EXPECT_EQ(static_cast<cytnx_int16>(Scalar(cytnx_int16(3)) + Scalar(cytnx_int16(2))),
            cytnx_int16(5));
  EXPECT_EQ(static_cast<cytnx_uint16>(Scalar(cytnx_uint16(3)) - Scalar(cytnx_uint16(2))),
            cytnx_uint16(1));
}

TEST(ScalarTest, MixedIntegerBinaryArithmeticPromotesCorrectly) {
  // Same-signedness mixed-width integers promote to the wider of the two
  // operand types (out of place always succeeds, matching Type.type_promote
  // exactly -- the wider type is not always Int64/Uint64, it is whichever of
  // the two inputs has more precision).
  ExpectInt64ScalarEq(Scalar(cytnx_int64(3)) + Scalar(cytnx_int32(2)), cytnx_int64(5));
  Scalar r = Scalar(cytnx_int32(3)) * Scalar(cytnx_int16(2));
  EXPECT_EQ(r.dtype(), Type.Int32);
  EXPECT_EQ(static_cast<cytnx_int32>(r), cytnx_int32(6));
  Scalar r2 = Scalar(cytnx_uint32(3)) / Scalar(cytnx_uint16(2));
  EXPECT_EQ(r2.dtype(), Type.Uint32);
  EXPECT_EQ(static_cast<cytnx_uint32>(r2), cytnx_uint32(1));
}

TEST(ScalarTest, MixedSignedUnsignedIntegerBinaryArithmeticPromotesPerTypePromote) {
  // Out-of-place arithmetic between signed/unsigned integers never throws:
  // it promotes via Type.type_promote (which resolves the mix to the signed
  // type when widths differ, or matches the documented promotion table).
  Scalar r1 = Scalar(cytnx_int64(3)) + Scalar(cytnx_uint64(2));
  EXPECT_EQ(r1.dtype(), Type_class::type_promote(Type.Int64, Type.Uint64));
  EXPECT_DOUBLE_EQ(AsComplex(r1).real(), 5.0);

  Scalar r2 = Scalar(cytnx_uint32(3)) - Scalar(cytnx_int32(2));
  EXPECT_EQ(r2.dtype(), Type_class::type_promote(Type.Uint32, Type.Int32));
  EXPECT_DOUBLE_EQ(AsComplex(r2).real(), 1.0);
}

TEST(ScalarTest, SameSignednessIntegerComparisonsReturnCorrectValues) {
  EXPECT_TRUE(Scalar(cytnx_int64(3)) > Scalar(cytnx_int32(2)));
  EXPECT_TRUE(Scalar(cytnx_int32(3)) >= Scalar(cytnx_int16(3)));
  EXPECT_TRUE(Scalar(cytnx_uint64(2)) < Scalar(cytnx_uint32(3)));
  EXPECT_TRUE(Scalar(cytnx_uint32(3)) <= Scalar(cytnx_uint16(3)));
  EXPECT_TRUE(Scalar(cytnx_int16(3)) == Scalar(cytnx_int64(3)));
}

TEST(ScalarTest, BoolBinaryArithmeticPromotesToBool) {
  // Bool + Bool stays Bool under Type.type_promote; out-of-place arithmetic
  // never throws.
  Scalar r = Scalar(cytnx_bool(true)) + Scalar(cytnx_bool(false));
  EXPECT_EQ(r.dtype(), Type.Bool);
}

TEST(ScalarTest, SameIntegerDtypeInPlaceArithmeticReturnsCorrectValues) {
  Scalar value(cytnx_int64(3));
  value += Scalar(cytnx_int64(2));
  ExpectInt64ScalarEq(value, cytnx_int64(5));
  value -= Scalar(cytnx_int64(1));
  ExpectInt64ScalarEq(value, cytnx_int64(4));
  value *= Scalar(cytnx_int64(3));
  ExpectInt64ScalarEq(value, cytnx_int64(12));
  value /= Scalar(cytnx_int64(5));
  ExpectInt64ScalarEq(value, cytnx_int64(2));

  Scalar unsigned_value(cytnx_uint64(3));
  unsigned_value -= Scalar(cytnx_uint64(2));
  ExpectUint64ScalarEq(unsigned_value, cytnx_uint64(1));

  Scalar int32_value(cytnx_int32(3));
  int32_value *= Scalar(cytnx_int32(2));
  EXPECT_EQ(static_cast<cytnx_int32>(int32_value), cytnx_int32(6));
}

TEST(ScalarTest, LosslessWideningInPlaceArithmeticIsAllowed) {
  // Ruling 1 (#935/#937): in-place ops are allowed whenever
  // Type.type_promote(lhs, rhs) == lhs.dtype() -- i.e. the RHS converts
  // losslessly into the LHS's dtype. This is strictly more permissive than
  // #937's original guard (which disabled ALL differing-dtype integer
  // in-place ops); Int64 += Int32 is a same-signedness lossless widening and
  // must remain valid.
  Scalar i64(cytnx_int64(3));
  i64 += Scalar(cytnx_int32(2));
  ExpectInt64ScalarEq(i64, cytnx_int64(5));

  Scalar d(cytnx_double(3.0));
  d += Scalar(cytnx_float(2.0));
  ExpectDoubleScalarEq(d, 5.0);

  Scalar cd(cytnx_complex128(3.0, 0.0));
  cd += Scalar(cytnx_double(2.0));
  ExpectComplexDoubleScalarEq(cd, cytnx_complex128(5.0, 0.0));

  Scalar u64(cytnx_uint64(3));
  u64 += Scalar(cytnx_bool(true));
  ExpectUint64ScalarEq(u64, cytnx_uint64(4));
}

TEST(ScalarTest, MixedSignedUnsignedIntegerInPlaceArithmeticIsDisabled) {
  // Under Type.type_promote, a signed/unsigned mix of equal width promotes
  // to the *signed* type (see Type.hpp's type_promote implementation and
  // Type_test.cpp's documentation of it). So when the unsigned dtype is the
  // in-place LHS, promotion picks the *other* (signed) dtype, which differs
  // from the LHS -> throws. (When the signed dtype is the LHS, promotion
  // stays at the LHS's own dtype, so it is allowed -- see
  // LosslessWideningInPlaceArithmeticIsAllowed.)
  ExpectThrows([] {
    Scalar value(cytnx_uint64(3));
    value -= Scalar(cytnx_int64(2));
  });
  ExpectThrows([] {
    Scalar value(cytnx_uint32(3));
    value -= Scalar(cytnx_int32(2));
  });
  ExpectThrows([] {
    Scalar value(cytnx_uint16(3));
    value *= Scalar(cytnx_int16(2));
  });
}

TEST(ScalarTest, IntWithFloatInPlaceArithmeticIsDisabled) {
  ExpectThrows([] {
    Scalar value(cytnx_int64(3));
    value += Scalar(cytnx_double(2.0));
  });
  ExpectThrows([] {
    Scalar value(cytnx_uint32(3));
    value /= Scalar(cytnx_double(2.0));
  });
}

TEST(ScalarTest, RealWithComplexInPlaceArithmeticIsDisabled) {
  ExpectThrows([] {
    Scalar value(cytnx_double(3.0));
    value += Scalar(cytnx_complex128(2.0, 1.0));
  });
  ExpectThrows([] {
    Scalar value(cytnx_float(3.0));
    value *= Scalar(cytnx_complex64(2.0, 1.0));
  });
}

TEST(ScalarTest, BoolInPlaceArithmeticWithNonBoolIsDisabled) {
  ExpectThrows([] {
    Scalar value(cytnx_bool(true));
    value += Scalar(cytnx_int64(1));
  });
}

TEST(ScalarTest, SelfAssignmentPreservesValue) {
  Scalar value(cytnx_double(3.5));
  value = value;
  ExpectDoubleScalarEq(value, 3.5);
}

TEST(ScalarTest, SelfInPlaceArithmeticIsSafe) {
  // Exercises the self-assignment/self-arithmetic path that used to be UB
  // under the old hand-written copy-assignment operator (#935).
  Scalar value(cytnx_double(3.0));
  value += value;
  ExpectDoubleScalarEq(value, 6.0);
}

// ===========================================================================
// #935 fix coverage
// ===========================================================================

TEST(ScalarTest, CopyAssignmentIsDefaultedValueSemantics) {
  Scalar a(cytnx_int64(7));
  Scalar b = a;
  b += Scalar(cytnx_int64(1));
  // b is an independent copy: mutating b must not affect a.
  ExpectInt64ScalarEq(a, cytnx_int64(7));
  ExpectInt64ScalarEq(b, cytnx_int64(8));
}

TEST(ScalarTest, MoveConstructionLeavesSourceValid) {
  Scalar a(cytnx_double(9.0));
  Scalar b = std::move(a);
  ExpectDoubleScalarEq(b, 9.0);
}

TEST(ScalarTest, MinvalForFloatingDtypesUsesLowest) {
  // #935: minval used numeric_limits<T>::min() (smallest positive normal),
  // not lowest() (most negative finite value).
  Scalar min_d = Scalar::minval(Type.Double);
  EXPECT_DOUBLE_EQ(static_cast<cytnx_double>(min_d), std::numeric_limits<cytnx_double>::lowest());
  EXPECT_LT(static_cast<cytnx_double>(min_d), 0.0);

  Scalar min_f = Scalar::minval(Type.Float);
  EXPECT_FLOAT_EQ(static_cast<cytnx_float>(min_f), std::numeric_limits<cytnx_float>::lowest());
  EXPECT_LT(static_cast<cytnx_float>(min_f), 0.0f);
}

TEST(ScalarTest, MaxvalForFloatingDtypesUsesMax) {
  Scalar max_d = Scalar::maxval(Type.Double);
  EXPECT_DOUBLE_EQ(static_cast<cytnx_double>(max_d), std::numeric_limits<cytnx_double>::max());
}

TEST(ScalarTest, MinvalMaxvalForIntegerDtypes) {
  EXPECT_EQ(static_cast<cytnx_int64>(Scalar::minval(Type.Int64)),
            std::numeric_limits<cytnx_int64>::min());
  EXPECT_EQ(static_cast<cytnx_int64>(Scalar::maxval(Type.Int64)),
            std::numeric_limits<cytnx_int64>::max());
  EXPECT_EQ(static_cast<cytnx_uint64>(Scalar::minval(Type.Uint64)),
            std::numeric_limits<cytnx_uint64>::min());
}

TEST(ScalarTest, AbsReturnsRealDtypeForComplex) {
  // #935: abs() must return a real dtype, not a complex dtype with zero
  // imaginary part.
  Scalar z(cytnx_complex128(3.0, 4.0));
  Scalar a = z.abs();
  EXPECT_EQ(a.dtype(), Type.Double);
  EXPECT_DOUBLE_EQ(static_cast<cytnx_double>(a), 5.0);

  Scalar zf(cytnx_complex64(3.0f, 4.0f));
  Scalar af = zf.abs();
  EXPECT_EQ(af.dtype(), Type.Float);
  EXPECT_FLOAT_EQ(static_cast<cytnx_float>(af), 5.0f);
}

TEST(ScalarTest, IabsOnComplexKeepsDtypeAndMatchesAbsMagnitude) {
  // #935 asked for iabs()/abs() to be made *consistent*. Since no in-place
  // op is allowed to change dtype (that is the whole point of "in-place"),
  // iabs() on a complex Scalar keeps the ComplexDouble/ComplexFloat dtype
  // and stores the magnitude as the real part with a zero imaginary part.
  // Its *value* (the magnitude) matches abs()'s value exactly; only the
  // dtype differs (abs() narrows to real, iabs() cannot narrow in place).
  Scalar z(cytnx_complex128(3.0, 4.0));
  Scalar reference_abs = z.abs();
  z.iabs();
  EXPECT_EQ(z.dtype(), Type.ComplexDouble);
  cytnx_complex128 v = cytnx::complex128(z);
  EXPECT_DOUBLE_EQ(v.real(), static_cast<cytnx_double>(reference_abs));
  EXPECT_DOUBLE_EQ(v.imag(), 0.0);
}

TEST(ScalarTest, AbsOnUnsignedIsIdentity) {
  Scalar u(cytnx_uint64(42));
  Scalar a = u.abs();
  EXPECT_EQ(a.dtype(), Type.Uint64);
  EXPECT_EQ(static_cast<cytnx_uint64>(a), cytnx_uint64(42));
}

TEST(ScalarTest, PromotionCentralizedOnTypePromoteForComplexFloatDoubleMix) {
  // #935: ComplexFloat + Double must promote to ComplexDouble, not
  // ComplexFloat (the old hand-rolled enum-order promotion picked the wrong
  // one because the enum interleaves complexness and precision).
  Scalar cf(cytnx_complex64(1.0f, 1.0f));
  Scalar d(cytnx_double(2.0));
  Scalar r = cf + d;
  EXPECT_EQ(r.dtype(), Type.ComplexDouble);
}

// ===========================================================================
// Programmatic 11x11 promotion matrix for +, -, *, / and comparisons.
// Loops over every ordered pair of the 11 non-Void dtypes, asserting:
//   - out-of-place binary op result dtype == Type.type_promote(a, b)
//   - the numeric value is correct (checked via complex128 widening)
//   - comparisons agree with the complex128-widened reference; mixed
//     signed/unsigned integer operands promote via Type.type_promote and
//     compare normally (they must NOT throw). Only ordering comparisons
//     (< <= > >=) with a complex operand throw (complex has no total
//     order); complex equality is allowed.
// ===========================================================================

class ScalarPromotionMatrixTest
    : public ::testing::TestWithParam<std::pair<unsigned int, unsigned int>> {};

TEST_P(ScalarPromotionMatrixTest, ArithmeticResultDtypeAndValueMatchTypePromote) {
  const unsigned int lt = GetParam().first;
  const unsigned int rt = GetParam().second;
  const Scalar a = RepresentativeValue(lt);
  const Scalar b = RepresentativeSecondValue(rt);
  const unsigned int expected_dtype = Type_class::type_promote(lt, rt);

  const cytnx_complex128 av = AsComplex(a);
  const cytnx_complex128 bv = AsComplex(b);

  // Exact value checks only make sense when the promoted dtype can
  // represent the mathematically-exact result without truncation,
  // saturation, or wraparound:
  //  - Bool saturates (true+true == true, not 2) -- never exact-comparable.
  //  - Unsigned dtypes wrap around on results that go negative
  //    mathematically (e.g. bool(1) - uint64(3)) -- not exact-comparable for
  //    -,  since our representative values (6, 3) can produce a negative
  //    mathematical difference when the Bool representative (1) is
  //    involved.
  const bool result_is_bool = (expected_dtype == Type.Bool);
  const bool result_is_unsigned = Type.is_unsigned(expected_dtype);

  {
    Scalar r = a + b;
    EXPECT_EQ(r.dtype(), expected_dtype) << Type.getname(lt) << " + " << Type.getname(rt);
    if (!result_is_bool) {
      cytnx_complex128 got = AsComplex(r);
      EXPECT_NEAR(got.real(), (av + bv).real(), 1e-6);
      EXPECT_NEAR(got.imag(), (av + bv).imag(), 1e-6);
    }
  }
  {
    Scalar r = a - b;
    EXPECT_EQ(r.dtype(), expected_dtype) << Type.getname(lt) << " - " << Type.getname(rt);
    const bool would_underflow_unsigned = result_is_unsigned && (av - bv).real() < 0;
    if (!result_is_bool && !would_underflow_unsigned) {
      cytnx_complex128 got = AsComplex(r);
      EXPECT_NEAR(got.real(), (av - bv).real(), 1e-6);
      EXPECT_NEAR(got.imag(), (av - bv).imag(), 1e-6);
    }
  }
  {
    Scalar r = a * b;
    EXPECT_EQ(r.dtype(), expected_dtype) << Type.getname(lt) << " * " << Type.getname(rt);
    if (!result_is_bool) {
      cytnx_complex128 got = AsComplex(r);
      EXPECT_NEAR(got.real(), (av * bv).real(), 1e-6);
      EXPECT_NEAR(got.imag(), (av * bv).imag(), 1e-6);
    }
  }
  {
    Scalar r = a / b;
    EXPECT_EQ(r.dtype(), expected_dtype) << Type.getname(lt) << " / " << Type.getname(rt);
    cytnx_complex128 got = AsComplex(r);
    cytnx_complex128 expected = av / bv;
    // Integer division truncates; only compare exactly for float/complex
    // promoted results, otherwise just check the promoted dtype and that no
    // exception was thrown (integer truncation/saturation is
    // dtype-appropriate).
    if (Type.is_float(expected_dtype)) {
      EXPECT_NEAR(got.real(), expected.real(), 1e-6);
      EXPECT_NEAR(got.imag(), expected.imag(), 1e-6);
    }
  }
}

TEST_P(ScalarPromotionMatrixTest, EqualityMatchesComplexWidenedReference) {
  const unsigned int lt = GetParam().first;
  const unsigned int rt = GetParam().second;
  const Scalar a = RepresentativeValue(lt);
  const Scalar b = RepresentativeValue(rt);  // same representative -> equal magnitude pattern
  bool expect_eq = (AsComplex(a) == AsComplex(b));
  EXPECT_EQ(a.eq(b), expect_eq) << Type.getname(lt) << " == " << Type.getname(rt);
}

TEST_P(ScalarPromotionMatrixTest, OrderingComparisonThrowsOnlyForComplexOtherwiseMatches) {
  // Ordering comparisons (< <= > >=) promote via Type.type_promote just like
  // arithmetic, and are well-defined for any real (non-complex) promoted
  // dtype -- including mixed signed/unsigned integers, which promote
  // unambiguously to a definite dtype before comparing. Only complex
  // operands are rejected (complex has no total order).
  const unsigned int lt = GetParam().first;
  const unsigned int rt = GetParam().second;
  const Scalar a = RepresentativeValue(lt);
  const Scalar b = RepresentativeSecondValue(rt);

  const bool involves_complex = Type.is_complex(lt) || Type.is_complex(rt);

  if (involves_complex) {
    ExpectThrows([&] { return a < b; });
  } else {
    bool expected = AsComplex(a).real() < AsComplex(b).real();
    EXPECT_EQ(a < b, expected) << Type.getname(lt) << " < " << Type.getname(rt);
  }
}

TEST(ScalarTest, MixedSignedUnsignedEqualityFollowsTypePromoteTableQuirk) {
  // Pins a known consequence of the sanctioned type_promote table (#982):
  // type_promote(Int64, Uint64) == Int64, so comparing Int64(-1) against
  // Uint64 max converts the unsigned value into Int64 (wrapping to -1) and
  // the equality evaluates TRUE, even though the two values are
  // mathematically different. This is intentional, per explicit maintainer
  // ruling (2026-07-07): Scalar comparisons and arithmetic follow the
  // type_promote table uniformly (in-place and out-of-place consistent);
  // any future tightening of this mixed-signedness behavior happens in the
  // table itself, not in Scalar. (#937 had instead disabled mixed
  // signed/unsigned comparisons outright; that guard is superseded.)
  const Scalar a(cytnx_int64(-1));
  const Scalar b(std::numeric_limits<cytnx_uint64>::max());
  EXPECT_TRUE(a == b);
}

std::vector<std::pair<unsigned int, unsigned int>> AllDtypePairs() {
  std::vector<std::pair<unsigned int, unsigned int>> pairs;
  for (unsigned int lt : kAllDtypes) {
    for (unsigned int rt : kAllDtypes) {
      pairs.emplace_back(lt, rt);
    }
  }
  return pairs;
}

// Short, unambiguous per-dtype tag for parameterized-test names (the full
// Type.getname() strings contain spaces/parentheses and collide when
// truncated).
std::string DtypeShortName(unsigned int dtype) {
  if (dtype == Type.ComplexDouble) return "CD";
  if (dtype == Type.ComplexFloat) return "CF";
  if (dtype == Type.Double) return "F64";
  if (dtype == Type.Float) return "F32";
  if (dtype == Type.Int64) return "I64";
  if (dtype == Type.Uint64) return "U64";
  if (dtype == Type.Int32) return "I32";
  if (dtype == Type.Uint32) return "U32";
  if (dtype == Type.Int16) return "I16";
  if (dtype == Type.Uint16) return "U16";
  if (dtype == Type.Bool) return "B";
  return "X" + std::to_string(dtype);
}

std::string DtypePairTestName(
  const ::testing::TestParamInfo<std::pair<unsigned int, unsigned int>> &info) {
  return DtypeShortName(info.param.first) + "_" + DtypeShortName(info.param.second);
}

INSTANTIATE_TEST_SUITE_P(AllDtypeCombinations, ScalarPromotionMatrixTest,
                         ::testing::ValuesIn(AllDtypePairs()), DtypePairTestName);

// ===========================================================================
// In-place throw matrix: for every ordered dtype pair, in-place += must
// throw iff Type.type_promote(lhs, rhs) != lhs.dtype().
// ===========================================================================

class ScalarInplaceThrowMatrixTest
    : public ::testing::TestWithParam<std::pair<unsigned int, unsigned int>> {};

TEST_P(ScalarInplaceThrowMatrixTest, InplaceAddThrowsExactlyWhenLossy) {
  const unsigned int lt = GetParam().first;
  const unsigned int rt = GetParam().second;
  const bool should_throw = Type_class::type_promote(lt, rt) != lt;

  Scalar lhs = RepresentativeValue(lt);
  Scalar rhs = RepresentativeSecondValue(rt);

  if (should_throw) {
    ExpectThrows([&] { lhs += rhs; });
  } else {
    EXPECT_NO_THROW({ lhs += rhs; }) << Type.getname(lt) << " += " << Type.getname(rt);
    EXPECT_EQ(lhs.dtype(), lt);
  }
}

INSTANTIATE_TEST_SUITE_P(AllDtypeCombinations, ScalarInplaceThrowMatrixTest,
                         ::testing::ValuesIn(AllDtypePairs()), DtypePairTestName);

// ===========================================================================
// Storage::set_item(idx, Scalar) / Sproxy round-trip
// ===========================================================================

TEST(ScalarStorageSproxyTest, SetItemViaSproxyRoundTripsAcrossDtypes) {
  for (unsigned int dt : kAllDtypes) {
    cytnx::Storage s(4, dt);
    Scalar v = RepresentativeValue(dt);
    s(1) = v;
    Scalar got = s(1);
    EXPECT_EQ(got.dtype(), dt) << Type.getname(dt);
    EXPECT_TRUE(got.eq(v)) << Type.getname(dt);
  }
}

TEST(ScalarStorageSproxyTest, SetItemFromCrossDtypeScalarConvertsIntoStorageDtype) {
  cytnx::Storage s(2, Type.Double);
  s(0) = Scalar(cytnx_int64(7));
  Scalar got = s(0);
  EXPECT_EQ(got.dtype(), Type.Double);
  EXPECT_DOUBLE_EQ(static_cast<cytnx_double>(got), 7.0);
}

TEST(ScalarStorageSproxyTest, SetItemFromVoidScalarThrows) {
  cytnx::Storage s(2, Type.Double);
  Scalar void_scalar;
  EXPECT_THROW({ s(0) = void_scalar; }, std::logic_error);
}
