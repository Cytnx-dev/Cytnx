#include <complex>
#include <stdexcept>

#include "backend/Scalar.hpp"
#include "gtest/gtest.h"

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
  void ExpectCorrectDoubleOrDisabled(Fn &&fn, double expected) {
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    try {
      const Scalar value = fn();
      static_cast<void>(testing::internal::GetCapturedStderr());
      static_cast<void>(testing::internal::GetCapturedStdout());
      ExpectDoubleScalarEq(value, expected);
    } catch (const std::logic_error &) {
      static_cast<void>(testing::internal::GetCapturedStderr());
      static_cast<void>(testing::internal::GetCapturedStdout());
      SUCCEED();
    }
  }

  template <typename Fn>
  void ExpectCorrectInt64OrDisabled(Fn &&fn, cytnx_int64 expected) {
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    try {
      const Scalar value = fn();
      static_cast<void>(testing::internal::GetCapturedStderr());
      static_cast<void>(testing::internal::GetCapturedStdout());
      EXPECT_EQ(static_cast<cytnx_int64>(value), expected);
    } catch (const std::logic_error &) {
      static_cast<void>(testing::internal::GetCapturedStderr());
      static_cast<void>(testing::internal::GetCapturedStdout());
      SUCCEED();
    }
  }

  template <typename Fn>
  void ExpectDisabled(Fn &&fn) {
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    try {
      static_cast<void>(fn());
      static_cast<void>(testing::internal::GetCapturedStderr());
      static_cast<void>(testing::internal::GetCapturedStdout());
      FAIL() << "Expected std::logic_error.";
    } catch (const std::logic_error &) {
      static_cast<void>(testing::internal::GetCapturedStderr());
      static_cast<void>(testing::internal::GetCapturedStdout());
      SUCCEED();
    }
  }

}  // namespace

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

TEST(ScalarTest, IntegerBinaryArithmeticIsCorrectOrDisabled) {
  ExpectCorrectDoubleOrDisabled([] { return Scalar(cytnx_int64(3)) + Scalar(cytnx_double(2.0)); },
                                5.0);
  ExpectCorrectDoubleOrDisabled([] { return Scalar(cytnx_int64(3)) - Scalar(cytnx_double(2.0)); },
                                1.0);
  ExpectCorrectDoubleOrDisabled([] { return Scalar(cytnx_int64(3)) * Scalar(cytnx_double(2.0)); },
                                6.0);
  ExpectCorrectDoubleOrDisabled([] { return Scalar(cytnx_int64(3)) / Scalar(cytnx_double(2.0)); },
                                1.5);
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

TEST(ScalarTest, MixedIntegerBinaryArithmeticIsCorrectOrDisabled) {
  ExpectCorrectInt64OrDisabled([] { return Scalar(cytnx_int64(3)) + Scalar(cytnx_int32(2)); },
                               cytnx_int64(5));
  ExpectCorrectInt64OrDisabled([] { return Scalar(cytnx_uint64(3)) - Scalar(cytnx_uint32(2)); },
                               cytnx_int64(1));
  ExpectCorrectInt64OrDisabled([] { return Scalar(cytnx_int32(3)) * Scalar(cytnx_int16(2)); },
                               cytnx_int64(6));
  ExpectCorrectInt64OrDisabled([] { return Scalar(cytnx_uint32(3)) / Scalar(cytnx_uint16(2)); },
                               cytnx_int64(1));
}

TEST(ScalarTest, MixedSignedUnsignedIntegerBinaryArithmeticIsDisabled) {
  ExpectDisabled([] { return Scalar(cytnx_int64(3)) + Scalar(cytnx_uint64(2)); });
  ExpectDisabled([] { return Scalar(cytnx_uint32(3)) - Scalar(cytnx_int32(2)); });
  ExpectDisabled([] { return Scalar(cytnx_int16(3)) * Scalar(cytnx_uint16(2)); });
}

TEST(ScalarTest, SameSignednessIntegerComparisonsReturnCorrectValues) {
  EXPECT_TRUE(Scalar(cytnx_int64(3)) > Scalar(cytnx_int32(2)));
  EXPECT_TRUE(Scalar(cytnx_int32(3)) >= Scalar(cytnx_int16(3)));
  EXPECT_TRUE(Scalar(cytnx_uint64(2)) < Scalar(cytnx_uint32(3)));
  EXPECT_TRUE(Scalar(cytnx_uint32(3)) <= Scalar(cytnx_uint16(3)));
  EXPECT_TRUE(Scalar(cytnx_int16(3)) == Scalar(cytnx_int64(3)));
}

TEST(ScalarTest, MixedSignedUnsignedIntegerComparisonsAreDisabled) {
  ExpectDisabled([] { return Scalar(cytnx_uint64(3)) < Scalar(cytnx_int64(4)); });
  ExpectDisabled([] { return Scalar(cytnx_int64(-1)) == Scalar(cytnx_uint64(~cytnx_uint64(0))); });
  ExpectDisabled([] { return Scalar(cytnx_uint32(3)) >= Scalar(cytnx_int32(2)); });
}

TEST(ScalarTest, BoolBinaryArithmeticIsDisabled) {
  ExpectDisabled([] { return Scalar(cytnx_bool(true)) + Scalar(cytnx_bool(false)); });
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

TEST(ScalarTest, MixedIntegerResultInPlaceArithmeticIsDisabled) {
  ExpectDisabled([] {
    Scalar value(cytnx_int64(3));
    value += Scalar(cytnx_double(2.0));
  });
  ExpectDisabled([] {
    Scalar value(cytnx_uint64(3));
    value -= Scalar(cytnx_int64(2));
  });
  ExpectDisabled([] {
    Scalar value(cytnx_int32(3));
    value *= Scalar(cytnx_int16(2));
  });
  ExpectDisabled([] {
    Scalar value(cytnx_uint32(3));
    value /= Scalar(cytnx_double(2.0));
  });
}

TEST(ScalarTest, IntegerAbsAndSqrtAreDisabled) {
  ExpectDisabled([] { return Scalar(cytnx_int64(-3)).abs(); });
  ExpectDisabled([] { return Scalar(cytnx_uint64(3)).sqrt(); });
  ExpectDisabled([] {
    Scalar value(cytnx_int64(-3));
    value.iabs();
  });
  ExpectDisabled([] {
    Scalar value(cytnx_uint64(3));
    value.isqrt();
  });
}

TEST(ScalarTest, SelfAssignmentPreservesValue) {
  Scalar value(cytnx_double(3.5));
  value = value;
  ExpectDoubleScalarEq(value, 3.5);
}
