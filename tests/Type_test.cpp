#include <string>
#include <variant>

#include "gtest/gtest.h"

#include "Type.hpp"
#include "Tensor.hpp"
#include "Generator.hpp"
#include "linalg.hpp"

namespace {

  using cytnx::cytnx_bool;
  using cytnx::cytnx_double;
  using cytnx::cytnx_int64;
  using cytnx::Type_list;
  using cytnx::variant_contains_v;
  using cytnx::variant_index_v;

  static_assert(variant_contains_v<cytnx_double, Type_list>);
  static_assert(variant_contains_v<cytnx_bool, Type_list>);
  static_assert(!variant_contains_v<std::string, Type_list>);

  static_assert(variant_index_v<void, Type_list> == 0);
  static_assert(variant_index_v<cytnx_double, Type_list> == 3);
  static_assert(variant_index_v<cytnx_int64, Type_list> == 5);
  static_assert(variant_index_v<cytnx_bool, Type_list> == 11);

  // to_real/to_complex are constexpr and usable at compile time.
  static_assert(cytnx::Type_class::to_real(cytnx::Type_class::ComplexDouble) ==
                cytnx::Type_class::Double);
  static_assert(cytnx::Type_class::to_complex(cytnx::Type_class::Float) ==
                cytnx::Type_class::ComplexFloat);

}  // namespace

TEST(TypeTest, VariantContainsRecognizesSupportedTypes) {
  EXPECT_TRUE((variant_contains_v<cytnx_double, Type_list>));
  EXPECT_TRUE((variant_contains_v<cytnx_bool, Type_list>));
  EXPECT_FALSE((variant_contains_v<std::string, Type_list>));
}

TEST(TypeTest, ToRealMapsToRealCounterpart) {
  EXPECT_EQ(cytnx::Type.to_real(cytnx::Type.ComplexDouble), cytnx::Type.Double);
  EXPECT_EQ(cytnx::Type.to_real(cytnx::Type.ComplexFloat), cytnx::Type.Float);
  EXPECT_EQ(cytnx::Type.to_real(cytnx::Type.Double), cytnx::Type.Double);
  EXPECT_EQ(cytnx::Type.to_real(cytnx::Type.Int64), cytnx::Type.Int64);
}

TEST(TypeTest, ToComplexMapsToComplexCounterpart) {
  EXPECT_EQ(cytnx::Type.to_complex(cytnx::Type.Double), cytnx::Type.ComplexDouble);
  EXPECT_EQ(cytnx::Type.to_complex(cytnx::Type.Float), cytnx::Type.ComplexFloat);
  EXPECT_EQ(cytnx::Type.to_complex(cytnx::Type.ComplexFloat), cytnx::Type.ComplexFloat);
  EXPECT_EQ(cytnx::Type.to_complex(cytnx::Type.Int64), cytnx::Type.ComplexDouble);
}

TEST(TypeTest, PromoteMixedComplexRealUsesMaxPrecision) {
  EXPECT_EQ(cytnx::Type.type_promote(cytnx::Type.ComplexFloat, cytnx::Type.Double),
            cytnx::Type.ComplexDouble);
  EXPECT_EQ(cytnx::Type.type_promote(cytnx::Type.Double, cytnx::Type.ComplexFloat),
            cytnx::Type.ComplexDouble);
  EXPECT_EQ(cytnx::Type.type_promote(cytnx::Type.ComplexFloat, cytnx::Type.Float),
            cytnx::Type.ComplexFloat);
  EXPECT_EQ(cytnx::Type.type_promote(cytnx::Type.ComplexDouble, cytnx::Type.Float),
            cytnx::Type.ComplexDouble);
  // unchanged same-kind rules
  EXPECT_EQ(cytnx::Type.type_promote(cytnx::Type.Double, cytnx::Type.Float), cytnx::Type.Double);
  EXPECT_EQ(cytnx::Type.type_promote(cytnx::Type.ComplexFloat, cytnx::Type.Int64),
            cytnx::Type.ComplexFloat);
  EXPECT_EQ(cytnx::Type.type_promote(cytnx::Type.Int64, cytnx::Type.Uint64), cytnx::Type.Int64);
  // unsigned/signed adjustment: promote to the next-wider signed type
  EXPECT_EQ(cytnx::Type.type_promote(cytnx::Type.Uint64, cytnx::Type.Int32), cytnx::Type.Int64);
}

TEST(TypeTest, TensorAddMixedComplexRealPromotes) {
  auto a = cytnx::zeros({2}, cytnx::Type.ComplexFloat);
  auto b = cytnx::zeros({2}, cytnx::Type.Double);
  EXPECT_EQ((a + b).dtype(), cytnx::Type.ComplexDouble);
  EXPECT_EQ((b + a).dtype(), cytnx::Type.ComplexDouble);
}
