#include <string>
#include <variant>

#include "gtest/gtest.h"

#include "Type.hpp"

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

}  // namespace

TEST(TypeTest, VariantContainsRecognizesSupportedTypes) {
  EXPECT_TRUE((variant_contains_v<cytnx_double, Type_list>));
  EXPECT_TRUE((variant_contains_v<cytnx_bool, Type_list>));
  EXPECT_FALSE((variant_contains_v<std::string, Type_list>));
}
