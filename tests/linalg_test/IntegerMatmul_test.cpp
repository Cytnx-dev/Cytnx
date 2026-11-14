#include "gtest/gtest.h"

#include "Device.hpp"
#include "Type.hpp"
#include "integer_matmul_cases.h"

namespace cytnx {
  namespace test {
    namespace {

      template <typename T>
      class IntegerMatmul : public ::testing::Test {};

      using IntegerTypes = ::testing::Types<cytnx_int64, cytnx_uint64, cytnx_int32, cytnx_uint32,
                                            cytnx_int16, cytnx_uint16>;
      TYPED_TEST_SUITE(IntegerMatmul, IntegerTypes);

      // #1246: the Int32 and Uint32 kernels read and wrote the 4-byte buffers as 8-byte elements.
      TYPED_TEST(IntegerMatmul, HandComputedProducts) {
        integer_matmul_test::check_integer_dtype<TypeParam>(Device.cpu);
      }

      TEST(BoolMatmul, HandComputedProducts) { integer_matmul_test::check_bool_dtype(Device.cpu); }

    }  // namespace
  }  // namespace test
}  // namespace cytnx
