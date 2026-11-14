#include "gtest/gtest.h"

#include "Device.hpp"
#include "Type.hpp"
#include "../../linalg_test/integer_matmul_cases.h"

namespace cytnx {
  namespace gpu_test {
    namespace {

      template <typename T>
      class GpuIntegerMatmul : public ::testing::Test {};

      using IntegerTypes = ::testing::Types<cytnx_int64, cytnx_uint64, cytnx_int32, cytnx_uint32,
                                            cytnx_int16, cytnx_uint16>;
      TYPED_TEST_SUITE(GpuIntegerMatmul, IntegerTypes);

      // #1246: the GPU kernel stored every row after the first at the wrong offset unless the
      // contracted dimension equalled the number of output columns, and the Int32 and Uint32
      // wrappers passed the 4-byte buffers as 8-byte elements.
      TYPED_TEST(GpuIntegerMatmul, HandComputedProducts) {
        integer_matmul_test::check_integer_dtype<TypeParam>(Device.cuda);
      }

      TEST(GpuBoolMatmul, HandComputedProducts) {
        integer_matmul_test::check_bool_dtype(Device.cuda);
      }

    }  // namespace
  }  // namespace gpu_test
}  // namespace cytnx
