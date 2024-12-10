#include <type_traits>

#include "cuda_runtime_api.h"
#include "gtest/gtest.h"

#include "../test_tools.h"
#include "cytnx_error.hpp"
#include "linalg.hpp"
#include "Tensor.hpp"
#include "Type.hpp"

namespace cytnx {
  template <typename T>
  class LinalgSumHomogeneousValuesTest : public ::testing::Test {
   public:
    static T value;
  };

  // Google Test (gtest) does not support combining type-parameterized and value-parameterized
  // tests. To work around this limitation, we use a static class member. For more details, refer
  // to: https://stackoverflow.com/a/22272614
  template <>
  cytnx_complex128 LinalgSumHomogeneousValuesTest<cytnx_complex128>::value{2.1, 3.2};
  template <>
  cytnx_complex64 LinalgSumHomogeneousValuesTest<cytnx_complex64>::value{-2., 3.};
  template <>
  cytnx_double LinalgSumHomogeneousValuesTest<cytnx_double>::value{
    7.3315594031297398e+252};  // 0_111 0100 0111_0000 0000 0000 0000 0000
                               // 0000  0000 0000 0000 0000 0000 0101 1100
  template <>
  cytnx_float LinalgSumHomogeneousValuesTest<cytnx_float>::value{
    4.0565264e+31};  // 0_111 0100 0_000 0000 0000 0000 0101 1100
  template <>
  cytnx_int64 LinalgSumHomogeneousValuesTest<cytnx_int64>::value{-10};
  template <>
  cytnx_uint64 LinalgSumHomogeneousValuesTest<cytnx_uint64>::value{10};
  template <>
  cytnx_int32 LinalgSumHomogeneousValuesTest<cytnx_int32>::value{-3};
  template <>
  cytnx_uint32 LinalgSumHomogeneousValuesTest<cytnx_uint32>::value{3};
  template <>
  cytnx_int16 LinalgSumHomogeneousValuesTest<cytnx_int16>::value{-3};
  template <>
  cytnx_uint16 LinalgSumHomogeneousValuesTest<cytnx_uint16>::value{4};

  using SupportedTypes =
    ::testing::Types<cytnx_complex128, cytnx_complex64, cytnx_double, cytnx_float, cytnx_int64,
                     cytnx_uint64, cytnx_int32, cytnx_uint32, cytnx_int16, cytnx_uint16>;
  TYPED_TEST_SUITE(LinalgSumHomogeneousValuesTest, SupportedTypes);

  /**
   * Validates the correctness of the summation result for each supported data type.
   * Note: `cytnx_bool` is not supported for the `linalg::Sum()` function.
   * This test also assesses the accuracy of summing floating-point numbers.
   */
  TYPED_TEST(LinalgSumHomogeneousValuesTest, GpuAccuracy) {
    TypeParam value = LinalgSumHomogeneousValuesTest<TypeParam>::value;
    int element_number = 10000;
    unsigned int dtype = Type_class().cy_typeid(value);

    Tensor tensor(/* shape */ {element_number}, dtype, Device.cuda, /* init_zero */ false);
    checkCudaErrors(cudaSetDevice(tensor.device()));
    tensor.fill(value);
    Tensor sum_result = linalg::Sum(tensor);
    checkCudaErrors(cudaDeviceSynchronize());

    EXPECT_EQ(sum_result.shape().size(), 1);
    EXPECT_EQ(sum_result.shape()[0], 1);

    EXPECT_NUMBER_EQ(sum_result.at<TypeParam>({0}), value * TypeParam{element_number});
  }
}  // namespace cytnx
