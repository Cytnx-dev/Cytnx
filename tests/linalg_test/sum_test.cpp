#include "gtest/gtest.h"

#include "test_tools.h"
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
  TYPED_TEST(LinalgSumHomogeneousValuesTest, Accuracy) {
    TypeParam value = LinalgSumHomogeneousValuesTest<TypeParam>::value;
    int element_number = 10000;
    unsigned int dtype = Type_class().cy_typeid(value);

    Tensor tensor(/* shape */ {static_cast<unsigned long>(element_number)}, dtype, Device.cpu,
                  /* init_zero */ false);
    tensor.fill(value);
    Tensor sum_result = linalg::Sum(tensor);

    EXPECT_EQ(sum_result.shape().size(), 1);
    EXPECT_EQ(sum_result.shape()[0], 1);

    EXPECT_NUMBER_EQ(sum_result.at<TypeParam>({0}), value * static_cast<TypeParam>(element_number));
  }

  /**
   * Exercises every branch of `PairwiseSumBlocks` -- the recursive core that
   * `linalg::Sum` dispatches floating-point reductions through.
   *
   *   * n < 8         : straight serial loop
   *   * 8 <= n <= 128 : 8-accumulator unrolled body, optionally with a scalar
   *                     tail when n % 8 != 0
   *   * n > 128       : recursive split into two halves rounded to a multiple
   *                     of 8
   *
   * The original Accuracy test only covers the n = 10000 recursive-split case;
   * a regression in either of the small-n branches would not be caught there.
   * Sizes 7/8/9/15/128/129/137 straddle the thresholds (including the off-by-one
   * tail cases). The expected result is exact, so any branch that drops or
   * double-counts an element fails immediately.
   */
  TEST(LinalgSumBoundaryTest, EachPairwiseSumBranch) {
    const cytnx_double value = 1.0;
    for (int n : {1, 7, 8, 9, 15, 128, 129, 137, 1024}) {
      Tensor tensor(/* shape */ {static_cast<unsigned long>(n)}, Type.Double, Device.cpu,
                    /* init_zero */ false);
      tensor.fill(value);
      Tensor sum_result = linalg::Sum(tensor);
      EXPECT_EQ(sum_result.shape().size(), 1);
      EXPECT_EQ(sum_result.shape()[0], 1);
      EXPECT_DOUBLE_EQ(sum_result.at<cytnx_double>({0}), value * static_cast<cytnx_double>(n))
        << "n=" << n;
    }
  }

  /**
   * The dynamic-range case that motivates pairwise summation, and the one
   * input distribution where naive serial accumulation visibly fails.
   *
   * Both arrays are [+L, 1, 1, ..., 1, -L] with L far above the precision
   * threshold (2^53 for double, 2^23 for float); the exact sum is N - 2. Under
   * naive accumulation the running total reaches L on the first element and
   * the subsequent +1's vanish in IEEE 754 rounding -- 1.0 is below the unit
   * in the last place at that magnitude -- so naive returns ~0. Pairwise
   * keeps small values together in the tree until the cancellation of +/-L at
   * the top, so the small terms survive (modulo a handful of 1's in the
   * unrolled-accumulator blocks that hold L itself).
   *
   * The contract this asserts is qualitative on purpose: any reasonable
   * pairwise implementation must land much closer to N - 2 than to 0. A
   * serial-accumulation regression would collapse to ~0 and fail
   * `EXPECT_GT(result, N / 2)` immediately; the exact pairwise result depends
   * on which accumulators receive +/-L (a few small terms are lost there).
   */
  TEST(LinalgSumHeterogeneousMagnitudeTest, RecoversTermsLostByNaiveAccumulation_Double) {
    constexpr int N = 1024;
    Tensor tensor(/* shape */ {static_cast<unsigned long>(N)}, Type.Double, Device.cpu,
                  /* init_zero */ false);
    tensor.fill(static_cast<cytnx_double>(1));
    tensor.at<cytnx_double>({0}) = 1e16;
    tensor.at<cytnx_double>({N - 1}) = -1e16;
    Tensor sum_result = linalg::Sum(tensor);
    const cytnx_double result = sum_result.at<cytnx_double>({0});
    EXPECT_GT(result, static_cast<cytnx_double>(N / 2));
    EXPECT_LE(result, static_cast<cytnx_double>(N));
  }

  TEST(LinalgSumHeterogeneousMagnitudeTest, RecoversTermsLostByNaiveAccumulation_Float) {
    constexpr int N = 1024;
    Tensor tensor(/* shape */ {static_cast<unsigned long>(N)}, Type.Float, Device.cpu,
                  /* init_zero */ false);
    tensor.fill(static_cast<cytnx_float>(1));
    // float has ~7.2 decimal digits of precision; 1e8 already exceeds 2^23, so
    // a serial `1e8 + 1` collapses to 1e8 and the unit terms are lost.
    tensor.at<cytnx_float>({0}) = 1e8f;
    tensor.at<cytnx_float>({N - 1}) = -1e8f;
    Tensor sum_result = linalg::Sum(tensor);
    const cytnx_float result = sum_result.at<cytnx_float>({0});
    EXPECT_GT(result, static_cast<cytnx_float>(N / 2));
    EXPECT_LE(result, static_cast<cytnx_float>(N));
  }
}  // namespace cytnx
