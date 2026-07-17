#include <gtest/gtest.h>

#include <cmath>

#include "Generator.hpp"
#include "Tensor.hpp"
#include "Type.hpp"

// Independent (hand-computed) expected values for cytnx::arange(start, end, step).
// The contract is a half-open range [start, end): values start, start+step, ... strictly
// on the `start` side of `end`. See #1076. Values are read from contiguous storage.

using cytnx::Device;
using cytnx::Type;

namespace {

  TEST(Arange, HalfOpenIntegerStep) {
    // 40 is the exclusive endpoint, so it is NOT included.
    auto t = cytnx::arange(10, 40, 10, Type.Int64);
    ASSERT_EQ(t.shape().size(), 1u);
    ASSERT_EQ(t.shape()[0], 3u);
    EXPECT_EQ(t.dtype(), Type.Int64);
    EXPECT_EQ(t.storage().at<cytnx::cytnx_int64>(0), 10);
    EXPECT_EQ(t.storage().at<cytnx::cytnx_int64>(1), 20);
    EXPECT_EQ(t.storage().at<cytnx::cytnx_int64>(2), 30);
  }

  TEST(Arange, EndpointExcludedWhenExactlyOnGrid) {
    // end lands exactly on start + 3*step; half-open excludes it -> 3 elements, not 4.
    auto t = cytnx::arange(0, 3, 1, Type.Int64);
    ASSERT_EQ(t.shape()[0], 3u);
    EXPECT_EQ(t.storage().at<cytnx::cytnx_int64>(0), 0);
    EXPECT_EQ(t.storage().at<cytnx::cytnx_int64>(1), 1);
    EXPECT_EQ(t.storage().at<cytnx::cytnx_int64>(2), 2);
  }

  TEST(Arange, FractionalStep) {
    // [0, 0.25, 0.5, 0.75]; 1.0 is the exclusive endpoint.
    auto t = cytnx::arange(0, 1, 0.25, Type.Double);
    ASSERT_EQ(t.shape()[0], 4u);
    EXPECT_DOUBLE_EQ(t.storage().at<cytnx::cytnx_double>(0), 0.0);
    EXPECT_DOUBLE_EQ(t.storage().at<cytnx::cytnx_double>(1), 0.25);
    EXPECT_DOUBLE_EQ(t.storage().at<cytnx::cytnx_double>(2), 0.5);
    EXPECT_DOUBLE_EQ(t.storage().at<cytnx::cytnx_double>(3), 0.75);
  }

  TEST(Arange, NegativeStep) {
    // Descending: [5, 4, 3, 2, 1]; 0 is the exclusive endpoint.
    auto t = cytnx::arange(5, 0, -1, Type.Int64);
    ASSERT_EQ(t.shape()[0], 5u);
    EXPECT_EQ(t.storage().at<cytnx::cytnx_int64>(0), 5);
    EXPECT_EQ(t.storage().at<cytnx::cytnx_int64>(4), 1);
  }

  TEST(Arange, SmallScaleSingleElement) {
    // #1076: count = 1e-15 / 2e-15 = 0.5 -> ceil -> 1 element [0.0].
    // The old fixed-1e-14 fmod threshold computed 0 elements here and threw.
    auto t = cytnx::arange(0.0, 1e-15, 2e-15, Type.Double);
    ASSERT_EQ(t.shape()[0], 1u);
    EXPECT_DOUBLE_EQ(t.storage().at<cytnx::cytnx_double>(0), 0.0);
  }

  TEST(Arange, EmptyRangesAreZeroExtent) {
    // Empty / direction-mismatched ranges are a zero-extent tensor, not an error (#1076).
    for (auto t : {cytnx::arange(5, 5, 1, Type.Int64),  // start == end
                   cytnx::arange(10, 0, 1, Type.Int64),  // positive step, end < start
                   cytnx::arange(0, 10, -1, Type.Int64)}) {  // negative step, end > start
      ASSERT_EQ(t.shape().size(), 1u);
      EXPECT_EQ(t.shape()[0], 0u);
      EXPECT_EQ(t.dtype(), Type.Int64);
    }
  }

  TEST(Arange, UlpBoundaryEndpoint) {
    // Push the endpoint one ULP past an integer grid point: end just ABOVE 3 now
    // includes 3 (count = 3.0000...4 -> ceil 4), so [0, 1, 2, 3].
    auto above = cytnx::arange(0.0, std::nextafter(3.0, INFINITY), 1.0, Type.Double);
    ASSERT_EQ(above.shape()[0], 4u);
    EXPECT_DOUBLE_EQ(above.storage().at<cytnx::cytnx_double>(3), 3.0);

    // end just BELOW 3 excludes 3 (count = 2.9999...6 -> ceil 3), so [0, 1, 2].
    auto below = cytnx::arange(0.0, std::nextafter(3.0, -INFINITY), 1.0, Type.Double);
    ASSERT_EQ(below.shape()[0], 3u);
    EXPECT_DOUBLE_EQ(below.storage().at<cytnx::cytnx_double>(2), 2.0);
  }

  TEST(Arange, StepZeroThrows) {
    EXPECT_THROW(cytnx::arange(0, 10, 0, Type.Double), std::logic_error);
  }

  TEST(Arange, NonFiniteThrows) {
    EXPECT_THROW(cytnx::arange(0.0, INFINITY, 1.0, Type.Double), std::logic_error);
    EXPECT_THROW(cytnx::arange(0.0, 10.0, NAN, Type.Double), std::logic_error);
  }

  TEST(Arange, CountOverloadHalfOpen) {
    // arange(N) == arange(0, N, 1): [0, 1, ..., N-1], default dtype Double.
    auto t = cytnx::arange(6);
    ASSERT_EQ(t.shape()[0], 6u);
    EXPECT_EQ(t.dtype(), Type.Double);
    EXPECT_DOUBLE_EQ(t.storage().at<cytnx::cytnx_double>(0), 0.0);
    EXPECT_DOUBLE_EQ(t.storage().at<cytnx::cytnx_double>(5), 5.0);
  }

  TEST(Arange, CountOverloadZeroIsEmptyNegativeThrows) {
    auto t = cytnx::arange(0);  // consistent with the empty-range case
    EXPECT_EQ(t.shape().size(), 1u);
    EXPECT_EQ(t.shape()[0], 0u);
    EXPECT_THROW(cytnx::arange(-1), std::logic_error);
  }

}  // namespace
