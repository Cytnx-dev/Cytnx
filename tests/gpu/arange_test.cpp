#include <gtest/gtest.h>

#include "Device.hpp"
#include "Generator.hpp"
#include "Tensor.hpp"
#include "Type.hpp"

// GPU arange: independent (hand-computed) expected values, NOT compared against the CPU
// arange -- the CPU front-end has had its own bugs (#1076), so it is not a trusted oracle.
// Covers the #1070 fix (per-thread offset must be scaled by step) and the >512-element
// multi-block linear index. Values are read back by copying to the host.

using cytnx::Device;
using cytnx::Type;

namespace {

  // arange(10, 40, 10) on the GPU must be [10, 20, 30] -- the pre-#1070 kernel dropped the
  // parentheses and produced [10, 11, 12] for a single block.
  template <typename T>
  void ExpectNonUnitStep(unsigned int dtype) {
    auto host = cytnx::arange(10, 40, 10, dtype, Device.cuda).to(Device.cpu);
    ASSERT_EQ(host.shape()[0], 3u);
    EXPECT_EQ(host.storage().template at<T>(0), static_cast<T>(10));
    EXPECT_EQ(host.storage().template at<T>(1), static_cast<T>(20));
    EXPECT_EQ(host.storage().template at<T>(2), static_cast<T>(30));
  }

  TEST(GpuArange, NonUnitStepRealDtypes) {
    ExpectNonUnitStep<cytnx::cytnx_int64>(Type.Int64);
    ExpectNonUnitStep<cytnx::cytnx_int32>(Type.Int32);
    ExpectNonUnitStep<cytnx::cytnx_double>(Type.Double);
    ExpectNonUnitStep<cytnx::cytnx_float>(Type.Float);
  }

  TEST(GpuArange, MultiBlockStepScaling) {
    // 1000 elements span more than one 512-thread block, so the block-boundary indices
    // (511 -> 512) exercise the full linear index. value[i] == 3*i (hand-computed).
    const cytnx::cytnx_uint64 n = 1000;
    auto host = cytnx::arange(0, 3000, 3, Type.Int64, Device.cuda).to(Device.cpu);
    ASSERT_EQ(host.shape()[0], n);
    EXPECT_EQ(host.storage().at<cytnx::cytnx_int64>(0), 0);
    EXPECT_EQ(host.storage().at<cytnx::cytnx_int64>(511), 1533);  // last of block 0
    EXPECT_EQ(host.storage().at<cytnx::cytnx_int64>(512), 1536);  // first of block 1
    EXPECT_EQ(host.storage().at<cytnx::cytnx_int64>(999), 2997);
  }

  TEST(GpuArange, FractionalStepDouble) {
    auto host = cytnx::arange(0, 1, 0.25, Type.Double, Device.cuda).to(Device.cpu);
    ASSERT_EQ(host.shape()[0], 4u);
    EXPECT_DOUBLE_EQ(host.storage().at<cytnx::cytnx_double>(0), 0.0);
    EXPECT_DOUBLE_EQ(host.storage().at<cytnx::cytnx_double>(1), 0.25);
    EXPECT_DOUBLE_EQ(host.storage().at<cytnx::cytnx_double>(2), 0.5);
    EXPECT_DOUBLE_EQ(host.storage().at<cytnx::cytnx_double>(3), 0.75);
  }

  TEST(GpuArange, EmptyRangeZeroExtent) {
    auto g = cytnx::arange(5, 5, 1, Type.Int64, Device.cuda);
    ASSERT_EQ(g.shape().size(), 1u);
    EXPECT_EQ(g.shape()[0], 0u);
    EXPECT_EQ(g.device(), Device.cuda);
  }

}  // namespace
