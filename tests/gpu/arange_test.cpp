#include "gtest/gtest.h"

#include "gpu_test_tools.h"
#include "cytnx.hpp"

// Regression for the GPU arange step bug: cuSetArange_kernel dropped the parentheses
// around the linear index, so `start + step * blockIdx.x * blockDim.x + threadIdx.x`
// added the per-thread offset without scaling it by step. For arrays that fit in one
// block this collapsed to `start + threadIdx.x` (step effectively 1); for larger
// arrays the per-thread offset was still unscaled. Only real dtypes were affected;
// the complex overloads already parenthesized the index.
namespace ArangeTest {

  using namespace cytnx;

  // Independent literal check: arange(10, 40, 10) is [10, 20, 30] on the GPU.
  TEST(GpuArangeTest, nonunit_step_literal) {
    Tensor g = arange(10, 40, 10, Type.Int64, Device.cuda).to(Device.cpu);
    ASSERT_EQ(g.shape()[0], static_cast<cytnx_uint64>(3));
    EXPECT_EQ(g.at<cytnx_int64>({0}), 10);
    EXPECT_EQ(g.at<cytnx_int64>({1}), 20);
    EXPECT_EQ(g.at<cytnx_int64>({2}), 30);
  }

  // Real dtypes, single-block (n=3) and multi-block (n=1000 > blockDim) sizes, checked
  // against the (correct, independent) CPU arange implementation.
  TEST(GpuArangeTest, nonunit_step_real_dtypes) {
    for (auto dt : {Type.Int64, Type.Int32, Type.Double, Type.Float}) {
      for (cytnx_uint64 n : {static_cast<cytnx_uint64>(3), static_cast<cytnx_uint64>(1000)}) {
        SCOPED_TRACE("dtype=" + std::to_string(dt) + " n=" + std::to_string(n));
        const double end = 2.0 * static_cast<double>(n);
        Tensor cpu = arange(0, end, 2, dt);
        Tensor gpu = arange(0, end, 2, dt, Device.cuda).to(Device.cpu);
        ASSERT_EQ(gpu.shape()[0], n);
        EXPECT_TRUE(TestTools::AreNearlyEqTensor(gpu, cpu, 1e-6));
      }
    }
  }

}  // namespace ArangeTest
