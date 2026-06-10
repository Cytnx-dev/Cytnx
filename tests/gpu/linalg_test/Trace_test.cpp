#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "gpu_test_tools.h"
#include "Device.hpp"
#include "Tensor.hpp"
#include "Type.hpp"
#include "linalg.hpp"
#include "random.hpp"

// GPU counterpart of tests/linalg_test/Trace_test.cpp. Each case builds the
// input on the GPU, traces it with the CUDA kernels, and compares against the
// trace of a contiguous CPU clone of the same data -- so the kernels' stride
// handling, per-dtype accumulation, and the 2D-vs-ND branch are all exercised
// against an independent reference. The GPU result is moved to the CPU before
// the comparison since AreNearlyEqTensor compares host-side.

namespace {

  using cytnx::cytnx_uint64;
  using cytnx::Device;
  using cytnx::Tensor;
  using cytnx::Type;

  // Reference: trace a contiguous CPU clone of the (possibly permuted) GPU
  // tensor through the same public API, then keep it on the CPU.
  static Tensor ContiguousCpuReferenceTrace(const Tensor& gpu_t, cytnx_uint64 a, cytnx_uint64 b) {
    return cytnx::linalg::Trace(gpu_t.to(Device.cpu).contiguous(), a, b);
  }

  static Tensor TraceOnGpuToCpu(const Tensor& gpu_t, cytnx_uint64 a, cytnx_uint64 b) {
    return cytnx::linalg::Trace(gpu_t, a, b).to(Device.cpu);
  }

  TEST(LinalgGpuTraceTest, PermutedRank3MatchesContiguous) {
    auto t = cytnx::random::random_tensor({4, 3, 4}, -1.0, 1.0, Device.cpu, 0, Type.Double)
               .to(Device.cuda);
    auto p = t.permute({2, 1, 0});  // shape {4, 3, 4} but non-contiguous
    ASSERT_FALSE(p.is_contiguous());
    auto gpu = TraceOnGpuToCpu(p, 0, 2);
    auto reference = ContiguousCpuReferenceTrace(p, 0, 2);
    EXPECT_TRUE(cytnx::TestTools::AreNearlyEqTensor(gpu, reference, 1e-12));
  }

  TEST(LinalgGpuTraceTest, PermutedTracesAcrossRanksAndDtypes) {
    struct Case {
      std::vector<cytnx_uint64> shape;
      std::vector<cytnx_uint64> perm;
      cytnx_uint64 ax1, ax2;
    };
    // Each case picks two equal-sized logical axes after a non-trivial permute.
    const std::vector<Case> cases = {
      // rank 3
      {{4, 3, 4}, {1, 0, 2}, 1, 2},
      // rank 4 with non-adjacent traced axes (permuted shape = {4, 4, 5, 3})
      {{4, 3, 4, 5}, {2, 0, 3, 1}, 0, 1},
      // rank 5
      {{3, 4, 3, 2, 3}, {4, 1, 0, 3, 2}, 0, 2},
    };
    for (unsigned int dtype : {Type.Double, Type.ComplexDouble, Type.Int32}) {
      for (const auto& c : cases) {
        auto t =
          cytnx::random::random_tensor(c.shape, -2.0, 2.0, Device.cpu, 0, dtype).to(Device.cuda);
        auto p = t.permute(c.perm);
        EXPECT_FALSE(p.is_contiguous());
        auto gpu = TraceOnGpuToCpu(p, c.ax1, c.ax2);
        auto reference = ContiguousCpuReferenceTrace(p, c.ax1, c.ax2);
        EXPECT_TRUE(cytnx::TestTools::AreNearlyEqTensor(gpu, reference, 1e-10))
          << "dtype=" << dtype << " rank=" << c.shape.size();
      }
    }
  }

  TEST(LinalgGpuTraceTest, Rank2Path) {
    // Exercises the rank-2 branch (out_rank == 0 / n_elem == 1): the single
    // block traces the whole matrix diagonal.
    auto t =
      cytnx::random::random_tensor({6, 6}, -1.0, 1.0, Device.cpu, 0, Type.Double).to(Device.cuda);
    auto p = t.permute({1, 0});
    ASSERT_FALSE(p.is_contiguous());
    auto out_p = TraceOnGpuToCpu(p, 0, 1);
    auto out_c = ContiguousCpuReferenceTrace(t, 0, 1);  // tr(A) == tr(A^T)
    ASSERT_EQ(out_p.shape().size(), 1u);
    ASSERT_EQ(out_p.shape()[0], 1u);
    EXPECT_TRUE(cytnx::TestTools::AreNearlyEqTensor(out_p, out_c, 1e-12));
  }

  TEST(LinalgGpuTraceTest, OutputRankIsInputMinusTwo) {
    // tr(rank-N) -> rank-(N-2); tr(rank-2) -> a 1-element rank-1 tensor.
    auto r4 = cytnx::random::random_tensor({2, 3, 2, 4}, -1.0, 1.0, Device.cpu, 0, Type.Double)
                .to(Device.cuda);
    auto out4 = TraceOnGpuToCpu(r4, 0, 2);
    EXPECT_EQ(out4.shape().size(), 2u);
    EXPECT_EQ(out4.shape()[0], 3u);
    EXPECT_EQ(out4.shape()[1], 4u);

    auto r3 = cytnx::random::random_tensor({3, 4, 3}, -1.0, 1.0, Device.cpu, 0, Type.Double)
                .to(Device.cuda);
    auto out3 = TraceOnGpuToCpu(r3, 0, 2);
    EXPECT_EQ(out3.shape().size(), 1u);
    EXPECT_EQ(out3.shape()[0], 4u);

    auto r2 =
      cytnx::random::random_tensor({5, 5}, -1.0, 1.0, Device.cpu, 0, Type.Double).to(Device.cuda);
    auto out2 = TraceOnGpuToCpu(r2, 0, 1);
    EXPECT_EQ(out2.shape().size(), 1u);
    EXPECT_EQ(out2.shape()[0], 1u);
  }

  TEST(LinalgGpuTraceTest, SwappedAxisOrderMatches) {
    // Trace(T, a, b) == Trace(T, b, a) (the kernels are symmetric in the axes).
    auto t = cytnx::random::random_tensor({3, 4, 3, 2}, -1.0, 1.0, Device.cpu, 0, Type.Double)
               .to(Device.cuda);
    auto ab = TraceOnGpuToCpu(t, 0, 2);
    auto ba = TraceOnGpuToCpu(t, 2, 0);
    EXPECT_TRUE(cytnx::TestTools::AreNearlyEqTensor(ab, ba, 1e-12));
  }

  TEST(LinalgGpuTraceTest, MatchesCpuTraceOnContiguousInput) {
    // The headline cross-device equivalence: the GPU kernel result equals the
    // CPU pairwise result on identical contiguous data, across dtypes.
    for (unsigned int dtype : {Type.Double, Type.ComplexDouble, Type.Int32}) {
      auto cpu = cytnx::random::random_tensor({8, 5, 8}, -2.0, 2.0, Device.cpu, 0, dtype);
      auto gpu = TraceOnGpuToCpu(cpu.to(Device.cuda), 0, 2);
      auto reference = cytnx::linalg::Trace(cpu, 0, 2);
      EXPECT_TRUE(cytnx::TestTools::AreNearlyEqTensor(gpu, reference, 1e-10)) << "dtype=" << dtype;
    }
  }

  TEST(LinalgGpuTraceTest, LargeDiagonalAccuracyBound) {
    // Upper bound the GPU's diagonal-sum accuracy at a large diagonal_length.
    // The current TraceDiagonalBlock accumulates serially per thread; a future
    // intra-block tree reduction would tighten the bound. The assertion is on
    // the *exact* sum (a fully homogeneous input, so the math is closed-form)
    // with a slack proportional to diagonal_length * eps * |value|. That bound
    // is generous enough to hold for the present serial accumulation, and is
    // also satisfied by any more accurate replacement -- so the test pins
    // correctness without freezing the current implementation's error regime.
    const cytnx::cytnx_int64 n = 4096;
    const cytnx::cytnx_double value = 0.5;
    auto t =
      cytnx::Tensor({static_cast<cytnx::cytnx_uint64>(n), static_cast<cytnx::cytnx_uint64>(n)},
                    Type.Double, Device.cuda, /*init_zero=*/false);
    t.fill(value);
    auto gpu = cytnx::linalg::Trace(t, 0, 1).to(Device.cpu);
    const double expected = value * static_cast<double>(n);
    const double slack = static_cast<double>(n) * 1e-15 * std::abs(value) + 1e-9;
    EXPECT_NEAR(gpu.at<cytnx::cytnx_double>({0}), expected, slack);
  }

}  // namespace
