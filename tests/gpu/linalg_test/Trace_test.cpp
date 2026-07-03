#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "gpu_test_tools.h"
#include "Device.hpp"
#include "Tensor.hpp"
#include "Type.hpp"
#include "backend/Storage.hpp"
#include "linalg.hpp"
#include "random.hpp"

// GPU counterpart of tests/linalg_test/Trace_test.cpp. Each case builds the
// input on the GPU, traces it with the CUDA kernels, and compares against the
// trace of a contiguous CPU clone of the same data -- so the kernels' stride
// handling, per-dtype accumulation, and the 2D-vs-ND branch are all exercised
// against an independent reference. The GPU result is moved to the CPU before
// the comparison since AreNearlyEqTensor compares host-side.

namespace {

  using cytnx::cytnx_double;
  using cytnx::cytnx_int64;
  using cytnx::cytnx_uint64;
  using cytnx::Device;
  using cytnx::Storage;
  using cytnx::Tensor;
  using cytnx::Type;

  // Reference: trace a contiguous CPU clone of the (possibly permuted) GPU
  // tensor through the same public API, then keep it on the CPU.
  static Tensor ContiguousCpuReferenceTrace(const Tensor& gpu_t, cytnx_uint64 a, cytnx_uint64 b) {
    return cytnx::linalg::Trace(gpu_t.to(Device.cpu).contiguous(), a, b);
  }

  // Tensor's own constructor and Tensor::get() (slicing) both reject a 0 in any
  // shape dimension, so a zero-extent Tensor can only be built by composing a
  // zero-element Storage directly and reshaping it onto the target shape.
  static Tensor ZeroExtentGpuTensor(const std::vector<cytnx_int64>& shape, unsigned int dtype) {
    Storage s(0, dtype, Device.cuda);
    Tensor t = Tensor::from_storage(s);
    t.reshape_(shape);
    return t;
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

  TEST(LinalgGpuTraceTest, BlockSizeBoundaries) {
    // The launch sizes each block to its diagonal:
    //   threads_per_block = min(round_up_to_warp(diagonal_length), 256).
    // Walk diagonal lengths that straddle every sizing boundary -- sub-warp
    // (1, 31 -> a 32-thread block), exactly one warp (32), one past a warp
    // boundary (33 -> 64 threads), the 256-thread cap (255, 256), one past the
    // cap (257 -> capped block, threads stride), and a multi-stride length
    // (1024 -> 4 strides per thread). Each is a rank-2 trace, so it is a single
    // block whose result must match the CPU pairwise reference exactly within
    // tolerance; an off-by-one in the warp rounding, the per-warp shared-memory
    // hand-off, or the single-warp early-return path would show up here.
    for (cytnx_uint64 diagonal_length : {1u, 31u, 32u, 33u, 64u, 255u, 256u, 257u, 1024u}) {
      auto cpu = cytnx::random::random_tensor({diagonal_length, diagonal_length}, -2.0, 2.0,
                                              Device.cpu, 0, Type.Double);
      auto gpu = TraceOnGpuToCpu(cpu.to(Device.cuda), 0, 1);
      auto reference = cytnx::linalg::Trace(cpu, 0, 1);
      EXPECT_TRUE(cytnx::TestTools::AreNearlyEqTensor(gpu, reference, 1e-9))
        << "diagonal_length=" << diagonal_length;
    }
  }

  TEST(LinalgGpuTraceTest, ShortDiagonalsManyOutputs) {
    // The many-blocks regime: {n, middle, n} traced over (0, 2) launches
    // `middle` blocks, each sized to the short diagonal n. Covers a sub-warp
    // diagonal (n = 2 -> 32-thread blocks), an exact warp (n = 32), and one
    // past a warp boundary (n = 33 -> 64-thread blocks), each with enough
    // output elements that block indexing -- not just the reduction -- is
    // exercised.
    struct Case {
      cytnx_uint64 n, middle;
    };
    for (const auto& c : {Case{2, 1000}, Case{32, 500}, Case{33, 100}}) {
      auto cpu =
        cytnx::random::random_tensor({c.n, c.middle, c.n}, -2.0, 2.0, Device.cpu, 0, Type.Double);
      auto gpu = TraceOnGpuToCpu(cpu.to(Device.cuda), 0, 2);
      auto reference = cytnx::linalg::Trace(cpu, 0, 2);
      EXPECT_TRUE(cytnx::TestTools::AreNearlyEqTensor(gpu, reference, 1e-9))
        << "n=" << c.n << " middle=" << c.middle;
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

  TEST(LinalgGpuTraceTest, SurvivingRankAtFastPathCapMatchesReference) {
    // TraceImplGpu's stack-resident TraceLayout fast path (a __grid_constant__
    // kernel parameter, no cudaMalloc/cudaFree) handles surviving_rank <=
    // kMaxTraceRank (32); this pins correctness exactly at that boundary.
    // shape = {5, 5, 1x32}: 32 size-1 axes beyond the two traced ones keep the
    // tensor -- and its element count -- tiny even at rank 34.
    std::vector<cytnx_uint64> shape = {5, 5};
    shape.insert(shape.end(), 32, 1);
    auto cpu = cytnx::random::random_tensor(shape, -2.0, 2.0, Device.cpu, 0, Type.Double);
    auto gpu = TraceOnGpuToCpu(cpu.to(Device.cuda), 0, 1);
    auto reference = cytnx::linalg::Trace(cpu, 0, 1);
    EXPECT_TRUE(cytnx::TestTools::AreNearlyEqTensor(gpu, reference, 1e-10));
    EXPECT_EQ(gpu.shape().size(), 32u);
  }

  TEST(LinalgGpuTraceTest, SurvivingRankAboveFastPathCapUsesFallback) {
    // One more surviving axis than kMaxTraceRank forces TraceImplGpu's
    // heap-allocated cudaMallocAsync/cudaFreeAsync fallback path -- pins that
    // path's correctness independently of the fast path above.
    std::vector<cytnx_uint64> shape = {5, 5};
    shape.insert(shape.end(), 33, 1);
    auto cpu = cytnx::random::random_tensor(shape, -2.0, 2.0, Device.cpu, 0, Type.Double);
    auto gpu = TraceOnGpuToCpu(cpu.to(Device.cuda), 0, 1);
    auto reference = cytnx::linalg::Trace(cpu, 0, 1);
    EXPECT_TRUE(cytnx::TestTools::AreNearlyEqTensor(gpu, reference, 1e-10));
    EXPECT_EQ(gpu.shape().size(), 33u);
  }

  // The following three cases mirror tests/linalg_test/Trace_test.cpp's CPU
  // zero-extent coverage. cuTraceImpl's zero-extent early-out (the GPU
  // counterpart of TraceImpl's `diagonal_length == 0 || output_size == 0`
  // branch) returns before any cudaMalloc/kernel launch, so these pin that
  // path -- and its NumPy-matching output -- against regression independently
  // of the CPU implementation.

  TEST(LinalgGpuTraceTest, Rank2ZeroExtentReturnsZeroScalar) {
    auto t = ZeroExtentGpuTensor({0, 0}, Type.Double);
    auto out = cytnx::linalg::Trace(t, 0, 1).to(Device.cpu);
    ASSERT_EQ(out.shape().size(), 1u);
    EXPECT_EQ(out.shape()[0], 1u);
    EXPECT_DOUBLE_EQ(out.at<cytnx_double>({0}), 0.0);
  }

  TEST(LinalgGpuTraceTest, ZeroTracedAxisReturnsZeroFilledOutput) {
    // shape {3, 0, 0} traced over (1, 2): the diagonal is empty for every one of
    // the 3 surviving indices, so each sums to 0 -> output shape {3}, all zero.
    auto t = ZeroExtentGpuTensor({3, 0, 0}, Type.Double);
    auto out = cytnx::linalg::Trace(t, 1, 2).to(Device.cpu);
    ASSERT_EQ(out.shape().size(), 1u);
    EXPECT_EQ(out.shape()[0], 3u);
    for (cytnx_uint64 i = 0; i < 3; i++) EXPECT_DOUBLE_EQ(out.at<cytnx_double>({i}), 0.0);
  }

  TEST(LinalgGpuTraceTest, ZeroRemainingAxisReturnsEmptyOutput) {
    // shape {3, 0, 3} traced over (0, 2): the surviving axis (1) is zero-length,
    // so the output is an empty tensor of shape {0}.
    auto t = ZeroExtentGpuTensor({3, 0, 3}, Type.Double);
    auto out = cytnx::linalg::Trace(t, 0, 2).to(Device.cpu);
    ASSERT_EQ(out.shape().size(), 1u);
    EXPECT_EQ(out.shape()[0], 0u);
  }

}  // namespace
