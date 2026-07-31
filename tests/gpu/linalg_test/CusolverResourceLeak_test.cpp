#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>

#include "cytnx.hpp"

using namespace cytnx;

namespace {

  // Identity with the last diagonal entry left at zero: singular, so getrf reports info > 0 and
  // InvM raises. Built on the CPU and moved, per the usual rule for GPU test inputs.
  Tensor MakeSingular(cytnx_uint64 n) {
    Tensor a = zeros({n, n}, Type.Double);
    for (cytnx_uint64 i = 0; i + 1 < n; ++i) a.at({i, i}) = 1.0;
    return a.to(Device.cuda);
  }

  double FreeDeviceMiB() {
    std::size_t free_bytes = 0, total_bytes = 0;
    EXPECT_EQ(cudaMemGetInfo(&free_bytes, &total_bytes), cudaSuccess);
    return static_cast<double>(free_bytes) / (1024.0 * 1024.0);
  }

  // Runs `body` and reports the device memory it failed to give back, in MiB.
  //
  // `cudaMemGetInfo` reports free memory for the whole *device*, not for this process, so any
  // other CUDA process on the same GPU moves the number under our feet. That is not hypothetical
  // here: the self-hosted machine that runs this suite also serves GPU CI, and a concurrent job
  // made the success-path test below report 68 MiB against its 64 MiB threshold in 1 run out of 5.
  //
  // A genuine leak reproduces on every attempt; contention does not. So measure the loop up to
  // `attempts` times and keep the smallest delta, stopping early as soon as one attempt comes in
  // under `slack`. The bias is deliberate and one-sided: if a neighbouring process *frees* during
  // our window the delta is understated, so this errs toward a false pass rather than a false
  // failure. For a guard, an intermittent red is worse than a missed regression -- and the leaks
  // these tests exist to catch are orders of magnitude above the threshold, not marginal.
  double MeasureLeakMiB(const std::function<void()> &body, int attempts, double slack) {
    double best = std::numeric_limits<double>::max();
    for (int i = 0; i < attempts && best >= slack; ++i) {
      const double before = FreeDeviceMiB();
      body();
      best = std::min(best, before - FreeDeviceMiB());
    }
    return best;
  }

}  // namespace

// #1146: the GPU cuSOLVER wrappers used to raise their LAPACK-`info` error *before* the cleanup
// block at the tail of the function. `cytnx_error_msg` is [[noreturn]], so unwinding skipped every
// cudaFree and the cuSOLVER handle -- and a leaked handle alone is ~12 MB of device memory.
//
// Measured on the pre-fix code, this exact loop at 200 calls leaked 2,465,792 KB (~2.4 GB, ~12.6 MB
// per call) and drove device free memory from 15393 MB to 12985 MB. After the fix the same loop
// leaks nothing. The threshold below therefore sits orders of magnitude below the regression while
// leaving generous slack for allocator caching.
//
// Reachable from user code: cytnx::error surfaces in Python as CytnxError, which is catchable, so
// retrying a failing inverse in a loop is the natural thing to write.
TEST(CusolverResourceLeak, GpuInvMErrorPathDoesNotLeakDeviceMemory) {
  constexpr int kCalls = 100;
  constexpr double kSlackMiB = 64.0;
  constexpr int kAttempts = 5;

  // Warm up so one-time CUDA context and cuSOLVER library allocation is outside the measurement.
  {
    Tensor warm = MakeSingular(4);
    EXPECT_THROW(linalg::InvM(warm), cytnx::error);
  }

  const double leaked = MeasureLeakMiB(
    [] {
      for (int i = 0; i < kCalls; ++i) {
        Tensor a = MakeSingular(4);
        EXPECT_THROW(linalg::InvM(a), cytnx::error);
      }
    },
    kAttempts, kSlackMiB);

  EXPECT_LT(leaked, kSlackMiB) << "leaked " << leaked << " MiB of device memory over " << kCalls
                               << " failing InvM calls (best of " << kAttempts << " attempts)";
}

// Companion guard for the same change: converting five files to scoped ownership must not break
// release on the *success* path either (a missing free, or a double free via a stale alias). This
// direction passed before the fix, so it is not a regression test -- it guards the refactor.
//
// Note on what this can still detect once #1144 lands: caching the cuSOLVER handle per device
// removes ~12 MB per call from the set of leakable resources, leaving the small work buffers
// (~82 KiB per call) as the only signal. At 20 rounds that is well under the threshold, so this
// guard catches a gross regression -- a whole workspace never freed -- and not a single small
// buffer. Raising the threshold would not help; the limit is the measurement, not the number.
TEST(CusolverResourceLeak, GpuCusolverSuccessPathsDoNotLeakDeviceMemory) {
  constexpr int kCalls = 20;
  constexpr double kSlackMiB = 64.0;
  constexpr int kAttempts = 5;

  Tensor a = zeros({8, 8}, Type.Double);
  for (cytnx_uint64 i = 0; i < 8; ++i) a.at({i, i}) = static_cast<double>(i + 1);
  a = a.to(Device.cuda);

  // Warm up every library path before measuring.
  {
    auto svd = linalg::Svd(a);
    auto gesvd = linalg::Gesvd(a);
    auto eigh = linalg::Eigh(a);
    auto det = linalg::Det(a);
    auto inv = linalg::InvM(a);
  }

  const double leaked = MeasureLeakMiB(
    [&a] {
      for (int i = 0; i < kCalls; ++i) {
        auto svd = linalg::Svd(a);
        auto gesvd = linalg::Gesvd(a);
        auto eigh = linalg::Eigh(a);
        auto det = linalg::Det(a);
        auto inv = linalg::InvM(a);
      }
    },
    kAttempts, kSlackMiB);

  EXPECT_LT(leaked, kSlackMiB) << "leaked " << leaked << " MiB of device memory over " << kCalls
                               << " successful Svd/Gesvd/Eigh/Det/InvM rounds (best of "
                               << kAttempts << " attempts)";
}

// The inverse of a diagonal matrix is the reciprocal diagonal -- an independently known value, not
// another Cytnx path. Confirms the converted cuInvM still computes correctly, since the conversion
// rewired which pointers the cuSOLVER calls receive.
TEST(CusolverResourceLeak, GpuInvMStillCorrectAfterScopedResources) {
  Tensor a = zeros({4, 4}, Type.Double);
  const double diag[4] = {2.0, -4.0, 0.5, 8.0};
  for (cytnx_uint64 i = 0; i < 4; ++i) a.at({i, i}) = diag[i];

  Tensor inv = linalg::InvM(a.to(Device.cuda)).to(Device.cpu);

  for (cytnx_uint64 i = 0; i < 4; ++i) {
    for (cytnx_uint64 j = 0; j < 4; ++j) {
      const double expected = (i == j) ? 1.0 / diag[i] : 0.0;
      EXPECT_NEAR(inv.at<cytnx_double>({i, j}), expected, 1e-12)
        << "element (" << i << ", " << j << ")";
    }
  }
}
