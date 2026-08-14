#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>

#include "cytnx.hpp"

#ifdef UNI_GPU
  #include <cuda_runtime.h>
#endif

// Which is faster on the GPU for a non-contiguous operand: letting Mul consume it directly
// through the layout mappers (the path #1096 enables), or calling contiguous() on both operands
// first and multiplying contiguous buffers?
//
// The direct path avoids two full device-to-device copies but makes every element access a
// gather through the inverse mappers. The contiguous path pays for two materialised copies up
// front and then runs a fully coalesced kernel. Which wins is an empirical question -- these
// benchmarks answer it rather than reasoning about it.
//
// GPU kernel launches are asynchronous, so every timed region ends in cudaDeviceSynchronize().
// Without it the loop measures launch overhead, not execution. (Note the other GPU benchmarks in
// this directory do not synchronize, so their device numbers are not comparable to these.)
namespace BMTest_MulNonContig {

#ifdef UNI_GPU

  using cytnx::cytnx_double;
  using cytnx::Tensor;

  namespace {

    // A rank-3 cube permuted so that neither operand is contiguous, which is what forces the
    // gather path. permute() alone only relabels the strides; no data moves.
    Tensor MakeNonContiguousCube(cytnx::cytnx_uint64 n, unsigned int seed, unsigned int dtype) {
      const cytnx_double kLow = -10.0;
      const cytnx_double kHigh = 10.0;
      Tensor t =
        cytnx::random::random_tensor({n, n, n}, kLow, kHigh, cytnx::Device.cuda, seed, dtype);
      return t.permute({2, 0, 1});
    }

    void SyncDevice() { cudaDeviceSynchronize(); }

  }  // namespace

  // Direct: Mul consumes both non-contiguous operands (the path enabled by #1096).
  template <unsigned int dtype>
  static void BM_GpuMul_NonContig_Direct(benchmark::State& state) {
    const auto n = static_cast<cytnx::cytnx_uint64>(state.range(0));
    const Tensor a = MakeNonContiguousCube(n, 0, dtype);
    const Tensor b = MakeNonContiguousCube(n, 1, dtype);
    SyncDevice();
    for (auto _ : state) {
      auto result = cytnx::linalg::Mul(a, b);
      SyncDevice();
      benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n * n * n);
  }

  // Contiguous-first: materialise both operands, then multiply contiguous buffers. This is what
  // callers had to do before #1096, and what Ian expects to be faster.
  template <unsigned int dtype>
  static void BM_GpuMul_NonContig_ContiguousFirst(benchmark::State& state) {
    const auto n = static_cast<cytnx::cytnx_uint64>(state.range(0));
    const Tensor a = MakeNonContiguousCube(n, 0, dtype);
    const Tensor b = MakeNonContiguousCube(n, 1, dtype);
    SyncDevice();
    for (auto _ : state) {
      auto result = cytnx::linalg::Mul(a.contiguous(), b.contiguous());
      SyncDevice();
      benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n * n * n);
  }

  // Control: both operands already contiguous. Neither a gather nor a copy -- the floor that
  // says how much of each number above is the multiply itself.
  template <unsigned int dtype>
  static void BM_GpuMul_Contig_Baseline(benchmark::State& state) {
    const auto n = static_cast<cytnx::cytnx_uint64>(state.range(0));
    const cytnx_double kLow = -10.0;
    const cytnx_double kHigh = 10.0;
    const Tensor a =
      cytnx::random::random_tensor({n, n, n}, kLow, kHigh, cytnx::Device.cuda, 0, dtype);
    const Tensor b =
      cytnx::random::random_tensor({n, n, n}, kLow, kHigh, cytnx::Device.cuda, 1, dtype);
    SyncDevice();
    for (auto _ : state) {
      auto result = cytnx::linalg::Mul(a, b);
      SyncDevice();
      benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n * n * n);
  }

  // Isolates the copy half of the contiguous-first path, so a difference can be attributed to
  // the copies rather than to the multiply.
  template <unsigned int dtype>
  static void BM_GpuContiguous_Only(benchmark::State& state) {
    const auto n = static_cast<cytnx::cytnx_uint64>(state.range(0));
    const Tensor a = MakeNonContiguousCube(n, 0, dtype);
    SyncDevice();
    for (auto _ : state) {
      auto result = a.contiguous();
      SyncDevice();
      benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n * n * n);
  }

  // 64^3 = 262k elements up to 384^3 = 56.6M, so the comparison covers both the
  // launch-overhead-dominated end and the bandwidth-dominated end.
  BENCHMARK(BM_GpuMul_NonContig_Direct<cytnx::Type.Double>)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(384)
    ->Unit(benchmark::kMicrosecond);
  BENCHMARK(BM_GpuMul_NonContig_ContiguousFirst<cytnx::Type.Double>)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(384)
    ->Unit(benchmark::kMicrosecond);
  BENCHMARK(BM_GpuMul_Contig_Baseline<cytnx::Type.Double>)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(384)
    ->Unit(benchmark::kMicrosecond);
  BENCHMARK(BM_GpuContiguous_Only<cytnx::Type.Double>)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(384)
    ->Unit(benchmark::kMicrosecond);

#endif  // UNI_GPU

}  // namespace BMTest_MulNonContig
