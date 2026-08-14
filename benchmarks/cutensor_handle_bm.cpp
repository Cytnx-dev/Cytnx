#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>

#include "cytnx.hpp"

#ifdef UNI_GPU
  #include <cuda_runtime.h>
#endif

// Cytnx used to call cutensorCreate() and cutensorDestroy() around every single cuTENSOR
// operation. cutensorCreate costs ~3.4 ms flat -- it initialises the library context and its
// plan cache -- and that cost is independent of tensor size, so on small tensors it was the
// entire runtime (#1132).
//
// Two live paths pay it. With UNI_CUTENSOR defined, every float/complex GPU Movemem goes through
// cuMovemem_cutensor_gpu, so contiguous() and permute() pay it; and Tensordot pays it. (Integer
// dtypes take the general kernel instead and are unaffected -- hence the Double dtype here.)
//
// These benchmarks are deliberately weighted towards small tensors, where a flat per-call cost
// dominates and the regression is visible; at large n the real work swamps it and the two
// revisions should converge. That convergence is itself the check that the handle cache changes
// only overhead and not the work being done.
//
// GPU kernel launches are asynchronous, so every timed region ends in cudaDeviceSynchronize().
// Without it the loop measures launch overhead rather than execution.
namespace BMTest_CutensorHandle {

#ifdef UNI_GPU

  using cytnx::cytnx_double;
  using cytnx::cytnx_uint64;
  using cytnx::Tensor;

  namespace {

    void SyncDevice() { cudaDeviceSynchronize(); }

    // A rank-3 cube permuted so the result is non-contiguous. permute() only relabels strides;
    // no data moves until contiguous() is called, which is what routes into Movemem.
    Tensor MakeNonContiguousCube(cytnx_uint64 n, unsigned int seed) {
      const cytnx_double kLow = -10.0;
      const cytnx_double kHigh = 10.0;
      Tensor t = cytnx::random::random_tensor({n, n, n}, kLow, kHigh, cytnx::Device.cuda, seed,
                                              cytnx::Type.Double);
      return t.permute({2, 0, 1});
    }

  }  // namespace

  // contiguous() on a non-contiguous GPU tensor -> cuMovemem_cutensor_gpu.
  static void BM_GpuContiguous(benchmark::State& state) {
    const auto n = static_cast<cytnx_uint64>(state.range(0));
    const Tensor a = MakeNonContiguousCube(n, 0);
    SyncDevice();
    for (auto _ : state) {
      auto result = a.contiguous();
      SyncDevice();
      benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n * n * n);
  }
  BENCHMARK(BM_GpuContiguous)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond);

  // Tensordot over one contracted index -> cuTensordot_internal.
  static void BM_GpuTensordot(benchmark::State& state) {
    const auto n = static_cast<cytnx_uint64>(state.range(0));
    const cytnx_double kLow = -10.0;
    const cytnx_double kHigh = 10.0;
    const Tensor a =
      cytnx::random::random_tensor({n, n}, kLow, kHigh, cytnx::Device.cuda, 0, cytnx::Type.Double);
    const Tensor b =
      cytnx::random::random_tensor({n, n}, kLow, kHigh, cytnx::Device.cuda, 1, cytnx::Type.Double);
    SyncDevice();
    for (auto _ : state) {
      auto result = cytnx::linalg::Tensordot(a, b, {1}, {0});
      SyncDevice();
      benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n * n);
  }
  BENCHMARK(BM_GpuTensordot)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond);

#endif  // UNI_GPU

}  // namespace BMTest_CutensorHandle
