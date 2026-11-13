#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>

#include "cytnx.hpp"

#ifdef UNI_GPU
  #include <cuda_runtime.h>
#endif

// Cytnx wrapped every cuBLAS and cuSOLVER call in a create/destroy pair for the library handle
// (#1144). Steady-state cost measured in isolation on an RTX 4070 Ti SUPER:
//
//     cublasCreate     + cublasDestroy      ~330 us
//     cusolverDnCreate + cusolverDnDestroy  ~456 us
//
// Neither scales with problem size, so on small operands the handle dominates. Measured directly
// against cublasDgemm in the same process, handle setup cost 19.6x the multiply at n=32 and still
// 3.5x at n=256, crossing over only around n=400.
//
// These benchmarks span n=32..512 so the crossover is visible: the saving should be a roughly
// constant per-call amount, large relative to the work at small n and negligible at large n. A
// constant absolute delta is the evidence that the change removes overhead rather than work.
//
// GPU kernel launches are asynchronous, so every timed region ends in cudaDeviceSynchronize().
namespace BMTest_CublasHandle {

#ifdef UNI_GPU

  using cytnx::cytnx_double;
  using cytnx::cytnx_uint64;
  using cytnx::Tensor;

  namespace {

    void SyncDevice() { cudaDeviceSynchronize(); }

    Tensor MakeMatrix(cytnx_uint64 n, unsigned int seed) {
      const cytnx_double kLow = -1.0;
      const cytnx_double kHigh = 1.0;
      return cytnx::random::random_tensor({n, n}, kLow, kHigh, cytnx::Device.cuda, seed,
                                          cytnx::Type.Double);
    }

  }  // namespace

  // Matmul -> cuMatmul_internal, one cublasCreate/Destroy per call before this change.
  static void BM_GpuMatmul(benchmark::State& state) {
    const auto n = static_cast<cytnx_uint64>(state.range(0));
    const Tensor a = MakeMatrix(n, 0);
    const Tensor b = MakeMatrix(n, 1);
    SyncDevice();
    for (auto _ : state) {
      auto result = cytnx::linalg::Matmul(a, b);
      SyncDevice();
      benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n * n);
  }
  BENCHMARK(BM_GpuMatmul)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Unit(benchmark::kMicrosecond);

  // Vectordot -> cuVectordot_internal. A dot product is nearly free, so this is close to a pure
  // measurement of the per-call handle overhead.
  static void BM_GpuVectordot(benchmark::State& state) {
    const auto n = static_cast<cytnx_uint64>(state.range(0));
    const cytnx_double kLow = -1.0;
    const cytnx_double kHigh = 1.0;
    const Tensor v =
      cytnx::random::random_tensor({n}, kLow, kHigh, cytnx::Device.cuda, 0, cytnx::Type.Double);
    const Tensor w =
      cytnx::random::random_tensor({n}, kLow, kHigh, cytnx::Device.cuda, 1, cytnx::Type.Double);
    SyncDevice();
    for (auto _ : state) {
      auto result = cytnx::linalg::Vectordot(v, w);
      SyncDevice();
      benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n);
  }
  BENCHMARK(BM_GpuVectordot)->Arg(64)->Arg(1024)->Arg(65536)->Unit(benchmark::kMicrosecond);

  // Det -> cuDet_internal, exercising the cuSOLVER handle rather than the cuBLAS one.
  static void BM_GpuDet(benchmark::State& state) {
    const auto n = static_cast<cytnx_uint64>(state.range(0));
    const Tensor a = MakeMatrix(n, 0);
    SyncDevice();
    for (auto _ : state) {
      auto result = cytnx::linalg::Det(a);
      SyncDevice();
      benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n * n);
  }
  BENCHMARK(BM_GpuDet)->Arg(32)->Arg(128)->Arg(512)->Unit(benchmark::kMicrosecond);

#endif  // UNI_GPU

}  // namespace BMTest_CublasHandle
