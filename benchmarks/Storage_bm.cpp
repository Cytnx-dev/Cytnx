#include <benchmark/benchmark.h>

#include <cstdint>

#include "cytnx.hpp"

#ifdef UNI_GPU
  #include <cuda_runtime.h>
#endif

// Typed element access to CPU storage, and GPU arithmetic with a host scalar.
//
// In a CUDA-enabled build, Storage_base::data<T>(), at<T>() and back<T>() called
// cudaDeviceSynchronize() whatever the storage's device (#1260). Every element access to a CPU
// tensor was then a driver call. Every GPU operation with a host scalar synchronized the device
// too, because the scalar is wrapped in a CPU rank-0 tensor (scalar_as_rank0_tensor in Tensor.cpp)
// before the kernel is launched. In a CPU-only build these benchmarks measure plain element access.
namespace BMTest_StorageAccess {

  using cytnx::cytnx_uint64;
  using cytnx::Storage;
  using cytnx::Tensor;

  // Reads every element of a CPU tensor through Tensor::at.
  static void BM_CpuTensorAt(benchmark::State& state) {
    const auto n = static_cast<cytnx_uint64>(state.range(0));
    Tensor t = cytnx::arange(0, static_cast<double>(n), 1, cytnx::Type.Double, cytnx::Device.cpu);
    for (auto _ : state) {
      double sum = 0.0;
      for (cytnx_uint64 i = 0; i < n; ++i) sum += t.at<double>({i});
      benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n);
  }
  BENCHMARK(BM_CpuTensorAt)->Arg(1 << 16)->Unit(benchmark::kMicrosecond);

  // Reads every element of a CPU storage through Storage::at.
  static void BM_CpuStorageAt(benchmark::State& state) {
    const auto n = static_cast<cytnx_uint64>(state.range(0));
    Tensor t = cytnx::arange(0, static_cast<double>(n), 1, cytnx::Type.Double, cytnx::Device.cpu);
    Storage storage = t.storage();
    for (auto _ : state) {
      double sum = 0.0;
      for (cytnx_uint64 i = 0; i < n; ++i) sum += storage.at<double>(i);
      benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n);
  }
  BENCHMARK(BM_CpuStorageAt)->Arg(1 << 16)->Unit(benchmark::kMicrosecond);

#ifdef UNI_GPU
  // A sequence of in-place GPU operations with a host scalar, followed by one synchronization.
  // With a synchronization inside every operation, the host waits for each kernel before launching
  // the next one; without it, the kernels are queued back to back.
  static void BM_GpuInplaceDivHostScalar(benchmark::State& state) {
    const auto n = static_cast<cytnx_uint64>(state.range(0));
    constexpr int kOpsPerIteration = 64;
    Tensor x = cytnx::ones({n}, cytnx::Type.Double, cytnx::Device.cuda);
    cudaDeviceSynchronize();
    for (auto _ : state) {
      for (int k = 0; k < kOpsPerIteration; ++k) x /= 1.0;
      cudaDeviceSynchronize();
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * kOpsPerIteration);
  }
  BENCHMARK(BM_GpuInplaceDivHostScalar)->Arg(2048)->Arg(1 << 20)->Unit(benchmark::kMicrosecond);
#endif  // UNI_GPU

}  // namespace BMTest_StorageAccess
