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
namespace storage_access_bm {

  using cytnx::Storage;
  using cytnx::Tensor;

  // Reads every element of a CPU tensor through Tensor::at.
  static void bm_cpu_tensor_at(benchmark::State& state) {
    const auto n = static_cast<std::uint64_t>(state.range(0));
    Tensor t = cytnx::arange(0, static_cast<double>(n), 1, cytnx::Type.Double, cytnx::Device.cpu);
    for ([[maybe_unused]] auto iteration : state) {
      double sum = 0.0;
      for (std::uint64_t i = 0; i < n; ++i) sum += t.at<double>({i});
      benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * n));
  }
  BENCHMARK(bm_cpu_tensor_at)->Arg(1 << 16)->Unit(benchmark::kMicrosecond);

  // Reads every element of a CPU storage through Storage::at.
  static void bm_cpu_storage_at(benchmark::State& state) {
    const auto n = static_cast<std::uint64_t>(state.range(0));
    Tensor t = cytnx::arange(0, static_cast<double>(n), 1, cytnx::Type.Double, cytnx::Device.cpu);
    Storage storage = t.storage();
    for ([[maybe_unused]] auto iteration : state) {
      double sum = 0.0;
      for (std::uint64_t i = 0; i < n; ++i) sum += storage.at<double>(i);
      benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * n));
  }
  BENCHMARK(bm_cpu_storage_at)->Arg(1 << 16)->Unit(benchmark::kMicrosecond);

#ifdef UNI_GPU
  // A sequence of in-place GPU operations with a host scalar, followed by one synchronization.
  // With a synchronization inside every operation, the host waits for each kernel before launching
  // the next one; without it, the kernels are queued back to back.
  static void bm_gpu_inplace_div_host_scalar(benchmark::State& state) {
    const auto n = static_cast<std::uint64_t>(state.range(0));
    constexpr std::int64_t ops_per_iteration = 64;
    Tensor x = cytnx::ones({n}, cytnx::Type.Double, cytnx::Device.cuda);
    cudaDeviceSynchronize();
    for ([[maybe_unused]] auto iteration : state) {
      for (std::int64_t k = 0; k < ops_per_iteration; ++k) x /= 1.0;
      cudaDeviceSynchronize();
    }
    state.SetItemsProcessed(state.iterations() * ops_per_iteration);
  }
  BENCHMARK(bm_gpu_inplace_div_host_scalar)
    ->Arg(2048)
    ->Arg(1 << 20)
    ->Unit(benchmark::kMicrosecond);
#endif  // UNI_GPU

}  // namespace storage_access_bm
