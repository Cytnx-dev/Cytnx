// CPU cost of materializing a permuted Tensor (Tensor::contiguous) across sizes. With USE_HPTT the
// copy goes through HPTT, whose thread count is the subject of #939; compare thread settings by
// running the binary under different environments (OMP_NUM_THREADS, taskset, OMP_THREAD_LIMIT).
#include <cstdint>
#include <initializer_list>

#include <benchmark/benchmark.h>
#include <cytnx.hpp>

namespace tensor_transpose_bm {

  // Reverses the axes of an n x n x n Double tensor and makes the result contiguous. Sizes run from
  // just above the 64-element cutoff below which Cytnx skips HPTT, to about 5.7e7 elements.
  void bm_contiguous_rank3_reverse(benchmark::State& state) {
    const auto n = static_cast<std::uint64_t>(state.range(0));
    const cytnx::Tensor source =
      cytnx::random::uniform({n, n, n}, -1.0, 1.0, cytnx::Device.cpu, /*seed=*/0);
    const cytnx::Tensor permuted = source.permute({2, 1, 0});
    for ([[maybe_unused]] auto iteration : state) {
      cytnx::Tensor result = permuted.contiguous();
      benchmark::DoNotOptimize(result);
    }
    const auto bytes_per_copy = static_cast<std::int64_t>(2 * n * n * n * sizeof(double));
    state.SetBytesProcessed(state.iterations() * bytes_per_copy);
  }
  // Real time, not CPU time: the copy may run on several threads.
  BENCHMARK(bm_contiguous_rank3_reverse)
    ->Apply([](benchmark::internal::Benchmark* bench) {
      for (const int n : {5, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384}) bench->Arg(n);
    })
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

}  // namespace tensor_transpose_bm
