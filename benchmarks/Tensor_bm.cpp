#include <benchmark/benchmark.h>
#include <cytnx.hpp>

// Cytnx test
namespace BMTest_Tensor {
  static void BM_Cytnx_declare(benchmark::State& state) {
    for (auto _ : state) {
      cytnx::Tensor A;
    }
  }
  BENCHMARK(BM_Cytnx_declare);

  static void BM_ones(benchmark::State& state) {
    int num_1 = state.range(0);
    int num_2 = state.range(1);
    for (auto _ : state) {
      auto A = cytnx::UniTensor(cytnx::ones({num_1, num_2, 3}));
    }
  }
  BENCHMARK(BM_ones)->Args({5, 3})->Args({10, 9});

  static void BM_Tensor_contiguous(benchmark::State& state) {
    int D = state.range(0);
    auto A = cytnx::UniTensor(cytnx::random::uniform({D, D}, -1.0, 1.0, cytnx::Device.cpu, 0));
    A.permute_(std::vector<long int>({1, 0}), 1);
    for (auto _ : state) {
      auto tmp = A.clone();
      tmp.contiguous_();
    }
  }
  BENCHMARK(BM_Tensor_contiguous)
    ->Args({100})
    ->Args({1000})
    ->Args({10000})
    ->Unit(benchmark::kMillisecond);

}  // namespace BMTest_Tensor
