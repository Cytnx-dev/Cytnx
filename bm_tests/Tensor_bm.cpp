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

  static void BM_Cytnx_contract(benchmark::State& state) {
    auto A = cytnx::UniTensor(cytnx::ones({3, 3, 3}));
    A.set_labels({"1", "2", "3"});
    auto B = cytnx::UniTensor(cytnx::ones({3, 3, 3, 3}));
    B.set_labels({"2", "3", "4", "5"});
    for (auto _ : state) {
      auto C = cytnx::Contract(A, B);
    }
  }
  BENCHMARK(BM_Cytnx_contract);

  // test with several arguments, ex, bond dimension
  static void BM_ones(benchmark::State& state) {
    int num_1 = state.range(0);
    int num_2 = state.range(1);
    for (auto _ : state) {
      auto A = cytnx::UniTensor(cytnx::ones({num_1, num_2, 3}));
    }
  }

  BENCHMARK(BM_ones)->Args({5, 3})->Args({10, 9});

}  // namespace BMTest_Tensor
