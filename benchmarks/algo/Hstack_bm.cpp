#include <benchmark/benchmark.h>
#include <cytnx.hpp>

using namespace cytnx;

namespace BMTest_Hstack {

  static void BM_Hstack_F64(benchmark::State& state) {
    // prepare data
    auto D_row = state.range(0);
    auto D_col = state.range(1);
    auto tens_num = state.range(2);
    std::vector<Tensor> tens_list(tens_num);
    for (int i = 0; i < tens_num; ++i) {
      tens_list[i] = (i + 1) * ones({D_row, D_col});
    }

    // start test here
    for (auto _ : state) {
      Tensor hstack_tens = algo::Hstack(tens_list);
    }
  }
  BENCHMARK(BM_Hstack_F64)
    //{D_row, D_col, tens_num}
    ->Args({10, 10, 10})
    ->Args({100, 100, 100})
    ->Args({1000, 1000, 10});

  static void BM_Hstack_C128(benchmark::State& state) {
    // prepare data
    auto D_row = state.range(0);
    auto D_col = state.range(1);
    auto tens_num = state.range(2);
    std::vector<Tensor> tens_list(tens_num);
    for (int i = 0; i < tens_num; ++i) {
      tens_list[i] = (i + 1) * ones({D_row, D_col}, Type.ComplexDouble);
    }

    // start test here
    for (auto _ : state) {
      Tensor hstack_tens = algo::Hstack(tens_list);
    }
  }
  BENCHMARK(BM_Hstack_C128)
    //{D_row, D_col, tens_num}
    ->Args({10, 10, 10})
    ->Args({100, 100, 100})
    ->Args({1000, 1000, 10});

  static void BM_Hstack_F64_non_conti(benchmark::State& state) {
    // prepare data
    auto D_row = state.range(0);
    auto D_col = state.range(1);
    auto tens_num = state.range(2);
    std::vector<Tensor> tens_list(tens_num);
    for (int i = 0; i < tens_num; ++i) {
      tens_list[i] = (i + 1) * ones({D_row, D_col});
      tens_list[i].permute_({1, 0});  // change row <-> col
    }

    // start test here
    for (auto _ : state) {
      Tensor hstack_tens = algo::Hstack(tens_list);
    }
  }
  BENCHMARK(BM_Hstack_F64_non_conti)
    //{D_row, D_col, tens_num}
    ->Args({10, 10, 10})
    ->Args({100, 100, 100})
    ->Args({1000, 1000, 10});

}  // namespace BMTest_Hstack
