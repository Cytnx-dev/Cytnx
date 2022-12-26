#include <benchmark/benchmark.h>
#include <cytnx.hpp>
 
namespace BMTest_Hstack {

static void BM_Hstack_F64(benchmark::State& state) 
{
  //prepare data
  auto D_row = state.range(0);
  auto D_col = state.range(1);
  auto tens_num = state.range(2);
  std::vector<cytnx::Tensor> tens_list(tens_num);
  for(int i = 0; i < tens_num; ++i) {
    tens_list[i] = (i + 1) * cytnx::ones({D_row, D_col});
  }

  //start test here
  for (auto _ : state) {
    cytnx::Tensor hstack_tens = cytnx::algo::Hstack(tens_list);
  }
}

BENCHMARK(BM_Hstack_F64)
    //{D_row, D_col, tens_num}
    ->Args({10, 10, 10})
    ->Args({100, 100, 100})
    ->Args({1000, 1000, 10});
 

} //namespace

