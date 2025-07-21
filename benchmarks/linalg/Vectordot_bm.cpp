#include <benchmark/benchmark.h>
#include <cytnx.hpp>
using namespace cytnx;

namespace BMTest_Vectordot {

  static void BM_Tensor_Vectordot_F64_cpu(benchmark::State& state) {
    // prepare data
    auto len = state.range(0);
    double low = -1.0;
    double high = 1.0;
    int seed = 0;
    Tensor a = random::uniform(len, low, high, Device.cpu, seed, Type.Double);
    Tensor b = random::uniform(len, low, high, Device.cpu, seed, Type.Double);

    // start test here
    for (auto _ : state) {
      auto c = linalg::Vectordot(a, b);
    }
  }
  BENCHMARK(BM_Tensor_Vectordot_F64_cpu)
    ->Args({1000})
    ->Args({100000})
    ->Args({10000000})
    ->Unit(benchmark::kMillisecond);

  static void BM_Tensor_Vectordot_C128_conj_cpu(benchmark::State& state) {
    // prepare data
    auto len = state.range(0);
    double low = -1.0;
    double high = 1.0;
    int seed = 0;
    Tensor a = random::uniform(len, low, high, Device.cpu, seed, Type.ComplexDouble);
    Tensor b = random::uniform(len, low, high, Device.cpu, seed, Type.ComplexDouble);

    // start test here
    for (auto _ : state) {
      auto c = linalg::Vectordot(a, b, true);
    }
  }
  BENCHMARK(BM_Tensor_Vectordot_C128_conj_cpu)
    ->Args({1000})
    ->Args({100000})
    ->Args({10000000})
    ->Unit(benchmark::kMillisecond);

  static void BM_Tensor_Vectordot_C128_noconj_cpu(benchmark::State& state) {
    // prepare data
    auto len = state.range(0);
    double low = -1.0;
    double high = 1.0;
    int seed = 0;
    Tensor a = random::uniform(len, low, high, Device.cpu, seed, Type.ComplexDouble);
    Tensor b = random::uniform(len, low, high, Device.cpu, seed, Type.ComplexDouble);

    // start test here
    for (auto _ : state) {
      auto c = linalg::Vectordot(a, b, false);
    }
  }
  BENCHMARK(BM_Tensor_Vectordot_C128_noconj_cpu)
    ->Args({1000})
    ->Args({100000})
    ->Args({10000000})
    ->Unit(benchmark::kMillisecond);

}  // namespace BMTest_Vectordot
