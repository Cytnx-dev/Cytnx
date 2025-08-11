#include <benchmark/benchmark.h>
#include <cytnx.hpp>
using namespace cytnx;

namespace BMTest_Directsum {

  static void BM_Directsum_F64(benchmark::State& state) {
    // prepare data
    auto D = state.range(0);
    auto shape = std::vector<cytnx_uint64>{D, D, D};
    auto T1 = ones(shape, Type.Double);
    auto T2 = ones(shape, Type.Double) * 2.0;
    std::vector<cytnx_uint64> shared_axes = {0};

    // start test here
    for (auto _ : state) {
      Tensor directsum_tens = linalg::Directsum(T1, T2, shared_axes);
    }
  }
  BENCHMARK(BM_Directsum_F64)->Args({1})->Args({10})->Args({100});

  static void BM_Directsum_C128(benchmark::State& state) {
    // prepare data
    auto D = state.range(0);
    auto shape = std::vector<cytnx_uint64>{D, D, D};
    auto T1 = ones(shape, Type.ComplexDouble);
    auto T2 = ones(shape, Type.ComplexDouble) * 2.0;
    std::vector<cytnx_uint64> shared_axes = {0};

    // start test here
    for (auto _ : state) {
      Tensor directsum_tens = linalg::Directsum(T1, T2, shared_axes);
    }
  }
  BENCHMARK(BM_Directsum_C128)->Args({1})->Args({10})->Args({100});

  static void BM_Directsum_F64_non_conti(benchmark::State& state) {
    // prepare data
    auto D = state.range(0);
    auto shape = std::vector<cytnx_uint64>{D, D, D};
    auto T1 = ones(shape, Type.Double);
    auto T2 = ones(shape, Type.Double) * 2.0;
    T1.permute_({0, 2, 1});
    T2.permute_({0, 2, 1});
    std::vector<cytnx_uint64> shared_axes = {0};

    // start test here
    for (auto _ : state) {
      Tensor directsum_tens = linalg::Directsum(T1, T2, shared_axes);
    }
  }
  BENCHMARK(BM_Directsum_F64_non_conti)->Args({1})->Args({10})->Args({100});

}  // namespace BMTest_Directsum
