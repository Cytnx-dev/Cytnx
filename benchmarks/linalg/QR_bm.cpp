#include <benchmark/benchmark.h>
#include <cytnx.hpp>

// Cytnx test
namespace BMTest_QR {
  static const int D_test = 50;

  // QR
  static void BM_Qr(benchmark::State& state) {
    auto D = state.range(0);
    const double low = -1;
    const double high = 1;
    const int device = cytnx::Device.cpu;
    const unsigned int seed = 0;
    auto dtype = state.range(1);
    auto A = cytnx::random::uniform({D, D}, low, high, device, seed, dtype);
    for (auto _ : state) {
      cytnx::linalg::Qr(A);
    }
  }
  BENCHMARK(BM_Qr)
    ->Args({D_test, cytnx::Type.Float})
    ->Args({D_test, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble})
    ->Unit(benchmark::kMillisecond);
}  // namespace BMTest_QR
