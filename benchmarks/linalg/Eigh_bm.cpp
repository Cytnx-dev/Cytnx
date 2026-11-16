#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include <cytnx.hpp>

// linalg::Eigh with its default arguments, and the same calls with algorithm = "dc" (#1214).
// The benchmarks without "Dc" in their name build at every revision; the "Dc" ones need the
// algorithm option, and running both from one binary compares the two CPU drivers.
namespace BMTest_Eigh {

  using namespace cytnx;

  // A dense Hermitian n x n matrix with a spectrum without structure: no randomness, so every
  // revision decomposes the same matrix.
  Tensor hermitian(std::size_t n, unsigned int dtype) {
    Tensor out({n, n}, dtype);
    for (std::size_t i = 0; i < n; i++) {
      for (std::size_t j = 0; j < n; j++) {
        const double re = std::sin(0.37 * (i + 1.) * (j + 1.) + 0.11 * (i + j));
        const double im = std::sin(0.19 * (static_cast<double>(i) - static_cast<double>(j)));
        if (dtype == Type.ComplexDouble) {
          out.storage().at<cytnx_complex128>(i * n + j) = cytnx_complex128(re, im);
        } else {
          out.storage().at<cytnx_double>(i * n + j) = re;
        }
      }
    }
    return out;
  }

  // A rank-2 Z2-symmetric UniTensor with two n x n Hermitian blocks.
  UniTensor hermitian_z2(std::size_t n) {
    Bond row = Bond(BD_IN, {Qs(0) >> n, Qs(1) >> n}, {Symmetry::Zn(2)});
    UniTensor out({row, row.redirect()}, {"r", "c"}, 1);
    const Tensor block = hermitian(n, Type.Double);
    out.put_block(block, 0);
    out.put_block(block, 1);
    return out;
  }

  static void BM_Tensor_Eigh_F64(benchmark::State& state) {
    const Tensor T = hermitian(state.range(0), Type.Double);
    for (auto _ : state) {
      std::vector<Tensor> out = linalg::Eigh(T);
      benchmark::DoNotOptimize(out);
    }
  }
  BENCHMARK(BM_Tensor_Eigh_F64)->Arg(100)->Arg(450)->Arg(1250)->Unit(benchmark::kMillisecond);

  static void BM_Tensor_Eigh_C128(benchmark::State& state) {
    const Tensor T = hermitian(state.range(0), Type.ComplexDouble);
    for (auto _ : state) {
      std::vector<Tensor> out = linalg::Eigh(T);
      benchmark::DoNotOptimize(out);
    }
  }
  BENCHMARK(BM_Tensor_Eigh_C128)->Arg(100)->Arg(450)->Arg(1250)->Unit(benchmark::kMillisecond);

  static void BM_Tensor_Eigh_ValuesOnly_F64(benchmark::State& state) {
    const Tensor T = hermitian(state.range(0), Type.Double);
    for (auto _ : state) {
      std::vector<Tensor> out = linalg::Eigh(T, false);
      benchmark::DoNotOptimize(out);
    }
  }
  BENCHMARK(BM_Tensor_Eigh_ValuesOnly_F64)->Arg(450)->Arg(1250)->Unit(benchmark::kMillisecond);

  static void BM_BlockUT_Eigh_F64(benchmark::State& state) {
    const UniTensor T = hermitian_z2(state.range(0));
    for (auto _ : state) {
      std::vector<UniTensor> out = linalg::Eigh(T);
      benchmark::DoNotOptimize(out);
    }
  }
  BENCHMARK(BM_BlockUT_Eigh_F64)->Arg(450)->Arg(1250)->Unit(benchmark::kMillisecond);

  static void BM_Tensor_Eigh_Dc_F64(benchmark::State& state) {
    const Tensor T = hermitian(state.range(0), Type.Double);
    for (auto _ : state) {
      std::vector<Tensor> out = linalg::Eigh(T, true, false, "dc");
      benchmark::DoNotOptimize(out);
    }
  }
  BENCHMARK(BM_Tensor_Eigh_Dc_F64)->Arg(100)->Arg(450)->Arg(1250)->Unit(benchmark::kMillisecond);

  static void BM_Tensor_Eigh_Dc_C128(benchmark::State& state) {
    const Tensor T = hermitian(state.range(0), Type.ComplexDouble);
    for (auto _ : state) {
      std::vector<Tensor> out = linalg::Eigh(T, true, false, "dc");
      benchmark::DoNotOptimize(out);
    }
  }
  BENCHMARK(BM_Tensor_Eigh_Dc_C128)->Arg(100)->Arg(450)->Arg(1250)->Unit(benchmark::kMillisecond);

  static void BM_Tensor_Eigh_Dc_ValuesOnly_F64(benchmark::State& state) {
    const Tensor T = hermitian(state.range(0), Type.Double);
    for (auto _ : state) {
      std::vector<Tensor> out = linalg::Eigh(T, false, false, "dc");
      benchmark::DoNotOptimize(out);
    }
  }
  BENCHMARK(BM_Tensor_Eigh_Dc_ValuesOnly_F64)->Arg(450)->Arg(1250)->Unit(benchmark::kMillisecond);

  static void BM_BlockUT_Eigh_Dc_F64(benchmark::State& state) {
    const UniTensor T = hermitian_z2(state.range(0));
    for (auto _ : state) {
      std::vector<UniTensor> out = linalg::Eigh(T, true, false, "dc");
      benchmark::DoNotOptimize(out);
    }
  }
  BENCHMARK(BM_BlockUT_Eigh_Dc_F64)->Arg(450)->Arg(1250)->Unit(benchmark::kMillisecond);

}  // namespace BMTest_Eigh
