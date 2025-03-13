#include <benchmark/benchmark.h>
#include <cytnx.hpp>

// Cytnx test
namespace BMTest_linalg_basic {
  static const int D_test = 1000;

  // Abs
  static void BM_linalg_basic_Abs(benchmark::State& state) {
    auto D = state.range(0);
    const double low = -1;
    const double high = 1;
    const int device = cytnx::Device.cpu;
    const unsigned int seed = 0;
    auto dtype = state.range(1);
    auto A = cytnx::random::uniform({D, D}, low, high, device, seed, dtype);
    for (auto _ : state) {
      cytnx::linalg::Abs(A);
    }
  }
  BENCHMARK(BM_linalg_basic_Abs)
    ->Args({D_test, cytnx::Type.Float})
    ->Args({D_test, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble})
    ->Unit(benchmark::kMillisecond);

  // Add
  static void BM_linalg_basic_Add(benchmark::State& state) {
    auto D = state.range(0);
    const double low = -1;
    const double high = 1;
    const int device = cytnx::Device.cpu;
    const unsigned int seed = 0;
    auto dtype1 = state.range(1);
    auto dtype2 = state.range(2);
    auto A = cytnx::random::uniform({D, D}, low, high, device, seed, dtype1);
    auto B = cytnx::random::uniform({D, D}, low, high, device, seed, dtype2);
    for (auto _ : state) {
      cytnx::linalg::Add(A, B);
    }
  }
  BENCHMARK(BM_linalg_basic_Add)
    ->Args({D_test, cytnx::Type.Float, cytnx::Type.Float})
    ->Args({D_test, cytnx::Type.Float, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.Double, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble, cytnx::Type.ComplexDouble})
    ->Unit(benchmark::kMillisecond);

  // Conj_
  static void BM_linalg_basic_Conj_(benchmark::State& state) {
    auto D = state.range(0);
    const double low = -1;
    const double high = 1;
    const int device = cytnx::Device.cpu;
    const unsigned int seed = 0;
    auto dtype = state.range(1);
    auto A = cytnx::random::uniform({D, D}, low, high, device, seed, dtype);
    for (auto _ : state) {
      cytnx::linalg::Conj_(A);
    }
  }
  BENCHMARK(BM_linalg_basic_Conj_)
    ->Args({D_test, cytnx::Type.ComplexFloat})
    ->Args({D_test, cytnx::Type.ComplexDouble})
    ->Unit(benchmark::kMillisecond);

  // Cpr_1
  static void BM_linalg_basic_Cpr1(benchmark::State& state) {
    auto D = state.range(0);
    const double low = -1;
    const double high = 1;
    const int device = cytnx::Device.cpu;
    const unsigned int seed = 0;
    auto dtype = state.range(1);
    auto A = cytnx::random::uniform({D, D}, low, high, device, seed, dtype);
    for (auto _ : state) {
      cytnx::linalg::Cpr(A, 2);
    }
  }
  BENCHMARK(BM_linalg_basic_Cpr1)
    ->Args({D_test, cytnx::Type.Float})
    ->Args({D_test, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble})
    ->Unit(benchmark::kMillisecond);

  // Cpr_2
  static void BM_linalg_basic_Cpr2(benchmark::State& state) {
    auto D = state.range(0);
    const double low = -1;
    const double high = 1;
    const int device = cytnx::Device.cpu;
    const unsigned int seed = 0;
    auto dtype1 = state.range(1);
    auto dtype2 = state.range(2);
    auto A = cytnx::random::uniform({D, D}, low, high, device, seed, dtype1);
    auto B = cytnx::random::uniform({D, D}, low, high, device, seed, dtype2);
    for (auto _ : state) {
      cytnx::linalg::Cpr(A, B);
    }
  }
  BENCHMARK(BM_linalg_basic_Cpr2)
    ->Args({D_test, cytnx::Type.Float, cytnx::Type.Float})
    ->Args({D_test, cytnx::Type.Float, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.Double, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble, cytnx::Type.ComplexDouble})
    ->Unit(benchmark::kMillisecond);

  // Diag
  static void BM_linalg_basic_Diag(benchmark::State& state) {
    auto D = state.range(0);
    const double low = -1;
    const double high = 1;
    const int device = cytnx::Device.cpu;
    const unsigned int seed = 0;
    auto dtype = state.range(1);
    auto A = cytnx::random::uniform({D}, low, high, device, seed, dtype);
    for (auto _ : state) {
      cytnx::linalg::Diag(A);
    }
  }
  BENCHMARK(BM_linalg_basic_Diag)
    ->Args({D_test, cytnx::Type.Float})
    ->Args({D_test, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble})
    ->Unit(benchmark::kMillisecond);

  // Div
  static void BM_linalg_basic_Div(benchmark::State& state) {
    auto D = state.range(0);
    const double low = 0.2;
    const double high = 1;
    const int device = cytnx::Device.cpu;
    const unsigned int seed = 0;
    auto dtype1 = state.range(1);
    auto dtype2 = state.range(2);
    auto A = cytnx::random::uniform({D, D}, low, high, device, seed, dtype1);
    auto B = cytnx::random::uniform({D, D}, low, high, device, seed, dtype2);
    for (auto _ : state) {
      cytnx::linalg::Add(A, B);
    }
  }
  BENCHMARK(BM_linalg_basic_Div)
    ->Args({D_test, cytnx::Type.Float, cytnx::Type.Float})
    ->Args({D_test, cytnx::Type.Float, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.Double, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble, cytnx::Type.ComplexDouble})
    ->Unit(benchmark::kMillisecond);

  // Exp_
  static void BM_linalg_basic_Exp_(benchmark::State& state) {
    auto D = state.range(0);
    const double low = -1;
    const double high = 1;
    const int device = cytnx::Device.cpu;
    const unsigned int seed = 0;
    auto dtype = state.range(1);
    auto A = cytnx::random::uniform({D, D}, low, high, device, seed, dtype);
    for (auto _ : state) {
      cytnx::linalg::Exp_(A);
    }
  }
  BENCHMARK(BM_linalg_basic_Exp_)
    ->Args({D_test, cytnx::Type.Float})
    ->Args({D_test, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble})
    ->Unit(benchmark::kMillisecond);

  // Inv_
  static void BM_linalg_basic_Inv_(benchmark::State& state) {
    auto D = state.range(0);
    const double low = -1;
    const double high = 1;
    const double clip = 1.0e-7;
    const int device = cytnx::Device.cpu;
    const unsigned int seed = 0;
    auto dtype = state.range(1);
    auto A = cytnx::random::uniform({D, D}, low, high, device, seed, dtype);
    for (auto _ : state) {
      cytnx::linalg::Inv_(A, clip);
    }
  }
  BENCHMARK(BM_linalg_basic_Inv_)
    ->Args({D_test, cytnx::Type.Float})
    ->Args({D_test, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble})
    ->Unit(benchmark::kMillisecond);

  // Matmul
  static void BM_linalg_basic_Matmul(benchmark::State& state) {
    auto D = state.range(0);
    const double low = -1;
    const double high = 1;
    const int device = cytnx::Device.cpu;
    const unsigned int seed = 0;
    auto dtype1 = state.range(1);
    auto dtype2 = state.range(2);
    auto A = cytnx::random::uniform({D, D}, low, high, device, seed, dtype1);
    auto B = cytnx::random::uniform({D, D}, low, high, device, seed, dtype2);
    for (auto _ : state) {
      cytnx::linalg::Matmul(A, B);
    }
  }
  BENCHMARK(BM_linalg_basic_Matmul)
    ->Args({D_test, cytnx::Type.Float, cytnx::Type.Float})
    ->Args({D_test, cytnx::Type.Float, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.Double, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble, cytnx::Type.ComplexDouble})
    ->Unit(benchmark::kMillisecond);

  // Matmul_dg
  static void BM_linalg_basic_Matmul_dg(benchmark::State& state) {
    auto D = state.range(0);
    const double low = -1;
    const double high = 1;
    const int device = cytnx::Device.cpu;
    const unsigned int seed = 0;
    auto dtype1 = state.range(1);
    auto dtype2 = state.range(2);
    auto A = cytnx::random::uniform({D}, low, high, device, seed, dtype1);
    auto B = cytnx::random::uniform({D, D}, low, high, device, seed, dtype2);
    for (auto _ : state) {
      cytnx::linalg::Matmul_dg(A, B);
    }
  }
  BENCHMARK(BM_linalg_basic_Matmul_dg)
    ->Args({D_test, cytnx::Type.Float, cytnx::Type.Float})
    ->Args({D_test, cytnx::Type.Float, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.Double, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble, cytnx::Type.ComplexDouble})
    ->Unit(benchmark::kMillisecond);

  // Mod
  static void BM_linalg_basic_Mod(benchmark::State& state) {
    auto D = state.range(0);
    const int device = cytnx::Device.cpu;
    auto dtype1 = state.range(1);
    auto dtype2 = state.range(2);
    auto A = cytnx::ones({D, D}, dtype1, device);
    auto B = cytnx::ones({D, D}, dtype2, device);
    for (auto _ : state) {
      cytnx::linalg::Mod(A, B);
    }
  }
  BENCHMARK(BM_linalg_basic_Mod)
    ->Args({D_test, cytnx::Type.Uint64, cytnx::Type.Int64})
    ->Args({D_test, cytnx::Type.Uint64, cytnx::Type.Int32})
    ->Args({D_test, cytnx::Type.Int64, cytnx::Type.Int64})
    ->Unit(benchmark::kMillisecond);

  // Mul
  static void BM_linalg_basic_Mul(benchmark::State& state) {
    auto D = state.range(0);
    const double low = -1;
    const double high = 1;
    const int device = cytnx::Device.cpu;
    const unsigned int seed = 0;
    auto dtype1 = state.range(1);
    auto dtype2 = state.range(2);
    auto A = cytnx::random::uniform({D, D}, low, high, device, seed, dtype1);
    auto B = cytnx::random::uniform({D, D}, low, high, device, seed, dtype2);
    for (auto _ : state) {
      cytnx::linalg::Mul(A, B);
    }
  }
  BENCHMARK(BM_linalg_basic_Mul)
    ->Args({D_test, cytnx::Type.Float, cytnx::Type.Float})
    ->Args({D_test, cytnx::Type.Float, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.Double, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble, cytnx::Type.ComplexDouble})
    ->Unit(benchmark::kMillisecond);

  // Outer
  static void BM_linalg_basic_Outer(benchmark::State& state) {
    auto D = state.range(0);
    const double low = -1;
    const double high = 1;
    const int device = cytnx::Device.cpu;
    const unsigned int seed = 0;
    auto dtype1 = state.range(1);
    auto dtype2 = state.range(2);
    auto A = cytnx::random::uniform({D}, low, high, device, seed, dtype1);
    auto B = cytnx::random::uniform({D}, low, high, device, seed, dtype2);
    for (auto _ : state) {
      cytnx::linalg::Outer(A, B);
    }
  }
  BENCHMARK(BM_linalg_basic_Outer)
    ->Args({D_test, cytnx::Type.Float, cytnx::Type.Float})
    ->Args({D_test, cytnx::Type.Float, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.Double, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble, cytnx::Type.ComplexDouble})
    ->Unit(benchmark::kMillisecond);

  // Pow
  static void BM_linalg_basic_Pow(benchmark::State& state) {
    auto D = state.range(0);
    const double low = -1;
    const double high = 1;
    const double p = 3.;
    const int device = cytnx::Device.cpu;
    const unsigned int seed = 0;
    auto dtype = state.range(1);
    auto A = cytnx::random::uniform({D, D}, low, high, device, seed, dtype);
    for (auto _ : state) {
      cytnx::linalg::Pow(A, p);
    }
  }
  BENCHMARK(BM_linalg_basic_Pow)
    ->Args({D_test, cytnx::Type.Float})
    ->Args({D_test, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble})
    ->Unit(benchmark::kMillisecond);

  // Abs
  static void BM_linalg_basic_Qr(benchmark::State& state) {
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
  BENCHMARK(BM_linalg_basic_Qr)
    ->Args({D_test, cytnx::Type.Float})
    ->Args({D_test, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble})
    ->Unit(benchmark::kMillisecond);

  // Sub
  static void BM_linalg_basic_Sub(benchmark::State& state) {
    auto D = state.range(0);
    const double low = -1;
    const double high = 1;
    const int device = cytnx::Device.cpu;
    const unsigned int seed = 0;
    auto dtype1 = state.range(1);
    auto dtype2 = state.range(2);
    auto A = cytnx::random::uniform({D, D}, low, high, device, seed, dtype1);
    auto B = cytnx::random::uniform({D, D}, low, high, device, seed, dtype2);
    for (auto _ : state) {
      cytnx::linalg::Sub(A, B);
    }
  }
  BENCHMARK(BM_linalg_basic_Sub)
    ->Args({D_test, cytnx::Type.Float, cytnx::Type.Float})
    ->Args({D_test, cytnx::Type.Float, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.Double, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble, cytnx::Type.ComplexDouble})
    ->Unit(benchmark::kMillisecond);

  // iAdd
  static void BM_linalg_basic_iAdd(benchmark::State& state) {
    auto D = state.range(0);
    const double low = -1;
    const double high = 1;
    const int device = cytnx::Device.cpu;
    const unsigned int seed = 0;
    auto dtype1 = state.range(1);
    auto dtype2 = state.range(2);
    auto A = cytnx::random::uniform({D, D}, low, high, device, seed, dtype1);
    auto B = cytnx::random::uniform({D, D}, low, high, device, seed, dtype2);
    for (auto _ : state) {
      cytnx::linalg::iAdd(A, B);
    }
  }
  BENCHMARK(BM_linalg_basic_iAdd)
    ->Args({D_test, cytnx::Type.Float, cytnx::Type.Float})
    ->Args({D_test, cytnx::Type.Float, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.Double, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble, cytnx::Type.ComplexDouble})
    ->Unit(benchmark::kMillisecond);

  // iDiv
  static void BM_linalg_basic_iDiv(benchmark::State& state) {
    auto D = state.range(0);
    const double low = 0.2;
    const double high = 1;
    const int device = cytnx::Device.cpu;
    const unsigned int seed = 0;
    auto dtype1 = state.range(1);
    auto dtype2 = state.range(2);
    auto A = cytnx::random::uniform({D, D}, low, high, device, seed, dtype1);
    auto B = cytnx::random::uniform({D, D}, low, high, device, seed, dtype2);
    for (auto _ : state) {
      cytnx::linalg::iDiv(A, B);
    }
  }
  BENCHMARK(BM_linalg_basic_iDiv)
    ->Args({D_test, cytnx::Type.Float, cytnx::Type.Float})
    ->Args({D_test, cytnx::Type.Float, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.Double, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble, cytnx::Type.ComplexDouble})
    ->Unit(benchmark::kMillisecond);

  // iMul
  static void BM_linalg_basic_iMul(benchmark::State& state) {
    auto D = state.range(0);
    const double low = -1;
    const double high = 1;
    const int device = cytnx::Device.cpu;
    const unsigned int seed = 0;
    auto dtype1 = state.range(1);
    auto dtype2 = state.range(2);
    auto A = cytnx::random::uniform({D, D}, low, high, device, seed, dtype1);
    auto B = cytnx::random::uniform({D, D}, low, high, device, seed, dtype2);
    for (auto _ : state) {
      cytnx::linalg::iMul(A, B);
    }
  }
  BENCHMARK(BM_linalg_basic_iMul)
    ->Args({D_test, cytnx::Type.Float, cytnx::Type.Float})
    ->Args({D_test, cytnx::Type.Float, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.Double, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble, cytnx::Type.ComplexDouble})
    ->Unit(benchmark::kMillisecond);

  // iSub
  static void BM_linalg_basic_iSub(benchmark::State& state) {
    auto D = state.range(0);
    const double low = -1;
    const double high = 1;
    const int device = cytnx::Device.cpu;
    const unsigned int seed = 0;
    auto dtype1 = state.range(1);
    auto dtype2 = state.range(2);
    auto A = cytnx::random::uniform({D, D}, low, high, device, seed, dtype1);
    auto B = cytnx::random::uniform({D, D}, low, high, device, seed, dtype2);
    for (auto _ : state) {
      cytnx::linalg::iSub(A, B);
    }
  }
  BENCHMARK(BM_linalg_basic_iSub)
    ->Args({D_test, cytnx::Type.Float, cytnx::Type.Float})
    ->Args({D_test, cytnx::Type.Float, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.Double, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble, cytnx::Type.ComplexDouble})
    ->Unit(benchmark::kMillisecond);

  // astype
  static void BM_linalg_basic_astype(benchmark::State& state) {
    auto D = state.range(0);
    const double low = -1;
    const double high = 1;
    const int device = cytnx::Device.cpu;
    const unsigned int seed = 0;
    auto dtype1 = state.range(1);
    auto dtype2 = state.range(2);
    auto A = cytnx::random::uniform({D, D}, low, high, device, seed, dtype1);
    for (auto _ : state) {
      A.astype(dtype2);
    }
  }
  BENCHMARK(BM_linalg_basic_astype)
    ->Args({D_test, cytnx::Type.Double, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.Double, cytnx::Type.Int64})
    ->Args({D_test, cytnx::Type.Double, cytnx::Type.Float})
    ->Unit(benchmark::kMillisecond);

  // real
  static void BM_linalg_basic_real(benchmark::State& state) {
    auto D = state.range(0);
    const double low = -1;
    const double high = 1;
    const int device = cytnx::Device.cpu;
    const unsigned int seed = 0;
    auto dtype = state.range(1);
    auto A = cytnx::random::uniform({D, D}, low, high, device, seed, dtype);
    for (auto _ : state) {
      A.real();
    }
  }
  BENCHMARK(BM_linalg_basic_real)
    ->Args({D_test, cytnx::Type.ComplexDouble})
    ->Args({D_test, cytnx::Type.ComplexFloat})
    ->Unit(benchmark::kMillisecond);

  // imag
  static void BM_linalg_basic_imag(benchmark::State& state) {
    auto D = state.range(0);
    const double low = -1;
    const double high = 1;
    const int device = cytnx::Device.cpu;
    const unsigned int seed = 0;
    auto dtype = state.range(1);
    auto A = cytnx::random::uniform({D, D}, low, high, device, seed, dtype);
    for (auto _ : state) {
      A.imag();
    }
  }
  BENCHMARK(BM_linalg_basic_imag)
    ->Args({D_test, cytnx::Type.ComplexDouble})
    ->Args({D_test, cytnx::Type.ComplexFloat})
    ->Unit(benchmark::kMillisecond);

  // Fill
  static void BM_linalg_basic_fill(benchmark::State& state) {
    auto D = state.range(0);
    const int device = cytnx::Device.cpu;
    auto dtype = state.range(1);
    auto A = cytnx::zeros({D, D}, dtype, device);
    for (auto _ : state) {
      A.fill(2.0);
    }
  }
  BENCHMARK(BM_linalg_basic_fill)
    ->Args({D_test, cytnx::Type.ComplexDouble})
    ->Args({D_test, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.Uint32})
    ->Unit(benchmark::kMillisecond);

  // arange
  static void BM_linalg_basic_arange(benchmark::State& state) {
    auto D = state.range(0);
    const int device = cytnx::Device.cpu;
    for (auto _ : state) {
      cytnx::arange(D * D);
    }
  }
  BENCHMARK(BM_linalg_basic_arange)->Args({D_test})->Unit(benchmark::kMillisecond);

  // range
  static void BM_linalg_basic_range(benchmark::State& state) {
    auto D = state.range(0);
    const double low = -1;
    const double high = 1;
    const int device = cytnx::Device.cpu;
    const unsigned int seed = 0;
    auto dtype = state.range(1);
    auto A = cytnx::random::uniform({D, D}, low, high, device, seed, dtype);
    for (auto _ : state) {
      auto tmp = A(cytnx::Accessor::range(0, D - 1, 1), cytnx::Accessor::range(0, D - 1, 2));
    }
  }
  BENCHMARK(BM_linalg_basic_range)
    ->Args({D_test, cytnx::Type.ComplexDouble})
    ->Args({D_test, cytnx::Type.ComplexFloat})
    ->Unit(benchmark::kMillisecond);

  // Trace
  static void BM_linalg_basic_Trace(benchmark::State& state) {
    auto D = state.range(0);
    const double low = -1;
    const double high = 1;
    const int device = cytnx::Device.cpu;
    const unsigned int seed = 0;
    auto dtype = state.range(1);
    auto A = cytnx::random::uniform({D, D}, low, high, device, seed, dtype);
    for (auto _ : state) {
      cytnx::linalg::Trace(A);
    }
  }
  BENCHMARK(BM_linalg_basic_Trace)
    ->Args({D_test, cytnx::Type.Float})
    ->Args({D_test, cytnx::Type.Double})
    ->Args({D_test, cytnx::Type.ComplexDouble})
    ->Unit(benchmark::kMillisecond);

}  // namespace BMTest_linalg_basic
