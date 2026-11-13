#include <benchmark/benchmark.h>

#include <cstdint>

#include "cytnx.hpp"

// In-place tensor arithmetic (`a += b`, `a *= b`, `a += scalar`) on contiguous CPU tensors,
// next to the out-of-place `a + b` it is meant to beat by not allocating a result.
//
// In the common in-place case the output buffer is the LHS buffer itself. These benchmarks
// time that case, so a kernel that handles aliased output more slowly than a fresh one shows
// up as in-place running behind out-of-place.
namespace BMTest_InplaceArithmetic {

  using cytnx::Tensor;

  namespace {

    Tensor MakeVector(std::int64_t n, unsigned int seed, unsigned int dtype) {
      const double low = -1.0;
      const double high = 1.0;
      return cytnx::random::uniform({static_cast<cytnx::cytnx_uint64>(n)}, low, high,
                                    cytnx::Device.cpu, seed, dtype);
    }

    void SetItems(benchmark::State& state, std::int64_t n) {
      state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * n);
    }

  }  // namespace

  // Out-of-place reference: allocates a fresh result every iteration.
  template <unsigned int dtype>
  static void BM_Tensor_Add_Tensor(benchmark::State& state) {
    const std::int64_t n = state.range(0);
    const Tensor a = MakeVector(n, 0, dtype);
    const Tensor b = MakeVector(n, 1, dtype);
    for (auto _ : state) {
      Tensor c = a + b;
      benchmark::DoNotOptimize(c);
    }
    SetItems(state, n);
  }

  // a += b: the output aliases a. a grows by at most 1 per iteration, far from overflow.
  template <unsigned int dtype>
  static void BM_Tensor_iAdd_Tensor(benchmark::State& state) {
    const std::int64_t n = state.range(0);
    Tensor a = MakeVector(n, 0, dtype);
    const Tensor b = MakeVector(n, 1, dtype);
    for (auto _ : state) {
      a += b;
      benchmark::DoNotOptimize(a);
    }
    SetItems(state, n);
  }

  // a *= b with b all ones, so a keeps its values and never drifts into subnormals.
  template <unsigned int dtype>
  static void BM_Tensor_iMul_Tensor(benchmark::State& state) {
    const std::int64_t n = state.range(0);
    Tensor a = MakeVector(n, 0, dtype);
    const Tensor b = cytnx::ones({static_cast<cytnx::cytnx_uint64>(n)}, dtype);
    for (auto _ : state) {
      a *= b;
      benchmark::DoNotOptimize(a);
    }
    SetItems(state, n);
  }

  // a += scalar: the broadcast-scalar loop, output again aliasing a.
  template <unsigned int dtype>
  static void BM_Tensor_iAdd_Scalar(benchmark::State& state) {
    const std::int64_t n = state.range(0);
    Tensor a = MakeVector(n, 0, dtype);
    for (auto _ : state) {
      a += 0.5;
      benchmark::DoNotOptimize(a);
    }
    SetItems(state, n);
  }

#define REGISTER_INPLACE_ARITHMETIC_BENCHMARKS(TypeName, TypeEnum) \
  BENCHMARK_TEMPLATE(BM_Tensor_Add_Tensor, TypeEnum)               \
    ->Name("BM_Tensor_Add_Tensor_" #TypeName)                      \
    ->Arg(1 << 16)                                                 \
    ->Arg(1 << 20)                                                 \
    ->Unit(benchmark::kMicrosecond);                               \
  BENCHMARK_TEMPLATE(BM_Tensor_iAdd_Tensor, TypeEnum)              \
    ->Name("BM_Tensor_iAdd_Tensor_" #TypeName)                     \
    ->Arg(1 << 16)                                                 \
    ->Arg(1 << 20)                                                 \
    ->Unit(benchmark::kMicrosecond);                               \
  BENCHMARK_TEMPLATE(BM_Tensor_iMul_Tensor, TypeEnum)              \
    ->Name("BM_Tensor_iMul_Tensor_" #TypeName)                     \
    ->Arg(1 << 16)                                                 \
    ->Arg(1 << 20)                                                 \
    ->Unit(benchmark::kMicrosecond);                               \
  BENCHMARK_TEMPLATE(BM_Tensor_iAdd_Scalar, TypeEnum)              \
    ->Name("BM_Tensor_iAdd_Scalar_" #TypeName)                     \
    ->Arg(1 << 16)                                                 \
    ->Arg(1 << 20)                                                 \
    ->Unit(benchmark::kMicrosecond);

  REGISTER_INPLACE_ARITHMETIC_BENCHMARKS(Float, cytnx::Type.Float)
  REGISTER_INPLACE_ARITHMETIC_BENCHMARKS(Double, cytnx::Type.Double)
  REGISTER_INPLACE_ARITHMETIC_BENCHMARKS(ComplexFloat, cytnx::Type.ComplexFloat)
  REGISTER_INPLACE_ARITHMETIC_BENCHMARKS(ComplexDouble, cytnx::Type.ComplexDouble)

#undef REGISTER_INPLACE_ARITHMETIC_BENCHMARKS

}  // namespace BMTest_InplaceArithmetic
