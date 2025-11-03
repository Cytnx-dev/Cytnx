#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>

#include "cytnx.hpp"

namespace BMTest_Abs {

  template <unsigned int dtype, cytnx::cytnx_int32 device = cytnx::Device.cpu>
  static void BM_Tensor_Abs_Template(benchmark::State& state) {
    const auto size = state.range(0);
    const cytnx::cytnx_double kLow = -10.0;
    const cytnx::cytnx_double kHigh = 10.0;
    const unsigned int kSeed = 0;
    const auto tensor_a =
      cytnx::random::random_tensor({size, size}, kLow, kHigh, device, kSeed, dtype);
    for (auto _ : state) {
      auto result = cytnx::linalg::Abs(tensor_a);
      benchmark::DoNotOptimize(result);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * size * size *
                            cytnx::Type.typeSize(dtype));
  }

  template <unsigned int dtype, cytnx::cytnx_int32 device = cytnx::Device.cpu>
  static void BM_Tensor_Abs_inplace_Template(benchmark::State& state) {
    const auto size = state.range(0);
    const cytnx::cytnx_double kLow = -10.0;
    const cytnx::cytnx_double kHigh = 10.0;
    const unsigned int kSeed = 0;
    const auto original_tensor =
      cytnx::random::random_tensor({size, size}, kLow, kHigh, device, kSeed, dtype);
    for (auto _ : state) {
      auto tensor_copy = original_tensor.clone();
      cytnx::linalg::Abs_(tensor_copy);
      benchmark::DoNotOptimize(tensor_copy);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * size * size *
                            cytnx::Type.typeSize(dtype));
  }

  template <unsigned int dtype, cytnx::cytnx_int32 device = cytnx::Device.cpu>
  static void BM_Tensor_Abs_1D_Template(benchmark::State& state) {
    const auto size = state.range(0);
    const cytnx::cytnx_double kLow = -10.0;
    const cytnx::cytnx_double kHigh = 10.0;
    const unsigned int kSeed = 0;
    const auto tensor_a = cytnx::random::random_tensor({size}, kLow, kHigh, device, kSeed, dtype);
    for (auto _ : state) {
      auto result = cytnx::linalg::Abs(tensor_a);
      benchmark::DoNotOptimize(result);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * size *
                            cytnx::Type.typeSize(dtype));
  }

// Macro to register 2D Abs benchmarks for a specific type
#define REGISTER_ABS_2D_BENCHMARK(TypeName, TypeEnum)  \
  BENCHMARK_TEMPLATE(BM_Tensor_Abs_Template, TypeEnum) \
    ->Name("BM_Tensor_Abs_" #TypeName "_2D")           \
    ->Args({10})                                       \
    ->Args({100})                                      \
    ->Args({1000})                                     \
    ->Unit(benchmark::kMillisecond);

// Macro to register 2D Abs_ (in-place) benchmarks for a specific type
#define REGISTER_ABS_INPLACE_2D_BENCHMARK(TypeName, TypeEnum)  \
  BENCHMARK_TEMPLATE(BM_Tensor_Abs_inplace_Template, TypeEnum) \
    ->Name("BM_Tensor_Abs_inplace_" #TypeName "_2D")           \
    ->Args({10})                                               \
    ->Args({100})                                              \
    ->Args({1000})                                             \
    ->Unit(benchmark::kMillisecond);

// Macro to register 1D Abs benchmarks for a specific type
#define REGISTER_ABS_1D_BENCHMARK(TypeName, TypeEnum)     \
  BENCHMARK_TEMPLATE(BM_Tensor_Abs_1D_Template, TypeEnum) \
    ->Name("BM_Tensor_Abs_" #TypeName "_1D")              \
    ->Args({100})                                         \
    ->Args({10000})                                       \
    ->Args({1000000})                                     \
    ->Unit(benchmark::kMillisecond);

// Macro to register GPU 2D Abs benchmarks for a specific type
#define REGISTER_GPU_ABS_2D_BENCHMARK(TypeName, TypeEnum)                  \
  BENCHMARK_TEMPLATE(BM_Tensor_Abs_Template, TypeEnum, cytnx::Device.cuda) \
    ->Name("BM_gpu_Tensor_Abs_" #TypeName "_2D")                           \
    ->Args({10})                                                           \
    ->Args({100})                                                          \
    ->Args({1000})                                                         \
    ->Unit(benchmark::kMillisecond);

// Macro to register GPU 2D Abs_ (in-place) benchmarks for a specific type
#define REGISTER_GPU_ABS_INPLACE_2D_BENCHMARK(TypeName, TypeEnum)                  \
  BENCHMARK_TEMPLATE(BM_Tensor_Abs_inplace_Template, TypeEnum, cytnx::Device.cuda) \
    ->Name("BM_gpu_Tensor_Abs_inplace_" #TypeName "_2D")                           \
    ->Args({10})                                                                   \
    ->Args({100})                                                                  \
    ->Args({1000})                                                                 \
    ->Unit(benchmark::kMillisecond);

// Macro to register GPU 1D Abs benchmarks for a specific type
#define REGISTER_GPU_ABS_1D_BENCHMARK(TypeName, TypeEnum)                     \
  BENCHMARK_TEMPLATE(BM_Tensor_Abs_1D_Template, TypeEnum, cytnx::Device.cuda) \
    ->Name("BM_gpu_Tensor_Abs_" #TypeName "_1D")                              \
    ->Args({100})                                                             \
    ->Args({10000})                                                           \
    ->Args({1000000})                                                         \
    ->Unit(benchmark::kMillisecond);

  // 2D Abs benchmarks for all types
  REGISTER_ABS_2D_BENCHMARK(Int16, cytnx::Type.Int16)
  REGISTER_ABS_2D_BENCHMARK(Int32, cytnx::Type.Int32)
  REGISTER_ABS_2D_BENCHMARK(Int64, cytnx::Type.Int64)
  REGISTER_ABS_2D_BENCHMARK(Uint16, cytnx::Type.Uint16)
  REGISTER_ABS_2D_BENCHMARK(Uint32, cytnx::Type.Uint32)
  REGISTER_ABS_2D_BENCHMARK(Uint64, cytnx::Type.Uint64)
  REGISTER_ABS_2D_BENCHMARK(Float, cytnx::Type.Float)
  REGISTER_ABS_2D_BENCHMARK(Double, cytnx::Type.Double)
  REGISTER_ABS_2D_BENCHMARK(ComplexFloat, cytnx::Type.ComplexFloat)
  REGISTER_ABS_2D_BENCHMARK(ComplexDouble, cytnx::Type.ComplexDouble)

  // 2D Abs_ benchmarks for all types
  REGISTER_ABS_INPLACE_2D_BENCHMARK(Int16, cytnx::Type.Int16)
  REGISTER_ABS_INPLACE_2D_BENCHMARK(Int32, cytnx::Type.Int32)
  REGISTER_ABS_INPLACE_2D_BENCHMARK(Int64, cytnx::Type.Int64)
  REGISTER_ABS_INPLACE_2D_BENCHMARK(Uint16, cytnx::Type.Uint16)
  REGISTER_ABS_INPLACE_2D_BENCHMARK(Uint32, cytnx::Type.Uint32)
  REGISTER_ABS_INPLACE_2D_BENCHMARK(Uint64, cytnx::Type.Uint64)
  REGISTER_ABS_INPLACE_2D_BENCHMARK(Float, cytnx::Type.Float)
  REGISTER_ABS_INPLACE_2D_BENCHMARK(Double, cytnx::Type.Double)
  REGISTER_ABS_INPLACE_2D_BENCHMARK(ComplexFloat, cytnx::Type.ComplexFloat)
  REGISTER_ABS_INPLACE_2D_BENCHMARK(ComplexDouble, cytnx::Type.ComplexDouble)

  // 1D Abs benchmarks for selected types
  REGISTER_ABS_1D_BENCHMARK(Int16, cytnx::Type.Int16)
  REGISTER_ABS_1D_BENCHMARK(Int32, cytnx::Type.Int32)
  REGISTER_ABS_1D_BENCHMARK(Int64, cytnx::Type.Int64)
  REGISTER_ABS_1D_BENCHMARK(Uint16, cytnx::Type.Uint16)
  REGISTER_ABS_1D_BENCHMARK(Uint32, cytnx::Type.Uint32)
  REGISTER_ABS_1D_BENCHMARK(Uint64, cytnx::Type.Uint64)
  REGISTER_ABS_1D_BENCHMARK(Float, cytnx::Type.Float)
  REGISTER_ABS_1D_BENCHMARK(Double, cytnx::Type.Double)
  REGISTER_ABS_1D_BENCHMARK(ComplexFloat, cytnx::Type.ComplexFloat)
  REGISTER_ABS_1D_BENCHMARK(ComplexDouble, cytnx::Type.ComplexDouble)

#ifdef UNI_GPU

  // GPU 2D Abs benchmarks
  REGISTER_GPU_ABS_2D_BENCHMARK(Int16, cytnx::Type.Int16)
  REGISTER_GPU_ABS_2D_BENCHMARK(Int32, cytnx::Type.Int32)
  REGISTER_GPU_ABS_2D_BENCHMARK(Int64, cytnx::Type.Int64)
  REGISTER_GPU_ABS_2D_BENCHMARK(Uint16, cytnx::Type.Uint16)
  REGISTER_GPU_ABS_2D_BENCHMARK(Uint32, cytnx::Type.Uint32)
  REGISTER_GPU_ABS_2D_BENCHMARK(Uint64, cytnx::Type.Uint64)
  REGISTER_GPU_ABS_2D_BENCHMARK(Float, cytnx::Type.Float)
  REGISTER_GPU_ABS_2D_BENCHMARK(Double, cytnx::Type.Double)
  REGISTER_GPU_ABS_2D_BENCHMARK(ComplexFloat, cytnx::Type.ComplexFloat)
  REGISTER_GPU_ABS_2D_BENCHMARK(ComplexDouble, cytnx::Type.ComplexDouble)

  // GPU 2D Abs_ benchmarks
  REGISTER_GPU_ABS_INPLACE_2D_BENCHMARK(Int16, cytnx::Type.Int16)
  REGISTER_GPU_ABS_INPLACE_2D_BENCHMARK(Int32, cytnx::Type.Int32)
  REGISTER_GPU_ABS_INPLACE_2D_BENCHMARK(Int64, cytnx::Type.Int64)
  REGISTER_GPU_ABS_INPLACE_2D_BENCHMARK(Uint16, cytnx::Type.Uint16)
  REGISTER_GPU_ABS_INPLACE_2D_BENCHMARK(Uint32, cytnx::Type.Uint32)
  REGISTER_GPU_ABS_INPLACE_2D_BENCHMARK(Uint64, cytnx::Type.Uint64)
  REGISTER_GPU_ABS_INPLACE_2D_BENCHMARK(Float, cytnx::Type.Float)
  REGISTER_GPU_ABS_INPLACE_2D_BENCHMARK(Double, cytnx::Type.Double)
  REGISTER_GPU_ABS_INPLACE_2D_BENCHMARK(ComplexFloat, cytnx::Type.ComplexFloat)
  REGISTER_GPU_ABS_INPLACE_2D_BENCHMARK(ComplexDouble, cytnx::Type.ComplexDouble)

  // GPU 1D Abs benchmarks
  REGISTER_GPU_ABS_1D_BENCHMARK(Int16, cytnx::Type.Int16)
  REGISTER_GPU_ABS_1D_BENCHMARK(Int32, cytnx::Type.Int32)
  REGISTER_GPU_ABS_1D_BENCHMARK(Int64, cytnx::Type.Int64)
  REGISTER_GPU_ABS_1D_BENCHMARK(Uint16, cytnx::Type.Uint16)
  REGISTER_GPU_ABS_1D_BENCHMARK(Uint32, cytnx::Type.Uint32)
  REGISTER_GPU_ABS_1D_BENCHMARK(Uint64, cytnx::Type.Uint64)
  REGISTER_GPU_ABS_1D_BENCHMARK(Float, cytnx::Type.Float)
  REGISTER_GPU_ABS_1D_BENCHMARK(Double, cytnx::Type.Double)
  REGISTER_GPU_ABS_1D_BENCHMARK(ComplexFloat, cytnx::Type.ComplexFloat)
  REGISTER_GPU_ABS_1D_BENCHMARK(ComplexDouble, cytnx::Type.ComplexDouble)

#endif  // UNI_GPU

}  // namespace BMTest_Abs
