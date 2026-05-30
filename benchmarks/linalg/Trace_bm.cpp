#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cytnx.hpp"

// Benchmarks for linalg::Trace.
//
// Two shapes are exercised:
//
//   * 3D path: a contiguous {n, middle, n} tensor traced over axes 0 and 2.
//     The two traced axes are not adjacent in storage, so the diagonal stride
//     (strides[0] + strides[2] = middle * n + 1) is large; each iteration is a
//     genuine higher-rank trace rather than a simple 2D matrix trace.
//   * 2D path: a contiguous {n, n} tensor traced over axes 0 and 1, exercising
//     the _trace_2d / cuTrace_2d_kernel branch.
//
// For each shape, the in-place strided reduction is compared against a
// "collect the traced elements contiguously and reduce them" baseline that
// avoids reading the diagonal with a large stride:
//
//   * 3D matvec baseline: tr(A)[m] = <vec(I_n), vec(A[:, m, :])>, so stacking
//     the middle outputs becomes a {middle, n*n} @ {n*n, 1} GEMM call against
//     vec(I_n) -- a BLAS matrix-vector multiplication.
//   * 2D vecdot baseline: tr(A) = <vec(I_n), vec(A)> -- a BLAS vector-vector
//     dot product on the n*n-element flattenings of I and A.
//   * 2D reshape trick: drop the last diagonal entry A[n-1, n-1], view the
//     remaining n*n - 1 entries as {n-1, n+1}, permute + contiguous so the
//     first column (which holds diag[0..n-2]) becomes a contiguous row, then
//     reduce that row over the raw buffer and add the saved last entry.
//
// Every variant runs once before the timing loop and is checked against
// linalg::Trace on the same input, so a wrong baseline would fail loudly
// instead of producing fast-but-meaningless numbers.

namespace BMTest_Trace {

  // Maps a benchmarked dtype enum to its C++ storage type, so a baseline can
  // read the raw buffer directly. Only the dtypes the 2D reshape trick is
  // registered for need an entry.
  template <unsigned int dtype>
  struct BmCType;
  template <>
  struct BmCType<cytnx::Type.Double> {
    using type = cytnx::cytnx_double;
  };
  template <>
  struct BmCType<cytnx::Type.ComplexDouble> {
    using type = cytnx::cytnx_complex128;
  };

  // Aborts the benchmark if `candidate` differs from `reference` by more than
  // a small absolute tolerance. Both tensors must have the same dtype, and
  // (after broadcasting via Tensor's arithmetic) be element-wise comparable.
  static void VerifyAgainstTrace(const cytnx::Tensor& reference, const cytnx::Tensor& candidate,
                                 const char* variant_name) {
    auto diff = (candidate - reference).Abs();
    const double max_err = diff.Max().item<cytnx::cytnx_double>();
    if (!(max_err < 1e-6)) {
      std::cerr << "[Trace benchmark] variant \"" << variant_name
                << "\" disagrees with linalg::Trace; max abs error = " << max_err << "\n";
      std::abort();
    }
  }

  template <unsigned int dtype, cytnx::cytnx_int32 device = cytnx::Device.cpu>
  static void BM_Trace_Strided_3D_Template(benchmark::State& state) {
    const auto n = state.range(0);
    const auto middle = state.range(1);
    const auto tensor = cytnx::random::random_tensor({n, middle, n}, -1.0, 1.0, device, 0, dtype);
    const auto reference = cytnx::linalg::Trace(tensor, 0, 2);
    VerifyAgainstTrace(reference, cytnx::linalg::Trace(tensor, 0, 2), "Strided_3D");
    for (auto _ : state) {
      auto result = cytnx::linalg::Trace(tensor, 0, 2);
      benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n * n * middle);
  }

  // ND matvec baseline: tr(A)[m] = <vec(I_n), vec(A[:, m, :])>. Stacking the
  // middle outputs becomes a {middle, n*n} @ {n*n, 1} GEMM call against vec(I_n).
  template <unsigned int dtype, cytnx::cytnx_int32 device = cytnx::Device.cpu>
  static void BM_Trace_Matvec_3D_Template(benchmark::State& state) {
    const auto n = state.range(0);
    const auto middle = state.range(1);
    const auto tensor = cytnx::random::random_tensor({n, middle, n}, -1.0, 1.0, device, 0, dtype);
    const auto vec_I = cytnx::eye(n, dtype, device).reshape({n * n, 1});
    const auto reference = cytnx::linalg::Trace(tensor, 0, 2);
    auto compute = [&]() {
      auto packed = tensor.permute({1, 0, 2}).contiguous().reshape({middle, n * n});
      return cytnx::linalg::Matmul(packed, vec_I).reshape({middle});
    };
    VerifyAgainstTrace(reference, compute(), "Matvec_3D");
    for (auto _ : state) {
      auto result = compute();
      benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n * n * middle);
  }

  template <unsigned int dtype, cytnx::cytnx_int32 device = cytnx::Device.cpu>
  static void BM_Trace_Strided_2D_Template(benchmark::State& state) {
    const auto n = state.range(0);
    const auto tensor = cytnx::random::random_tensor({n, n}, -1.0, 1.0, device, 0, dtype);
    const auto reference = cytnx::linalg::Trace(tensor, 0, 1);
    VerifyAgainstTrace(reference, cytnx::linalg::Trace(tensor, 0, 1), "Strided_2D");
    for (auto _ : state) {
      auto result = cytnx::linalg::Trace(tensor, 0, 1);
      benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n * n);
  }

  // 2D vector-dot baseline: tr(A) = <vec(I_n), vec(A)> as a BLAS dot product
  // on the n*n-element flattenings.
  template <unsigned int dtype, cytnx::cytnx_int32 device = cytnx::Device.cpu>
  static void BM_Trace_Vecdot_2D_Template(benchmark::State& state) {
    const auto n = state.range(0);
    const auto tensor = cytnx::random::random_tensor({n, n}, -1.0, 1.0, device, 0, dtype);
    const auto vec_I = cytnx::eye(n, dtype, device).reshape({n * n});
    const auto reference = cytnx::linalg::Trace(tensor, 0, 1);
    auto compute = [&]() {
      auto vec_A = tensor.reshape({n * n});
      return cytnx::linalg::Vectordot(vec_I, vec_A).reshape({1});
    };
    VerifyAgainstTrace(reference, compute(), "Vecdot_2D");
    for (auto _ : state) {
      auto result = compute();
      benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n * n);
  }

  // 2D reshape trick: drop A[n-1, n-1], view the remaining n*n - 1 elements as
  // {n-1, n+1}, permute + contiguous so the first column (which holds
  // diag[0..n-2]) becomes a contiguous row, reduce that row, and add the saved
  // last element.
  //
  // To keep the bulk data movement honest, the {n-1, n+1} view reuses the
  // source storage directly: shrinking its size to (n-1)*(n+1) is a no-realloc
  // metadata change (Storage::resize only reallocates when growing past the
  // capacity), so the permute -> contiguous gather is the only copy the trick
  // performs. The size is restored to n*n afterwards (and the dropped corner
  // written back, since a grow-resize zero-fills any tail it has to reallocate)
  // so the source tensor stays intact across iterations.
  //
  // The row reduction is device-specific: on CPU the contiguous row is summed
  // over the raw buffer (no linalg::Sum / Accessor allocations on top of the
  // gather); on GPU the raw buffer lives in device memory and cannot be read
  // host-side, so the row is wrapped as a {n-1} tensor over the resized gather
  // storage and reduced with linalg::Sum, and the corner is taken as a {1}
  // tensor so the final add stays on the device.
  template <unsigned int dtype, cytnx::cytnx_int32 device = cytnx::Device.cpu>
  static void BM_Trace_Reshape_2D_Template(benchmark::State& state) {
    using T = typename BmCType<dtype>::type;
    const auto n = state.range(0);
    const auto tensor = cytnx::random::random_tensor({n, n}, -1.0, 1.0, device, 0, dtype);
    const auto reference = cytnx::linalg::Trace(tensor, 0, 1);
    auto compute = [&]() {
      auto& storage = tensor.storage();
      if constexpr (device == cytnx::Device_class::cpu) {
        const T last = tensor.at<T>(
          {static_cast<cytnx::cytnx_uint64>(n - 1), static_cast<cytnx::cytnx_uint64>(n - 1)});
        storage.resize((n - 1) * (n + 1));
        auto view = cytnx::Tensor::from_storage(storage);
        view.reshape_({n - 1, n + 1});
        auto packed = view.permute({1, 0}).contiguous();
        storage.resize(n * n);  // restore the source storage size
        storage.at<T>((n * n) - 1) = last;  // and recover the dropped corner

        const T* row = packed.storage().data<T>();
        T sum = T(0);
        for (cytnx::cytnx_int64 k = 0; k < n - 1; ++k) sum += row[k];
        sum += last;

        auto out = cytnx::Tensor({static_cast<cytnx::cytnx_uint64>(1)}, dtype, device);
        out.storage().at<T>(0) = sum;
        return out;
      } else {
        // Read the corner as a {1} device tensor before the storage shrinks.
        auto last = tensor.reshape({n * n}).get({cytnx::Accessor((n * n) - 1)});
        storage.resize((n - 1) * (n + 1));
        auto view = cytnx::Tensor::from_storage(storage);
        view.reshape_({n - 1, n + 1});
        auto packed = view.permute({1, 0}).contiguous();
        storage.resize(n * n);  // restore the source storage size

        auto row_storage = packed.storage();
        row_storage.resize(n - 1);  // first contiguous row = diag[0..n-2]
        auto row = cytnx::Tensor::from_storage(row_storage);
        return (cytnx::linalg::Sum(row) + last).reshape({1});
      }
    };
    VerifyAgainstTrace(reference, compute(), "Reshape_2D");
    for (auto _ : state) {
      auto result = compute();
      benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n * n);
  }

#define REGISTER_TRACE_3D_BENCHMARK(TypeName, TypeEnum)      \
  BENCHMARK_TEMPLATE(BM_Trace_Strided_3D_Template, TypeEnum) \
    ->Name("BM_Trace_Strided_3D_" #TypeName)                 \
    ->Args({64, 64})                                         \
    ->Args({256, 64})                                        \
    ->Args({1024, 16})                                       \
    ->Args({2048, 16})                                       \
    ->Args({4096, 8})                                        \
    ->Unit(benchmark::kMicrosecond);                         \
  BENCHMARK_TEMPLATE(BM_Trace_Matvec_3D_Template, TypeEnum)  \
    ->Name("BM_Trace_Matvec_3D_" #TypeName)                  \
    ->Args({64, 64})                                         \
    ->Args({256, 64})                                        \
    ->Args({1024, 16})                                       \
    ->Args({2048, 16})                                       \
    ->Args({4096, 8})                                        \
    ->Unit(benchmark::kMicrosecond);

#define REGISTER_TRACE_2D_BENCHMARK(TypeName, TypeEnum)      \
  BENCHMARK_TEMPLATE(BM_Trace_Strided_2D_Template, TypeEnum) \
    ->Name("BM_Trace_Strided_2D_" #TypeName)                 \
    ->Args({64})                                             \
    ->Args({256})                                            \
    ->Args({1024})                                           \
    ->Args({4096})                                           \
    ->Args({8192})                                           \
    ->Unit(benchmark::kMicrosecond);                         \
  BENCHMARK_TEMPLATE(BM_Trace_Vecdot_2D_Template, TypeEnum)  \
    ->Name("BM_Trace_Vecdot_2D_" #TypeName)                  \
    ->Args({64})                                             \
    ->Args({256})                                            \
    ->Args({1024})                                           \
    ->Args({4096})                                           \
    ->Args({8192})                                           \
    ->Unit(benchmark::kMicrosecond);                         \
  BENCHMARK_TEMPLATE(BM_Trace_Reshape_2D_Template, TypeEnum) \
    ->Name("BM_Trace_Reshape_2D_" #TypeName)                 \
    ->Args({64})                                             \
    ->Args({256})                                            \
    ->Args({1024})                                           \
    ->Args({4096})                                           \
    ->Args({8192})                                           \
    ->Unit(benchmark::kMicrosecond);

  REGISTER_TRACE_3D_BENCHMARK(Double, cytnx::Type.Double)
  REGISTER_TRACE_3D_BENCHMARK(ComplexDouble, cytnx::Type.ComplexDouble)
  REGISTER_TRACE_2D_BENCHMARK(Double, cytnx::Type.Double)
  REGISTER_TRACE_2D_BENCHMARK(ComplexDouble, cytnx::Type.ComplexDouble)

#ifdef UNI_GPU
  #define REGISTER_GPU_TRACE_3D_BENCHMARK(TypeName, TypeEnum)                      \
    BENCHMARK_TEMPLATE(BM_Trace_Strided_3D_Template, TypeEnum, cytnx::Device.cuda) \
      ->Name("BM_gpu_Trace_Strided_3D_" #TypeName)                                 \
      ->Args({256, 64})                                                            \
      ->Args({1024, 16})                                                           \
      ->Args({2048, 16})                                                           \
      ->Args({4096, 8})                                                            \
      ->Unit(benchmark::kMicrosecond);                                             \
    BENCHMARK_TEMPLATE(BM_Trace_Matvec_3D_Template, TypeEnum, cytnx::Device.cuda)  \
      ->Name("BM_gpu_Trace_Matvec_3D_" #TypeName)                                  \
      ->Args({256, 64})                                                            \
      ->Args({1024, 16})                                                           \
      ->Args({2048, 16})                                                           \
      ->Args({4096, 8})                                                            \
      ->Unit(benchmark::kMicrosecond);

  #define REGISTER_GPU_TRACE_2D_BENCHMARK(TypeName, TypeEnum)                      \
    BENCHMARK_TEMPLATE(BM_Trace_Strided_2D_Template, TypeEnum, cytnx::Device.cuda) \
      ->Name("BM_gpu_Trace_Strided_2D_" #TypeName)                                 \
      ->Args({256})                                                                \
      ->Args({1024})                                                               \
      ->Args({4096})                                                               \
      ->Args({8192})                                                               \
      ->Unit(benchmark::kMicrosecond);                                             \
    BENCHMARK_TEMPLATE(BM_Trace_Vecdot_2D_Template, TypeEnum, cytnx::Device.cuda)  \
      ->Name("BM_gpu_Trace_Vecdot_2D_" #TypeName)                                  \
      ->Args({256})                                                                \
      ->Args({1024})                                                               \
      ->Args({4096})                                                               \
      ->Args({8192})                                                               \
      ->Unit(benchmark::kMicrosecond);                                             \
    BENCHMARK_TEMPLATE(BM_Trace_Reshape_2D_Template, TypeEnum, cytnx::Device.cuda) \
      ->Name("BM_gpu_Trace_Reshape_2D_" #TypeName)                                 \
      ->Args({256})                                                                \
      ->Args({1024})                                                               \
      ->Args({4096})                                                               \
      ->Args({8192})                                                               \
      ->Unit(benchmark::kMicrosecond);

  REGISTER_GPU_TRACE_3D_BENCHMARK(Double, cytnx::Type.Double)
  REGISTER_GPU_TRACE_3D_BENCHMARK(ComplexDouble, cytnx::Type.ComplexDouble)
  REGISTER_GPU_TRACE_2D_BENCHMARK(Double, cytnx::Type.Double)
  REGISTER_GPU_TRACE_2D_BENCHMARK(ComplexDouble, cytnx::Type.ComplexDouble)
#endif  // UNI_GPU

}  // namespace BMTest_Trace
