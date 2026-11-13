// L2 Float64 get benchmark: includes host metadata allocation/copy, kernel launch,
// per-call buffer destruction and a completed CUDA operation; excludes fixture setup.
#include <benchmark/benchmark.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifdef UNI_GPU
#include <cuda_runtime.h>
#include "backend/utils_internal_gpu/cuGetElems_contiguous_gpu.hpp"
#include "backend/utils_internal_gpu/cuGetElems_contiguous_gpu_bench.hpp"

namespace {
using cytnx::cytnx_uint64;
using cytnx::utils_internal::cuGetElems_contiguous_gpu_d;
using cytnx::utils_internal::cuGetElems_contiguous_gpu_d_bench;

struct Case {
  const char* family;
  cytnx_uint64 units;
  cytnx_uint64 nunit;
  const char* pattern;
};

bool IsOk(cudaError_t status, benchmark::State& state, const char* where) {
  if (status == cudaSuccess) return true;
  const std::string message = std::string(where) + ": " + cudaGetErrorString(status);
  state.SkipWithError(message.c_str());
  return false;
}

void Invoke(int strategy, double* out, double* in,
            const std::vector<cytnx_uint64>& offj,
            const std::vector<cytnx_uint64>& new_offj,
            const std::vector<std::vector<cytnx_uint64>>& locators,
            cytnx_uint64 total, cytnx_uint64 nunit) {
  // S0 is the exact original PR #1230 entrypoint, not a surrogate kernel.
  if (strategy == 0)
    cuGetElems_contiguous_gpu_d(out, in, offj, new_offj, locators, total, nunit);
  else
    cuGetElems_contiguous_gpu_d_bench(out, in, offj, new_offj, locators,
                                     total, nunit, strategy);
}

void BM_GetL2(benchmark::State& state, Case c, int strategy) {
  state.PauseTiming();
  const cytnx_uint64 source_units =
      std::string(c.pattern) == "Strided" ? c.units * 2 : c.units;
  const std::size_t source_count = static_cast<std::size_t>(source_units * c.nunit);
  const std::size_t output_count = static_cast<std::size_t>(c.units * c.nunit);
  std::vector<double> host_input(source_count), host_expected(output_count),
                      host_output(output_count);
  for (std::size_t i = 0; i < source_count; ++i)
    host_input[i] = static_cast<double>(i) * 0.125 - 7.25;
  std::vector<cytnx_uint64> picked(c.units);
  std::iota(picked.begin(), picked.end(), 0);
  if (std::strcmp(c.pattern, "Strided") == 0) {
    for (cytnx_uint64 i = 0; i < c.units; ++i) picked[i] *= 2;
  } else if (std::strcmp(c.pattern, "Reversed") == 0) {
    std::reverse(picked.begin(), picked.end());
  } else if (std::strcmp(c.pattern, "Random") == 0) {
    std::mt19937_64 generator(1230);
    std::shuffle(picked.begin(), picked.end(), generator);
  }
  for (cytnx_uint64 n = 0; n < c.units; ++n) {
    if (picked[n] >= source_units) {
      state.SkipWithError("invalid source locator"); return;
    }
    for (cytnx_uint64 x = 0; x < c.nunit; ++x)
      host_expected[n * c.nunit + x] = host_input[picked[n] * c.nunit + x];
  }
  const std::vector<cytnx_uint64> offj{1}, new_offj{1};
  const std::vector<std::vector<cytnx_uint64>> locators{picked};
  double* input = nullptr;
  double* output = nullptr;
  if (!IsOk(cudaMalloc(reinterpret_cast<void**>(&input), source_count * sizeof(double)),
            state, "cudaMalloc(input)")) return;
  if (!IsOk(cudaMalloc(reinterpret_cast<void**>(&output), output_count * sizeof(double)),
            state, "cudaMalloc(output)")) {
    cudaFree(input); return;
  }
  const auto cleanup = [&]() {
    cudaFree(output);
    cudaFree(input);
  };
  if (!IsOk(cudaMemcpy(input, host_input.data(), source_count * sizeof(double),
                       cudaMemcpyHostToDevice), state, "input H2D")) {
    cleanup(); return;
  }
  if (!IsOk(cudaMemset(output, 0xa5, output_count * sizeof(double)),
            state, "output sentinel")) {
    cleanup(); return;
  }
  // Pre-timing correctness gate, against an independently assembled CPU reference.
  // Check every element: do not report a fast but incorrect strategy.
  Invoke(strategy, output, input, offj, new_offj, locators, c.units, c.nunit);
  if (!IsOk(cudaDeviceSynchronize(), state, "correctness sync") ||
      !IsOk(cudaMemcpy(host_output.data(), output,
                       output_count * sizeof(double), cudaMemcpyDeviceToHost),
            state, "correctness D2H")) {
    cleanup(); return;
  }
  if (std::memcmp(host_output.data(), host_expected.data(),
                  output_count * sizeof(double)) != 0) {
    state.SkipWithError("correctness failure: output differs from CPU reference");
    cleanup(); return;
  }
  // Warm up the same full backend path before measuring.
  for (int i = 0; i < 3; ++i)
    Invoke(strategy, output, input, offj, new_offj, locators, c.units, c.nunit);
  if (!IsOk(cudaDeviceSynchronize(), state, "warm-up sync")) {
    cleanup(); return;
  }

  state.ResumeTiming();
  for (auto _ : state) {
    Invoke(strategy, output, input, offj, new_offj, locators, c.units, c.nunit);
    const auto status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
      const std::string message = std::string("timed sync: ") + cudaGetErrorString(status);
      state.SkipWithError(message.c_str());
      break;
    }
    benchmark::DoNotOptimize(output);
  }
  state.PauseTiming();
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(output_count) * sizeof(double));
  cleanup();
}

std::vector<Case> MakeCases() {
  std::vector<Case> c = {
    // A1: fixed 262144 elements.
    {"A262144", 1, 262144, "Identity"},
    {"A262144", 32, 8192, "Identity"},
    {"A262144", 256, 1024, "Identity"},
    {"A262144", 4096, 64, "Identity"},
    {"A262144", 65536, 4, "Identity"},
    {"A262144", 262144, 1, "Identity"},
    // A2: four times the number of units at the same per-unit length.
    {"A1048576", 4, 262144, "Identity"},
    {"A1048576", 128, 8192, "Identity"},
    {"A1048576", 1024, 1024, "Identity"},
    {"A1048576", 16384, 64, "Identity"},
    {"A1048576", 262144, 4, "Identity"},
    {"A1048576", 1048576, 1, "Identity"},
    // B: vary the number of units at fixed unit lengths.
    {"B", 256, 1, "Identity"},
    {"B", 2048, 1, "Identity"},
    {"B", 32768, 1, "Identity"},
    {"B", 256, 32, "Identity"},
    {"B", 2048, 32, "Identity"},
    {"B", 32768, 32, "Identity"},
    {"B", 1, 1024, "Identity"},
    {"B", 32, 1024, "Identity"},
    {"B", 256, 1024, "Identity"},
    {"B", 1024, 1024, "Identity"},
    {"B", 1, 65536, "Identity"},
    {"B", 4, 65536, "Identity"},
    {"B", 16, 65536, "Identity"},
    // C: all algorithms must use exactly the same source-index mapping.
    {"C", 4096, 64, "Identity"},
    {"C", 4096, 64, "Strided"},
    {"C", 4096, 64, "Reversed"},
    {"C", 4096, 64, "Random"},
  };
  return c;
}

bool RegisterAll() {
  for (const auto c : MakeCases()) {
    for (int strategy = 0; strategy <= 4; ++strategy) {
      const auto name = std::string("GPUGetL2/") + c.family +
                        "/T" + std::to_string(c.units) +
                        "/U" + std::to_string(c.nunit) +
                        "/" + c.pattern + "/S" + std::to_string(strategy);
      benchmark::RegisterBenchmark(name.c_str(), [=](benchmark::State& state) {
        BM_GetL2(state, c, strategy);
      })->UseRealTime()->Unit(benchmark::kMicrosecond)->Repetitions(5)->MinTime(0.06);
    }
  }
  return true;
}
const bool registered = RegisterAll();
}  // namespace
#endif  // UNI_GPU
