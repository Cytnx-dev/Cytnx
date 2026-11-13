// Experimental Float64 get kernels. Tensor::get and the existing S0 CUDA
// implementation are unchanged; each candidate pays the same four device
// metadata allocations and host-to-device copies as PR #1230.
#include "backend/utils_internal_gpu/cuGetElems_contiguous_gpu_bench.hpp"
#include "backend/utils_internal_gpu/cuGetElems_contiguous_gpu.hpp"
#include "backend/utils_internal_gpu/cuScopedResource_gpu.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace cytnx {
namespace utils_internal {
namespace {

__device__ __forceinline__ cytnx_uint64 GetSourceUnit(
    cytnx_uint64 n, const cytnx_uint64* offj,
    const cytnx_uint64* new_offj, const cytnx_uint64* locators,
    const cytnx_uint64* picksize, cytnx_uint64 rank) {
  cytnx_uint64 loc = 0;
  cytnx_uint64 offset = 0;
  for (cytnx_uint64 r = 0; r < rank; ++r) {
    const cytnx_uint64 index = n / new_offj[r];
    loc += (picksize[r] ? locators[offset + index] : index) * offj[r];
    n %= new_offj[r];
    offset += picksize[r];
  }
  return loc;
}

// S1: one thread per element (with repeated locator calculations).
__global__ void GetFlat(double* out, const double* in,
                        const cytnx_uint64* offj, const cytnx_uint64* new_offj,
                        const cytnx_uint64* locators, const cytnx_uint64* picksize,
                        cytnx_uint64 rank, cytnx_uint64 units, cytnx_uint64 nunit) {
  const cytnx_uint64 tid =
      static_cast<cytnx_uint64>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= units * nunit) return;
  const cytnx_uint64 n = tid / nunit;
  const cytnx_uint64 x = tid % nunit;
  const cytnx_uint64 loc = GetSourceUnit(n, offj, new_offj, locators, picksize, rank);
  out[tid] = in[loc * nunit + x];
}

// S2: one warp per unit, leader computes the location, then broadcasts it.
__global__ void GetWarp(double* out, const double* in,
                        const cytnx_uint64* offj, const cytnx_uint64* new_offj,
                        const cytnx_uint64* locators, const cytnx_uint64* picksize,
                        cytnx_uint64 rank, cytnx_uint64 units, cytnx_uint64 nunit) {
  const cytnx_uint64 tid =
      static_cast<cytnx_uint64>(blockIdx.x) * blockDim.x + threadIdx.x;
  const cytnx_uint64 n = tid >> 5;
  if (n >= units) return;  // All lanes of the warp branch together.
  const unsigned int lane = static_cast<unsigned int>(tid & 31);
  cytnx_uint64 loc = 0;
  if (lane == 0) loc = GetSourceUnit(n, offj, new_offj, locators, picksize, rank);
  loc = __shfl_sync(0xffffffffu, loc, 0);
  for (cytnx_uint64 x = lane; x < nunit; x += 32)
    out[n * nunit + x] = in[loc * nunit + x];
}

// S3: 32 x 8 thread tile; tile index flattened into grid.x to avoid 65535
// grid.y limit. Multiple blocks can copy different 32-element tiles of a unit.
__global__ void GetTile(double* out, const double* in,
                        const cytnx_uint64* offj, const cytnx_uint64* new_offj,
                        const cytnx_uint64* locators, const cytnx_uint64* picksize,
                        cytnx_uint64 rank, cytnx_uint64 units, cytnx_uint64 nunit,
                        cytnx_uint64 tiles_per_unit) {
  const cytnx_uint64 group = blockIdx.x / tiles_per_unit;
  const cytnx_uint64 tile = blockIdx.x % tiles_per_unit;
  const cytnx_uint64 n = group * blockDim.y + threadIdx.y;
  const cytnx_uint64 x = tile * blockDim.x + threadIdx.x;
  if (n >= units || x >= nunit) return;
  const cytnx_uint64 loc = GetSourceUnit(n, offj, new_offj, locators, picksize, rank);
  out[n * nunit + x] = in[loc * nunit + x];
}

int DispatchByUnitSize(cytnx_uint64 nunit) {
  // Pilot-tunable policy. Boundaries 4/32/1024 are explicit so the Notebook
  // can measure a stratified pilot and pick the fastest per bucket.
  // Defaults are hypotheses only, NOT claims of a measured optimum.
  static const char* const policy = []() {
    const char* str = std::getenv("CYTNX_BENCH_GPU_GET_DISPATCH");
    if (!str || std::strlen(str) != 4) return "0223";
    for (unsigned int i = 0; i < 4; ++i)
      if (str[i] < '0' || str[i] > '3') return "0223";
    return str;
  }();
  const int bucket = nunit <= 4 ? 0 : (nunit <= 32 ? 1 : (nunit <= 1024 ? 2 : 3));
  return policy[bucket] - '0';
}

}  // namespace

int cuGetElems_contiguous_gpu_d_adaptive_choice(cytnx_uint64 TotalElem,
                                                 cytnx_uint64 Nelem_grp) {
  (void)TotalElem;
  return DispatchByUnitSize(Nelem_grp);
}

void cuGetElems_contiguous_gpu_d_bench(
    void* out, void* in, const std::vector<cytnx_uint64>& offj,
    const std::vector<cytnx_uint64>& new_offj,
    const std::vector<std::vector<cytnx_uint64>>& locators,
    const cytnx_uint64& TotalElem, const cytnx_uint64& Nelem_grp,
    int strategy) {
  if (strategy == 4) strategy = DispatchByUnitSize(Nelem_grp);
  if (strategy == 0) {
    // Call the ORIGINAL PR #1230 backend verbatim, including its metadata and
    // one-thread-per-unit kernel. This preserves a faithful S0 baseline.
    cuGetElems_contiguous_gpu_d(out, in, offj, new_offj, locators, TotalElem, Nelem_grp);
    return;
  }
  if (strategy < 1 || strategy > 3 || TotalElem == 0 || Nelem_grp == 0 ||
      offj.empty() || offj.size() != new_offj.size() || offj.size() != locators.size())
    CYTNX_ERROR(true, "[gpu-get-bench] invalid strategy or metadata", );

  // This is intentionally the same metadata path as S0, including cudaMalloc,
  // H2D transfer and device-buffer teardown on EVERY invocation.
  DeviceBuffer<cytnx_uint64> d_offj(offj.size());
  checkCudaErrors(cudaMemcpy(d_offj.get(), offj.data(),
                             offj.size() * sizeof(cytnx_uint64), cudaMemcpyHostToDevice));
  DeviceBuffer<cytnx_uint64> d_new_offj(new_offj.size());
  checkCudaErrors(cudaMemcpy(d_new_offj.get(), new_offj.data(),
                             new_offj.size() * sizeof(cytnx_uint64), cudaMemcpyHostToDevice));

  std::vector<cytnx_uint64> composite, picksize;
  cytnx_uint64 count = 0;
  for (const auto& loc : locators) {
    picksize.push_back(loc.size());
    count += loc.size();
  }
  composite.resize(count);
  count = 0;
  for (const auto& loc : locators) {
    if (!loc.empty()) std::memcpy(composite.data() + count, loc.data(),
                                  loc.size() * sizeof(cytnx_uint64));
    count += loc.size();
  }
  DeviceBuffer<cytnx_uint64> d_locators(composite.size());
  if (!composite.empty())
    checkCudaErrors(cudaMemcpy(d_locators.get(), composite.data(),
                               composite.size() * sizeof(cytnx_uint64),
                               cudaMemcpyHostToDevice));
  DeviceBuffer<cytnx_uint64> d_picksize(picksize.size());
  checkCudaErrors(cudaMemcpy(d_picksize.get(), picksize.data(),
                             picksize.size() * sizeof(cytnx_uint64), cudaMemcpyHostToDevice));

  double* const dest = static_cast<double*>(out);
  const double* const src = static_cast<const double*>(in);
  const cytnx_uint64 rank = offj.size();
  const cytnx_uint64* const a = d_offj.get();
  const cytnx_uint64* const b = d_new_offj.get();
  const cytnx_uint64* const c = d_locators.get();
  const cytnx_uint64* const d = d_picksize.get();
  if (strategy == 1) {
    const cytnx_uint64 threads = TotalElem * Nelem_grp;
    GetFlat<<<(threads + 255) / 256, 256>>>(
        dest, src, a, b, c, d, rank, TotalElem, Nelem_grp);
    CYTNX_CHECK_CUDA_LAUNCH(GetFlat);
  } else if (strategy == 2) {
    const cytnx_uint64 threads = TotalElem * 32;
    GetWarp<<<(threads + 255) / 256, 256>>>(
        dest, src, a, b, c, d, rank, TotalElem, Nelem_grp);
    CYTNX_CHECK_CUDA_LAUNCH(GetWarp);
  } else {
    const cytnx_uint64 tiles = (Nelem_grp + 31) / 32;
    const cytnx_uint64 blocks = ((TotalElem + 7) / 8) * tiles;
    GetTile<<<blocks, dim3(32, 8)>>>(
        dest, src, a, b, c, d, rank, TotalElem, Nelem_grp, tiles);
    CYTNX_CHECK_CUDA_LAUNCH(GetTile);
  }
}

}  // namespace utils_internal
}  // namespace cytnx
