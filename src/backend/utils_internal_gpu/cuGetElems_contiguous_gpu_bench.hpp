#ifndef CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUGETELEMS_CONTIGUOUS_BENCH_HPP_
#define CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUGETELEMS_CONTIGUOUS_BENCH_HPP_

#include <vector>
#include "Type.hpp"

namespace cytnx {
  namespace utils_internal {

    // Experimental benchmark-only implementation, deliberately not wired into Tensor::get.
    // Strategy 0 forwards to the unmodified PR #1230 Double backend. 1..3 select
    // alternative kernels with the same per-call metadata setup; 4 is an adaptive
    // choice controlled by CYTNX_BENCH_GPU_GET_DISPATCH (four digits 0..3).
    void cuGetElems_contiguous_gpu_d_bench(
      void* out, void* in, const std::vector<cytnx_uint64>& offj,
      const std::vector<cytnx_uint64>& new_offj,
      const std::vector<std::vector<cytnx_uint64>>& locators,
      const cytnx_uint64& TotalElem, const cytnx_uint64& Nelem_grp,
      int strategy);

    int cuGetElems_contiguous_gpu_d_adaptive_choice(cytnx_uint64 TotalElem,
                                                     cytnx_uint64 Nelem_grp);
  }  // namespace utils_internal
}  // namespace cytnx

#endif
