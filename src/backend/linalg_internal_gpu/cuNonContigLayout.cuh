#ifndef CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUNONCONTIGLAYOUT_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUNONCONTIGLAYOUT_H_

#include <vector>

#include "Type.hpp"
#include "cytnx_error.hpp"

// Shared layout metadata for the non-contiguous tensor(op)tensor GPU elementwise
// kernels (arithmetic out-of-place, arithmetic in-place, and comparison). Every
// such kernel needs, per tensor dimension, the output strides (accu_shape), each
// operand's original strides (old_accu_shape{L,R}) and permutation (invmapper_{L,R})
// so it can turn a linear index into the two operands' physical offsets.
//
// Previously each launch cuMalloc_gpu'd five device arrays, cudaMemcpy'd two of
// them, and cudaFree'd all five on *every* call -- managed-memory alloc/free is
// expensive and fully serial with the launch (#1003 step 4). Instead we pack the
// five rank-sized arrays into one POD passed to the kernel BY VALUE, so it rides in
// the kernel's constant/param space with no device allocation at all.
//
// The five arrays are bounded by kGpuNonContigMaxRank. This is never the binding
// limit in practice: the non-contiguous kernels also stage a per-thread scratch of
// 512 * rank cytnx_uint64 in dynamic shared memory, which exhausts the 48 KB default
// block budget around rank 12 -- far below this cap. A rank above the cap now raises
// a clear error instead of the previous cryptic kernel-launch failure.

namespace cytnx {
  namespace linalg_internal {
    namespace gpu_layout {

      constexpr int kGpuNonContigMaxRank = 32;

      struct GpuNonContigLayout {
        cytnx_uint64 accu_shape[kGpuNonContigMaxRank];
        cytnx_uint64 old_accu_shapeL[kGpuNonContigMaxRank];
        cytnx_uint64 old_accu_shapeR[kGpuNonContigMaxRank];
        cytnx_uint64 invmapper_L[kGpuNonContigMaxRank];
        cytnx_uint64 invmapper_R[kGpuNonContigMaxRank];
        cytnx_uint64 shapesize;
      };

      // Host-side builder: fills the layout from the output shape and the two operand
      // inverse mappers (identical arithmetic to the old per-launch stride loop, but
      // writing into the by-value struct instead of five freshly-allocated device
      // buffers). `shape` has one entry per dimension; empty shape means the operands
      // are contiguous and the caller must not take the non-contiguous path.
      inline GpuNonContigLayout MakeGpuNonContigLayout(
        const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L,
        const std::vector<cytnx_uint64> &invmapper_R) {
        const cytnx_uint64 rank = shape.size();
        cytnx_error_msg(
          rank > static_cast<cytnx_uint64>(kGpuNonContigMaxRank),
          "[GPU elementwise] non-contiguous tensor rank %d exceeds the supported maximum %d.%s",
          static_cast<int>(rank), kGpuNonContigMaxRank, "\n");

        GpuNonContigLayout layout{};
        layout.shapesize = rank;
        cytnx_uint64 accu = 1, accuL = 1, accuR = 1;
        for (cytnx_uint64 i = 0; i < rank; i++) {
          const cytnx_uint64 d = rank - 1 - i;
          layout.accu_shape[d] = accu;
          accu *= shape[d];
          layout.old_accu_shapeL[d] = accuL;
          accuL *= shape[invmapper_L[d]];
          layout.old_accu_shapeR[d] = accuR;
          accuR *= shape[invmapper_R[d]];
        }
        for (cytnx_uint64 j = 0; j < rank; j++) {
          layout.invmapper_L[j] = invmapper_L[j];
          layout.invmapper_R[j] = invmapper_R[j];
        }
        return layout;
      }

      // Device-side index math shared by all three non-contiguous kernels: decompose
      // the linear index `idx` into the output multi-index (staged in the caller's
      // per-thread `tmpv` shared-memory slice) via the output strides, then recompose
      // each operand's physical offset via its permutation and original strides.
      __device__ inline void ComputeGpuNonContigIndices(const cytnx_uint64 idx, cytnx_uint64 *tmpv,
                                                        const GpuNonContigLayout &layout,
                                                        cytnx_uint64 &Lidx, cytnx_uint64 &Ridx) {
        const cytnx_uint64 offset = threadIdx.x * layout.shapesize;
        cytnx_uint64 tmp = idx;
        for (cytnx_uint64 j = 0; j < layout.shapesize; j++) {
          tmpv[offset + j] = tmp / layout.accu_shape[j];
          tmp = tmp % layout.accu_shape[j];
        }
        Lidx = 0;
        Ridx = 0;
        for (cytnx_uint64 j = 0; j < layout.shapesize; j++) {
          Lidx += tmpv[offset + layout.invmapper_L[j]] * layout.old_accu_shapeL[j];
          Ridx += tmpv[offset + layout.invmapper_R[j]] * layout.old_accu_shapeR[j];
        }
      }

    }  // namespace gpu_layout
  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUNONCONTIGLAYOUT_H_
