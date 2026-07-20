#ifndef CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUNONCONTIGLAYOUT_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUNONCONTIGLAYOUT_H_

#include <vector>

#include "Type.hpp"
#include "cytnx_error.hpp"

// Shared layout metadata for the non-contiguous tensor(op)tensor GPU elementwise
// kernels (arithmetic out-of-place, arithmetic in-place, and comparison). Every
// such kernel needs, per output dimension, the output stride (to decompose a
// linear index into the output coordinate) and each operand's stride in its own
// physical buffer for that output dimension (to gather that operand's element).
//
// Previously each launch cuMalloc_gpu'd five device arrays, cudaMemcpy'd two of
// them, and cudaFree'd all five on *every* call -- managed-memory alloc/free is
// expensive and fully serial with the launch (#1003 step 4). Instead we pack the
// per-dimension strides into one POD passed to the kernel BY VALUE, so it rides in
// the kernel's constant/param space with no device allocation at all.
//
// The layout stores per-output-dimension operand strides directly rather than the
// operands' permutations + original strides: precomputing strideL/strideR on the
// host (`strideL[invmapper_L[d]] = old_stride_of_L_at_d`) folds the operand offset
// straight into the index-decompose loop. That removes the inverse-mapper
// indirection and, crucially, the per-thread shared-memory multi-index scratch the
// old recompose needed -- so the kernels launch with no dynamic shared memory and
// the rank is bounded only by the by-value struct's parameter-space footprint.

namespace cytnx {
  namespace linalg_internal {
    namespace gpu_layout {

      // Bounds the three rank-sized arrays carried by value in GpuNonContigLayout.
      // The struct is 3 * kGpuNonContigMaxRank * sizeof(cytnx_uint64) + 8 B (= 776 B
      // at 32), well within the >=4 KB CUDA kernel parameter budget. (No longer
      // bounded by shared memory: the strided decompose uses none.)
      constexpr int kGpuNonContigMaxRank = 32;

      struct GpuNonContigLayout {
        cytnx_uint64 accu_shape[kGpuNonContigMaxRank];  // output stride per output dim
        cytnx_uint64 strideL[kGpuNonContigMaxRank];  // operand-L stride per output dim
        cytnx_uint64 strideR[kGpuNonContigMaxRank];  // operand-R stride per output dim
        cytnx_uint64 shapesize;
      };

      // Host-side builder: fills the layout from the output shape and the two operand
      // inverse mappers. For each output dimension d it records the output stride
      // (accu_shape) and folds each operand's original stride into the output-dim slot
      // it maps to (strideL[invmapper_L[d]] / strideR[invmapper_R[d]]), so the device
      // side never touches the inverse mappers again. `shape` has one entry per
      // dimension; an empty shape means the operands are contiguous and the caller must
      // not take the non-contiguous path.
      inline GpuNonContigLayout make_gpu_non_contig_layout(
        const std::vector<cytnx_uint64> &shape, const std::vector<cytnx_uint64> &invmapper_L,
        const std::vector<cytnx_uint64> &invmapper_R) {
        const cytnx_uint64 rank = shape.size();
        cytnx_error_msg(
          rank > static_cast<cytnx_uint64>(kGpuNonContigMaxRank),
          "[GPU elementwise] non-contiguous tensor rank %d exceeds the supported maximum %d.%s",
          static_cast<int>(rank), kGpuNonContigMaxRank, "\n");
        cytnx_error_msg(invmapper_L.size() != rank || invmapper_R.size() != rank,
                        "[GPU elementwise] invmapper size mismatch: rank=%d, |invmapper_L|=%d, "
                        "|invmapper_R|=%d.%s",
                        static_cast<int>(rank), static_cast<int>(invmapper_L.size()),
                        static_cast<int>(invmapper_R.size()), "\n");

        GpuNonContigLayout layout{};
        layout.shapesize = rank;
        cytnx_uint64 accu = 1, accuL = 1, accuR = 1;
        for (cytnx_uint64 i = 0; i < rank; i++) {
          const cytnx_uint64 d = rank - 1 - i;
          layout.accu_shape[d] = accu;
          accu *= shape[d];
          // Original stride of each operand at output dimension d, routed to the output
          // slot that operand's permutation sends d to.
          layout.strideL[invmapper_L[d]] = accuL;
          accuL *= shape[invmapper_L[d]];
          layout.strideR[invmapper_R[d]] = accuR;
          accuR *= shape[invmapper_R[d]];
        }
        return layout;
      }

      // Device-side index math shared by all three non-contiguous kernels. Decompose
      // the linear index `idx` into each output coordinate via the output strides and,
      // in the same pass, accumulate each operand's physical offset via its
      // per-output-dimension stride. No shared-memory scratch is needed.
      __device__ inline void compute_gpu_non_contig_indices(cytnx_uint64 idx,
                                                            const GpuNonContigLayout &layout,
                                                            cytnx_uint64 &Lidx,
                                                            cytnx_uint64 &Ridx) {
        Lidx = 0;
        Ridx = 0;
        cytnx_uint64 tmp = idx;
        for (cytnx_uint64 j = 0; j < layout.shapesize; j++) {
          const cytnx_uint64 coord = tmp / layout.accu_shape[j];
          tmp %= layout.accu_shape[j];
          Lidx += coord * layout.strideL[j];
          Ridx += coord * layout.strideR[j];
        }
      }

    }  // namespace gpu_layout
  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUNONCONTIGLAYOUT_H_
