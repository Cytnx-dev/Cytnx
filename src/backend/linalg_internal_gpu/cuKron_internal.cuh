#ifndef CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUKRON_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUKRON_INTERNAL_H_

#include <vector>

#include "Type.hpp"
#include "cytnx_error.hpp"

#ifndef __CUDACC__
  #error This file requires a CUDA compiler
#endif

// GPU Kronecker product. The kernel needs, per output dimension d, three strides --
// the output stride (to decompose a linear output index into the output coordinate),
// each operand's stride in its own buffer -- plus the rhs extent, which splits the
// output coordinate into the lhs coordinate (quotient) and the rhs coordinate
// (remainder).
//
// That is launch-time metadata, not tensor data. It used to be assembled on *every*
// call (#1003) into one cuMalloc_gpu'd array (cudaMallocManaged) filled by four
// pageable cudaMemcpy's, staged into dynamic shared memory by the kernel, and then
// cudaFree'd -- six API calls that each synchronize with the device, so all of them
// serialize against the launch they exist to configure. The shared-memory staging was
// unsound on top of that: it copied 4*rank entries using only blockDim.x threads, so a
// rank above 64 silently read uninitialised shared memory. Both go away by packing the
// metadata into one POD passed to the kernel BY VALUE, which rides in the kernel's
// parameter space with no device allocation and no shared memory. This mirrors
// GpuNonContigLayout for the elementwise binary kernels (cuNonContigLayout.cuh).

namespace cytnx {

  namespace linalg_internal {

    namespace gpu_kron {

      // Bounds the four rank-sized arrays carried by value in GpuKronLayout:
      // 4 * kGpuKronMaxRank * sizeof(cytnx_uint64) + 16 B (= 1040 B at 32), well within
      // the >=4 KB CUDA kernel parameter budget. Matches kGpuNonContigMaxRank so the GPU
      // elementwise paths share one documented rank limit.
      constexpr int kGpuKronMaxRank = 32;

      constexpr int kKronThreadsPerBlock = 256;

      struct GpuKronLayout {
        cytnx_uint64 out_stride[kGpuKronMaxRank];  // output stride per output dim
        cytnx_uint64 lhs_stride[kGpuKronMaxRank];  // lhs stride per output dim
        cytnx_uint64 rhs_stride[kGpuKronMaxRank];  // rhs stride per output dim
        cytnx_uint64 rhs_shape[kGpuKronMaxRank];  // rhs extent per output dim
        cytnx_uint64 rank;
        cytnx_uint64 total_elems;
      };

      // Host-side builder. Both shapes are the operands' shapes after Kron's rank
      // padding, so they have one entry per output dimension and equal length. Output
      // dimension d has extent shape_lhs[d] * shape_rhs[d]; the strides are the usual
      // row-major suffix products of each shape, and total_elems is the output size.
      inline GpuKronLayout make_gpu_kron_layout(const std::vector<cytnx_uint64> &shape_lhs,
                                                const std::vector<cytnx_uint64> &shape_rhs) {
        cytnx_error_msg(shape_lhs.size() != shape_rhs.size(),
                        "[ERROR][Internal cuKron] T1 rank != T2 rank %s", "\n");
        const cytnx_uint64 rank = shape_lhs.size();
        cytnx_error_msg(rank > static_cast<cytnx_uint64>(kGpuKronMaxRank),
                        "[ERROR][Internal cuKron] rank %d exceeds the supported GPU maximum %d.%s",
                        static_cast<int>(rank), kGpuKronMaxRank, "\n");

        GpuKronLayout layout{};
        layout.rank = rank;
        cytnx_uint64 out_accu = 1, lhs_accu = 1, rhs_accu = 1;
        for (cytnx_uint64 i = 0; i < rank; i++) {
          const cytnx_uint64 d = rank - 1 - i;
          layout.out_stride[d] = out_accu;
          layout.lhs_stride[d] = lhs_accu;
          layout.rhs_stride[d] = rhs_accu;
          layout.rhs_shape[d] = shape_rhs[d];
          out_accu *= shape_lhs[d] * shape_rhs[d];
          lhs_accu *= shape_lhs[d];
          rhs_accu *= shape_rhs[d];
        }
        layout.total_elems = out_accu;
        return layout;
      }

    }  // namespace gpu_kron

    template <class TO, class TL, class TR>
    __global__ void cuKron_kernel(TO *out, const TL *Lin, const TR *Rin,
                                  const gpu_kron::GpuKronLayout layout) {
      const cytnx_uint64 idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx >= layout.total_elems) return;

      cytnx_uint64 tmp = idx;
      cytnx_uint64 x = 0, y = 0;
      for (cytnx_uint64 j = 0; j < layout.rank; j++) {
        const cytnx_uint64 coord = tmp / layout.out_stride[j];
        tmp %= layout.out_stride[j];
        x += (coord / layout.rhs_shape[j]) * layout.lhs_stride[j];
        y += (coord % layout.rhs_shape[j]) * layout.rhs_stride[j];
      }
      // Compute through the operation's output type (#1003): TO is the promoted dtype the
      // caller already allocated, so converting both operands first keeps the arithmetic on
      // Cytnx's promotion rules instead of C++'s native ones.
      out[idx] = static_cast<TO>(Lin[x]) * static_cast<TO>(Rin[y]);
    }

    template <class TO, class TL, class TR>
    void cuKron_general(TO *out, const TL *Lin, const TR *Rin,
                        const std::vector<cytnx_uint64> &shape1,
                        const std::vector<cytnx_uint64> &shape2) {
      const gpu_kron::GpuKronLayout layout = gpu_kron::make_gpu_kron_layout(shape1, shape2);
      if (layout.total_elems == 0) return;

      cytnx_uint64 nblocks = layout.total_elems / gpu_kron::kKronThreadsPerBlock;
      if (layout.total_elems % gpu_kron::kKronThreadsPerBlock) nblocks += 1;

      cuKron_kernel<<<nblocks, gpu_kron::kKronThreadsPerBlock>>>(out, Lin, Rin, layout);
    }

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUKRON_INTERNAL_H_
