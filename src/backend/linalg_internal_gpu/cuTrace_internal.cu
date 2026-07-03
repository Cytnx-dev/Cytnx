#include "cuTrace_internal.hpp"
#include "cytnx_error.hpp"
#include "Tensor.hpp"
#include "backend/Storage.hpp"
#include "backend/utils_internal_gpu/cuTypeTraits_gpu.hpp"

#include <algorithm>
#include <limits>
#include <vector>

#include "cuda/std/complex"

namespace cytnx {
  namespace linalg_internal {
    namespace {

      // Number of threads a GPU executes in lockstep (warpSize in the CUDA
      // runtime). Fixed at 32 on every current NVIDIA architecture; the
      // __shfl_down_sync reductions below assume this value.
      constexpr int kWarpSize = 32;
      // Cap on threads per block for the trace kernel. Each block is sized to
      // its diagonal (rounded up to a whole warp) and saturates at this cap.
      constexpr int kMaxTraceThreadsPerBlock = 256;
      // Capacity of the shared array of per-warp partial sums: one slot per
      // warp of a maximum-size block.
      constexpr int kMaxTraceWarpsPerBlock = kMaxTraceThreadsPerBlock / kWarpSize;
      static_assert(kMaxTraceThreadsPerBlock % kWarpSize == 0,
                    "Blocks are sized in whole warps, so the cap must be a warp multiple.");
      // The per-warp tree reduction below halves the active lane count each step,
      // which only covers every lane when the warp size is a power of two.
      static_assert((kWarpSize & (kWarpSize - 1)) == 0,
                    "kWarpSize must be a power of two for the tree reduction.");
      // Max extent of grid.x on every architecture this project targets (CUDA
      // compute capability >= 3.0). One block is launched per output element;
      // output counts at or below this fit in a 1-D grid. Above it, the launch
      // spills into grid.y (max 65535), so passing output_size directly as the
      // <<<...>>> launch's first argument would implicitly narrow to dim3.x
      // (32-bit) and silently truncate the grid -- see the 2-D grid dispatch in
      // TraceImplGpu below.
      constexpr cytnx_int64 kMaxGridDimX = 2147483647LL;
      constexpr cytnx_int64 kMaxGridDimY = 65535LL;

      // Cap on the surviving (non-traced) rank TraceKernel handles via its
      // stack-resident TraceLayout. Every surviving axis has extent >= 2 (a
      // size-1 axis contributes nothing to a diagonal decode and would be
      // squeezed away, not kept as a surviving axis), so a tensor with
      // surviving_rank axes plus the two traced axes has at least
      // 2^(surviving_rank + 2) elements. x86-64 addresses at most 2^52 bytes,
      // so even a 1-byte-per-element dtype caps surviving_rank at 50 on any
      // physically constructible machine -- there is no dtype/hardware
      // combination that can reach 51. 50 costs nothing extra against CUDA's
      // per-launch parameter/constant-memory budget, so there is no reason to
      // size this any tighter, and no fallback path is needed: the case this
      // constant would guard against cannot occur.
      constexpr cytnx_int64 kMaxTraceRank = 50;

      // Surviving-axis shape/stride, sized to kMaxTraceRank so it can be
      // passed to TraceKernel by value as a __grid_constant__ parameter
      // instead of two separate device allocations.
      struct TraceLayout {
        cytnx_int64 rank;
        cytnx_int64 shape[kMaxTraceRank];
        cytnx_int64 stride[kMaxTraceRank];
      };

      // Narrows a shape/stride extent (mathematically non-negative, stored as
      // cytnx_uint64) to cytnx_int64, rejecting values that would overflow.
      // internal::CheckedCastToInt64 (utils/checked_cast.hpp) does the same
      // check on the CPU path, but its std::source_location::current() default
      // argument does not compile under nvcc's host-code frontend, so the GPU
      // path checks inline instead.
      cytnx_int64 CheckedCastToInt64Gpu(cytnx_uint64 value, const char *name) {
        cytnx_error_msg(value > static_cast<cytnx_uint64>(std::numeric_limits<cytnx_int64>::max()),
                        "[internal][cuTrace] %s=%llu exceeds cytnx_int64 max.\n", name,
                        static_cast<unsigned long long>(value));
        return static_cast<cytnx_int64>(value);
      }

      // value + (the value `reduction_stride` lanes higher in the warp). For real
      // types this is a single __shfl_down_sync; complex storage is shuffled
      // component-wise since __shfl_down_sync only moves scalar lanes. (Small
      // integer types promote to int through the shuffle and truncate back, which
      // preserves the modular-sum semantics the trace already has.)
      template <class T>
      __device__ T WarpShuffleDownAdd(T value, int reduction_stride) {
        return value + __shfl_down_sync(0xffffffff, value, reduction_stride);
      }
      template <class F>
      __device__ cuda::std::complex<F> WarpShuffleDownAdd(cuda::std::complex<F> value,
                                                          int reduction_stride) {
        return cuda::std::complex<F>(
          value.real() + __shfl_down_sync(0xffffffff, value.real(), reduction_stride),
          value.imag() + __shfl_down_sync(0xffffffff, value.imag(), reduction_stride));
      }

      // Decodes a flat output index into the input storage offset of that output
      // element's diagonal start: walk the surviving axes, accumulating
      // (index_along_axis * input_stride_of_axis). For a rank-2 trace
      // (surviving_rank == 0) the loop is empty and the offset is 0.
      //
      // Every index/extent here is a single signed type (cytnx_int64), mirroring
      // the CPU TraceImpl (Trace_internal.cpp) so this arithmetic needs no casts
      // between calls.
      __device__ cytnx_int64 DecodeDiagonalStartOffset(cytnx_int64 output_index,
                                                       const cytnx_int64 *surviving_shape,
                                                       const cytnx_int64 *surviving_input_stride,
                                                       cytnx_int64 surviving_rank) {
        cytnx_int64 remaining_flat_index = output_index;
        cytnx_int64 diagonal_start_offset = 0;
        for (cytnx_int64 axis = surviving_rank; axis-- > 0;) {
          diagonal_start_offset +=
            (remaining_flat_index % surviving_shape[axis]) * surviving_input_stride[axis];
          remaining_flat_index /= surviving_shape[axis];
        }
        return diagonal_start_offset;
      }

      // Sums one diagonal with the whole block (blockDim.x threads): each thread
      // strides over the diagonal, each warp reduces its lanes with a warp-shuffle
      // tree, the per-warp sums land in shared memory, and the first warp reduces
      // those. Thread 0 returns the result. The reduction adapts to whatever
      // blockDim.x the kernel was launched with -- the caller sizes blockDim per
      // call to diagonal_length so a short diagonal never spawns idle warps.
      template <class T>
      __device__ T BlockTraceDiagonal(const T *input_data, cytnx_int64 diagonal_start_offset,
                                      cytnx_int64 diagonal_length, cytnx_int64 diagonal_stride) {
        T thread_partial_sum = T(0);
        // threadIdx.x/blockDim.x are CUDA's native unsigned int; cast once here at
        // that boundary and keep the stride math in cytnx_int64 from then on.
        for (cytnx_int64 i = threadIdx.x; i < diagonal_length; i += blockDim.x)
          thread_partial_sum += input_data[diagonal_start_offset + i * diagonal_stride];

        // Reduce within each warp.
        for (int reduction_stride = kWarpSize / 2; reduction_stride > 0; reduction_stride /= 2)
          thread_partial_sum = WarpShuffleDownAdd(thread_partial_sum, reduction_stride);

        const unsigned int warps_in_block = (blockDim.x + kWarpSize - 1) / kWarpSize;
        if (warps_in_block == 1) {
          // Single-warp launch: thread_partial_sum on lane 0 already holds the
          // total. No shared memory or barrier needed.
          return thread_partial_sum;
        }

        // Lane 0 of each warp publishes its warp's sum; the first warp reduces them.
        __shared__ T warp_sums[kMaxTraceWarpsPerBlock];
        const unsigned int lane = threadIdx.x % kWarpSize;
        const unsigned int warp_in_block = threadIdx.x / kWarpSize;
        if (lane == 0) warp_sums[warp_in_block] = thread_partial_sum;
        __syncthreads();

        T block_sum = T(0);
        if (warp_in_block == 0) {
          T v = (lane < warps_in_block) ? warp_sums[lane] : T(0);
          for (int reduction_stride = kWarpSize / 2; reduction_stride > 0; reduction_stride /= 2)
            v = WarpShuffleDownAdd(v, reduction_stride);
          block_sum = v;
        }
        return block_sum;  // valid on thread 0
      }

      // One block per output element (blockIdx.x is the output index). The
      // caller sizes blockDim per call so each block has roughly
      // diagonal_length threads (rounded up to a warp, capped at
      // kMaxTraceThreadsPerBlock): a short diagonal launches a small block (no
      // idle warps), a long diagonal launches the maximum-size block (all
      // threads working). A rank-2 trace (output_size == 1) is just one such
      // block; many short-diagonal outputs launch many small blocks running
      // concurrently across SMs.
      //
      // The surviving-axis layout is a small fixed-size struct (TraceLayout)
      // passed by value as a __grid_constant__ parameter rather than two
      // device pointers, so a trace call needs no cudaMalloc/cudaMemcpy/
      // cudaFree at all -- kMaxTraceRank comfortably covers every surviving
      // rank reachable on any physically constructible machine (see its
      // definition above), so there is no larger-rank fallback to dispatch
      // to. __grid_constant__ places the parameter in read-only per-grid
      // constant memory rather than copying it into every thread's local
      // memory.
      template <class T>
      __global__ void TraceKernel(T *output_data, const T *input_data,
                                  const __grid_constant__ TraceLayout layout,
                                  cytnx_int64 output_size, cytnx_int64 diagonal_length,
                                  cytnx_int64 diagonal_stride) {
        const cytnx_int64 output_index =
          static_cast<cytnx_int64>(blockIdx.y) * gridDim.x + blockIdx.x;
        if (output_index >= output_size) return;

        const cytnx_int64 diagonal_start_offset =
          DecodeDiagonalStartOffset(output_index, layout.shape, layout.stride, layout.rank);
        T sum = BlockTraceDiagonal<T>(input_data, diagonal_start_offset, diagonal_length,
                                      diagonal_stride);
        if (threadIdx.x == 0) output_data[output_index] = sum;
      }

      // Composes the output Tensor from its already-filled storage, reshaping to
      // output_shape unless the trace is a rank-2 scalar. Tensor::from_storage
      // keeps the storage on its current device, so no host round-trip is
      // involved.
      Tensor ComposeTraceOutput(Storage &output_storage,
                                const std::vector<cytnx_int64> &output_shape,
                                bool output_is_scalar) {
        Tensor out = Tensor::from_storage(output_storage);
        if (!output_is_scalar) out.reshape_(output_shape);
        return out;
      }

      template <class T>
      Tensor TraceImplGpu(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
        using CudaT = typename utils_internal::ToCudaDType<T>::type;
        // The kernel launch below runs on this stream, named rather than
        // passed as a bare 0 so the whole function only has to change in one
        // place if this ever moves off the default stream.
        const cudaStream_t stream = 0;
        // Trace() validates upstream that ax1 != ax2 and shape[ax1] == shape[ax2],
        // so their order is irrelevant: the diagonal stride and the set of
        // surviving axes are symmetric in (ax1, ax2).
        // Every index/extent below is kept as a single signed type (cytnx_int64),
        // mirroring the CPU TraceImpl (Trace_internal.cpp) -- input_shape's
        // elements and surviving_rank are the only values that start out
        // unsigned (Tn.shape() and std::vector::size()), so those are the only
        // two places a cast is needed at all.
        const auto &input_shape = Tn.shape();
        const std::vector<cytnx_int64> input_strides = Tn.strides();
        const cytnx_int64 diagonal_length =
          CheckedCastToInt64Gpu(input_shape[ax1], "input_shape[ax1]");
        const cytnx_int64 diagonal_stride = input_strides[ax1] + input_strides[ax2];

        // Build the reduced output shape (needed for Tensor::reshape_ on every
        // path) and, in the same pass, write directly into a stack-resident
        // TraceLayout wherever it still fits (idx < kMaxTraceRank) -- the
        // common case needs no heap allocation or host->device copy at all
        // for the surviving-axis layout; see the dispatch below.
        TraceLayout layout;
        std::vector<cytnx_int64> output_shape;  // for Tensor::reshape_
        cytnx_int64 idx = 0;
        for (cytnx_uint64 axis = 0; axis < input_shape.size(); ++axis) {
          if (axis != ax1 && axis != ax2) {
            const cytnx_int64 dim = CheckedCastToInt64Gpu(input_shape[axis], "input_shape[axis]");
            output_shape.push_back(dim);
            if (idx < kMaxTraceRank) {
              layout.shape[idx] = dim;
              layout.stride[idx] = input_strides[axis];
            }
            ++idx;
          }
        }
        const cytnx_int64 surviving_rank = idx;
        // Defensive, not a supported case: kMaxTraceRank's derivation above
        // shows no physically constructible tensor can reach this. If it were
        // ever hit anyway, layout.shape/stride beyond kMaxTraceRank weren't
        // populated by the loop above, so proceeding would read uninitialized
        // stack memory on the device.
        cytnx_error_msg(surviving_rank > kMaxTraceRank,
                        "[internal][cuTrace] surviving_rank=%lld exceeds kMaxTraceRank (%lld).\n",
                        static_cast<long long>(surviving_rank),
                        static_cast<long long>(kMaxTraceRank));
        layout.rank = surviving_rank;
        cytnx_int64 output_size = 1;
        for (cytnx_int64 dim : output_shape) output_size *= dim;
        const bool output_is_scalar = surviving_rank == 0;

        // Fill a device-resident result Storage.
        Storage output_storage(
          output_is_scalar ? cytnx_uint64{1} : static_cast<cytnx_uint64>(output_size), Tn.dtype(),
          Tn.device());
        if (diagonal_length == 0 || output_size == 0) {
          output_storage.set_zeros();
          return ComposeTraceOutput(output_storage, output_shape, output_is_scalar);
        }

        // Size each block to its diagonal: diagonal_length is known here, so the
        // block gets just enough threads to cover it (rounded up to a whole warp,
        // capped at kMaxTraceThreadsPerBlock). A short diagonal launches a small
        // block with no idle warps; a long diagonal gets the full-size block and
        // each thread strides. One block per output element, so many short
        // diagonals still reduce concurrently across SMs, and the rank-2 trace
        // (output_size == 1) puts every thread of its single block on the one
        // diagonal.
        const cytnx_int64 threads_per_block = std::min<cytnx_int64>(
          ((diagonal_length + kWarpSize - 1) / kWarpSize) * kWarpSize, kMaxTraceThreadsPerBlock);

        // One block per output element. output_size can exceed dim3::x's 32-bit
        // range, so it is spread across grid.x and grid.y instead of being passed
        // directly as the <<<...>>> launch's first argument (which would
        // implicitly narrow to dim3 and silently truncate to the low 32 bits).
        // The final casts to unsigned int here are the actual CUDA API boundary
        // (dim3's fields), not a signed/unsigned bookkeeping cast.
        cytnx_error_msg(output_size > kMaxGridDimX * kMaxGridDimY,
                        "[internal][cuTrace] output_size=%lld exceeds the maximum grid size "
                        "(%lld) this launch can address.\n",
                        static_cast<long long>(output_size),
                        static_cast<long long>(kMaxGridDimX * kMaxGridDimY));
        const dim3 grid_dim(
          static_cast<unsigned int>(std::min(output_size, kMaxGridDimX)),
          static_cast<unsigned int>((output_size + kMaxGridDimX - 1) / kMaxGridDimX));

        // layout is already fully populated above -- launch directly, no
        // cudaMalloc/cudaMemcpy/cudaFree at all. 0 below is the dynamic
        // shared-memory size in bytes (the kernel uses none; its __shared__
        // warp_sums array is statically sized).
        TraceKernel<CudaT><<<grid_dim, threads_per_block, /* shared_mem_bytes */ 0, stream>>>(
          reinterpret_cast<CudaT *>(output_storage.data()),
          reinterpret_cast<const CudaT *>(Tn.storage().data()), layout, output_size,
          diagonal_length, diagonal_stride);
        // Surface a launch/configuration failure at the trace call rather than
        // at the next, unrelated CUDA call.
        checkCudaErrors(cudaGetLastError());

        return ComposeTraceOutput(output_storage, output_shape, output_is_scalar);
      }

    }  // namespace

    Tensor cuTrace_internal_cd(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceImplGpu<cytnx_complex128>(Tn, ax1, ax2);
    }
    Tensor cuTrace_internal_cf(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceImplGpu<cytnx_complex64>(Tn, ax1, ax2);
    }
    Tensor cuTrace_internal_d(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceImplGpu<cytnx_double>(Tn, ax1, ax2);
    }
    Tensor cuTrace_internal_f(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceImplGpu<cytnx_float>(Tn, ax1, ax2);
    }
    Tensor cuTrace_internal_u64(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceImplGpu<cytnx_uint64>(Tn, ax1, ax2);
    }
    Tensor cuTrace_internal_i64(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceImplGpu<cytnx_int64>(Tn, ax1, ax2);
    }
    Tensor cuTrace_internal_u32(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceImplGpu<cytnx_uint32>(Tn, ax1, ax2);
    }
    Tensor cuTrace_internal_i32(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceImplGpu<cytnx_int32>(Tn, ax1, ax2);
    }
    Tensor cuTrace_internal_u16(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceImplGpu<cytnx_uint16>(Tn, ax1, ax2);
    }
    Tensor cuTrace_internal_i16(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      return TraceImplGpu<cytnx_int16>(Tn, ax1, ax2);
    }

    Tensor cuTrace_internal_b(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
      cytnx_error_msg(true, "[internal][cuTrace] bool is not available. %s", "\n");
      return Tensor();
    }

  }  // namespace linalg_internal

}  // namespace cytnx
