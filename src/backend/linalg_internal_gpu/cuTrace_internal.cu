#include "cuTrace_internal.hpp"
#include "cytnx_error.hpp"
#include "Tensor.hpp"
#include "backend/Storage.hpp"
#include "backend/utils_internal_gpu/cuTypeTraits_gpu.hpp"

#include <algorithm>
#include <iterator>
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
      constexpr cytnx_uint64 kMaxGridDimX = 2147483647ULL;
      constexpr cytnx_uint64 kMaxGridDimY = 65535ULL;

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
      __device__ cytnx_uint64 DecodeDiagonalStartOffset(cytnx_uint64 output_index,
                                                        const cytnx_uint64 *surviving_shape,
                                                        const cytnx_uint64 *surviving_input_stride,
                                                        cytnx_uint64 surviving_rank) {
        cytnx_uint64 remaining_flat_index = output_index;
        cytnx_uint64 diagonal_start_offset = 0;
        for (cytnx_uint64 axis = surviving_rank; axis-- > 0;) {
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
      __device__ T BlockTraceDiagonal(const T *input_data, cytnx_uint64 diagonal_start_offset,
                                      cytnx_uint64 diagonal_length, cytnx_uint64 diagonal_stride) {
        T thread_partial_sum = T(0);
        for (cytnx_uint64 i = threadIdx.x; i < diagonal_length; i += blockDim.x)
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

      // One block per output element (blockIdx.x is the output index). The caller
      // sizes blockDim per call so each block has roughly diagonal_length threads
      // (rounded up to a warp, capped at kMaxTraceThreadsPerBlock): a short
      // diagonal launches a small block (no idle warps), a long diagonal launches
      // the maximum-size block (all threads working). A rank-2 trace
      // (output_size == 1) is just one such block; many short-diagonal outputs
      // launch many small blocks running concurrently across SMs.
      template <class T>
      __global__ void TraceKernel(T *output_data, const T *input_data,
                                  const cytnx_uint64 *surviving_shape,
                                  const cytnx_uint64 *surviving_input_stride,
                                  cytnx_uint64 surviving_rank, cytnx_uint64 output_size,
                                  cytnx_uint64 diagonal_length, cytnx_uint64 diagonal_stride) {
        const cytnx_uint64 output_index =
          static_cast<cytnx_uint64>(blockIdx.y) * gridDim.x + blockIdx.x;
        if (output_index >= output_size) return;

        const cytnx_uint64 diagonal_start_offset = DecodeDiagonalStartOffset(
          output_index, surviving_shape, surviving_input_stride, surviving_rank);
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
        // Trace() validates upstream that ax1 != ax2 and shape[ax1] == shape[ax2],
        // so their order is irrelevant: the diagonal stride and the set of
        // surviving axes are symmetric in (ax1, ax2).
        const auto &input_shape = Tn.shape();
        const std::vector<cytnx_int64> input_strides = Tn.strides();
        const cytnx_uint64 diagonal_length = input_shape[ax1];
        const cytnx_uint64 diagonal_stride =
          static_cast<cytnx_uint64>(input_strides[ax1] + input_strides[ax2]);

        // Build the reduced output shape and the matching per-surviving-axis input
        // strides in a single pass over the surviving axes (no separate surviving-
        // axis index list).
        std::vector<cytnx_int64> output_shape;  // for Tensor::reshape_
        std::vector<cytnx_uint64> host_surviving_shape;  // same dims, device-bound
        std::vector<cytnx_uint64> host_surviving_input_stride;
        for (cytnx_uint64 axis = 0; axis < input_shape.size(); ++axis) {
          if (axis != ax1 && axis != ax2) {
            output_shape.push_back(CheckedCastToInt64Gpu(input_shape[axis], "input_shape[axis]"));
            host_surviving_shape.push_back(input_shape[axis]);
            host_surviving_input_stride.push_back(static_cast<cytnx_uint64>(input_strides[axis]));
          }
        }
        const cytnx_uint64 surviving_rank = host_surviving_shape.size();
        cytnx_uint64 output_size = 1;
        for (auto dim : host_surviving_shape) output_size *= dim;
        const bool output_is_scalar = surviving_rank == 0;

        // Fill a device-resident result Storage.
        Storage output_storage(output_is_scalar ? cytnx_uint64{1} : output_size, Tn.dtype(),
                               Tn.device());
        if (diagonal_length == 0 || output_size == 0) {
          output_storage.set_zeros();
          return ComposeTraceOutput(output_storage, output_shape, output_is_scalar);
        }

        // Ship the two surviving_rank-sized layout arrays the multi-index decode
        // needs; the rank-2 case (surviving_rank == 0) needs neither, so the
        // kernel reads nullptr only where the decode loop never runs.
        //
        // Allocated with cudaMallocAsync (stream-ordered) rather than cudaMalloc
        // so they can be released with cudaFreeAsync below -- cudaFreeAsync only
        // accepts pointers allocated by cudaMallocAsync/cudaMallocFromPoolAsync.
        cytnx_uint64 *device_surviving_shape = nullptr;
        cytnx_uint64 *device_surviving_input_stride = nullptr;
        if (surviving_rank > 0) {
          checkCudaErrors(cudaMallocAsync((void **)&device_surviving_shape,
                                          sizeof(cytnx_uint64) * surviving_rank, 0));
          checkCudaErrors(cudaMallocAsync((void **)&device_surviving_input_stride,
                                          sizeof(cytnx_uint64) * surviving_rank, 0));
          checkCudaErrors(cudaMemcpyAsync(device_surviving_shape, host_surviving_shape.data(),
                                          sizeof(cytnx_uint64) * surviving_rank,
                                          cudaMemcpyHostToDevice, 0));
          checkCudaErrors(
            cudaMemcpyAsync(device_surviving_input_stride, host_surviving_input_stride.data(),
                            sizeof(cytnx_uint64) * surviving_rank, cudaMemcpyHostToDevice, 0));
        }

        // Size each block to its diagonal: diagonal_length is known here, so the
        // block gets just enough threads to cover it (rounded up to a whole warp,
        // capped at kMaxTraceThreadsPerBlock). A short diagonal launches a small
        // block with no idle warps; a long diagonal gets the full-size block and
        // each thread strides. One block per output element, so many short
        // diagonals still reduce concurrently across SMs, and the rank-2 trace
        // (output_size == 1) puts every thread of its single block on the one
        // diagonal.
        const cytnx_uint64 threads_per_block = std::min<cytnx_uint64>(
          ((diagonal_length + kWarpSize - 1) / kWarpSize) * kWarpSize, kMaxTraceThreadsPerBlock);

        // One block per output element. output_size can exceed dim3::x's 32-bit
        // range, so it is spread across grid.x and grid.y instead of being passed
        // directly as the <<<...>>> launch's first argument (which would
        // implicitly narrow to dim3 and silently truncate to the low 32 bits).
        cytnx_error_msg(output_size > kMaxGridDimX * kMaxGridDimY,
                        "[internal][cuTrace] output_size=%llu exceeds the maximum grid size "
                        "(%llu) this launch can address.\n",
                        static_cast<unsigned long long>(output_size),
                        static_cast<unsigned long long>(kMaxGridDimX * kMaxGridDimY));
        const dim3 grid_dim(
          static_cast<unsigned int>(std::min(output_size, kMaxGridDimX)),
          static_cast<unsigned int>((output_size + kMaxGridDimX - 1) / kMaxGridDimX));
        TraceKernel<CudaT><<<grid_dim, threads_per_block>>>(
          reinterpret_cast<CudaT *>(output_storage.data()),
          reinterpret_cast<const CudaT *>(Tn.storage().data()), device_surviving_shape,
          device_surviving_input_stride, surviving_rank, output_size, diagonal_length,
          diagonal_stride);
        // Surface a launch/configuration failure at the trace call rather than at
        // the next, unrelated CUDA call.
        checkCudaErrors(cudaGetLastError());

        if (surviving_rank > 0) {
          // cudaEventSynchronize waits only for TraceKernel (recorded on this
          // same stream), not the whole device; cudaFreeAsync then enqueues the
          // release into the stream-ordered memory pool without the host
          // blocking on the free itself. This is groundwork for a future where
          // GPU work runs across multiple concurrent streams -- today, with
          // every op on the default stream, it costs the same as a direct
          // cudaFree, but it stops scaling into a device-wide stall once that
          // stops being true.
          cudaEvent_t kernel_done;
          checkCudaErrors(cudaEventCreate(&kernel_done));
          checkCudaErrors(cudaEventRecord(kernel_done, 0));
          checkCudaErrors(cudaEventSynchronize(kernel_done));
          checkCudaErrors(cudaFreeAsync(device_surviving_shape, 0));
          checkCudaErrors(cudaFreeAsync(device_surviving_input_stride, 0));
          checkCudaErrors(cudaEventDestroy(kernel_done));
        }

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
