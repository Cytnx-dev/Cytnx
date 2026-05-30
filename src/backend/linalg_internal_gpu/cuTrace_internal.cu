#include "cuTrace_internal.hpp"
#include "cytnx_error.hpp"
#include "Tensor.hpp"
#include "backend/Storage.hpp"

#include <algorithm>
#include <vector>

#include "cuda/std/complex"

namespace cytnx {
  namespace linalg_internal {
    namespace {

      constexpr int kTraceThreadsPerBlock = 512;
      // The single-block tree reduction below halves the active thread count each
      // step, which only covers every element when the block size is a power of two.
      static_assert((kTraceThreadsPerBlock & (kTraceThreadsPerBlock - 1)) == 0,
                    "kTraceThreadsPerBlock must be a power of two for the tree reduction.");

      // Maps a cytnx storage type to the device arithmetic type used inside the
      // kernels. Complex storage is bit-compatible with cuda::std::complex, which
      // provides device operator+ / construction-from-zero, so the kernels stay
      // type-generic.
      template <class T>
      struct TraceCudaType {
        using type = T;
      };
      template <>
      struct TraceCudaType<cytnx_complex128> {
        using type = cuda::std::complex<double>;
      };
      template <>
      struct TraceCudaType<cytnx_complex64> {
        using type = cuda::std::complex<float>;
      };
      template <class T>
      using TraceCudaTypeT = typename TraceCudaType<T>::type;

      // Sums one diagonal: every thread strides over the diagonal_length entries
      // starting at diagonal_start_offset (step diagonal_stride), then a
      // shared-memory tree reduction combines the per-thread partials into the
      // single *output_element. This is the whole computation for a rank-2 trace
      // (one diagonal -> one scalar); the trace of higher-rank inputs runs one
      // block per output element and calls this with that element's offset.
      template <class T>
      __device__ void TraceDiagonalBlock(T *output_element, const T *input_data,
                                         cytnx_uint64 diagonal_start_offset,
                                         cytnx_uint64 diagonal_length,
                                         cytnx_uint64 diagonal_stride) {
        __shared__ T block_partial_sums[kTraceThreadsPerBlock];
        T thread_partial_sum = T(0);
        for (cytnx_uint64 i = threadIdx.x; i < diagonal_length; i += blockDim.x)
          thread_partial_sum += input_data[diagonal_start_offset + i * diagonal_stride];
        block_partial_sums[threadIdx.x] = thread_partial_sum;
        __syncthreads();

        for (unsigned int reduction_stride = blockDim.x >> 1; reduction_stride > 0;
             reduction_stride >>= 1) {
          if (threadIdx.x < reduction_stride)
            block_partial_sums[threadIdx.x] += block_partial_sums[threadIdx.x + reduction_stride];
          __syncthreads();
        }

        if (threadIdx.x == 0) *output_element = block_partial_sums[0];
      }

      // One block per output element. The block decodes its flat index into the
      // surviving-axis multi-index, accumulating the input base offset from
      // surviving_input_stride (the input stride of each surviving axis), then
      // sums that element's diagonal via TraceDiagonalBlock. A rank-2 trace is
      // the special case surviving_rank == 0 / output_size == 1: the decode loop
      // is empty and the diagonal starts at offset 0, so the single block traces
      // the whole matrix diagonal.
      template <class T>
      __global__ void TraceKernel(T *output_data, const T *input_data,
                                  const cytnx_uint64 *surviving_shape,
                                  const cytnx_uint64 *surviving_input_stride,
                                  cytnx_uint64 surviving_rank, cytnx_uint64 output_size,
                                  cytnx_uint64 diagonal_length, cytnx_uint64 diagonal_stride) {
        cytnx_uint64 output_index = blockIdx.x;
        if (output_index >= output_size) return;

        cytnx_uint64 remaining_flat_index = output_index;
        cytnx_uint64 diagonal_start_offset = 0;
        for (cytnx_uint64 axis = surviving_rank; axis-- > 0;) {
          diagonal_start_offset +=
            (remaining_flat_index % surviving_shape[axis]) * surviving_input_stride[axis];
          remaining_flat_index /= surviving_shape[axis];
        }
        TraceDiagonalBlock<T>(&output_data[output_index], input_data, diagonal_start_offset,
                              diagonal_length, diagonal_stride);
      }

      template <class T>
      Tensor TraceImplGpu(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2) {
        using CudaT = TraceCudaTypeT<T>;
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
            output_shape.push_back(static_cast<cytnx_int64>(input_shape[axis]));
            host_surviving_shape.push_back(input_shape[axis]);
            host_surviving_input_stride.push_back(static_cast<cytnx_uint64>(input_strides[axis]));
          }
        }
        const cytnx_uint64 surviving_rank = host_surviving_shape.size();
        cytnx_uint64 output_size = 1;
        for (auto dim : host_surviving_shape) output_size *= dim;
        const bool output_is_scalar = surviving_rank == 0;

        // Fill a device-resident result Storage, then compose the output Tensor
        // from it; Tensor::from_storage keeps the storage on its current device, so
        // no host round-trip is involved.
        Storage output_storage(output_is_scalar ? cytnx_uint64{1} : output_size, Tn.dtype(),
                               Tn.device());
        if (diagonal_length == 0 || output_size == 0) {
          output_storage.set_zeros();
          Tensor out = Tensor::from_storage(output_storage);
          if (!output_is_scalar) out.reshape_(output_shape);
          return out;
        }

        // Ship the two surviving_rank-sized layout arrays the multi-index decode
        // needs; the rank-2 case (surviving_rank == 0) needs neither, so the
        // kernel reads nullptr only where the decode loop never runs.
        cytnx_uint64 *device_surviving_shape = nullptr;
        cytnx_uint64 *device_surviving_input_stride = nullptr;
        if (surviving_rank > 0) {
          checkCudaErrors(
            cudaMalloc((void **)&device_surviving_shape, sizeof(cytnx_uint64) * surviving_rank));
          checkCudaErrors(cudaMalloc((void **)&device_surviving_input_stride,
                                     sizeof(cytnx_uint64) * surviving_rank));
          checkCudaErrors(cudaMemcpy(device_surviving_shape, host_surviving_shape.data(),
                                     sizeof(cytnx_uint64) * surviving_rank,
                                     cudaMemcpyHostToDevice));
          checkCudaErrors(
            cudaMemcpy(device_surviving_input_stride, host_surviving_input_stride.data(),
                       sizeof(cytnx_uint64) * surviving_rank, cudaMemcpyHostToDevice));
        }

        // One block per output element (output_size == 1 for the rank-2 case).
        TraceKernel<CudaT><<<output_size, kTraceThreadsPerBlock>>>(
          reinterpret_cast<CudaT *>(output_storage.data()),
          reinterpret_cast<const CudaT *>(Tn.storage().data()), device_surviving_shape,
          device_surviving_input_stride, surviving_rank, output_size, diagonal_length,
          diagonal_stride);
        // Surface a launch/configuration failure at the trace call rather than at
        // the next, unrelated CUDA call.
        checkCudaErrors(cudaGetLastError());

        if (surviving_rank > 0) {
          // cudaFree synchronizes the device, so the kernel is guaranteed complete
          // (and its reads of the layout arrays done) before the buffers are freed.
          checkCudaErrors(cudaFree(device_surviving_shape));
          checkCudaErrors(cudaFree(device_surviving_input_stride));
        }

        Tensor out = Tensor::from_storage(output_storage);
        if (!output_is_scalar) out.reshape_(output_shape);
        return out;
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
