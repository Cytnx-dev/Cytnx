#include "cuMaxMin_internal.hpp"

#include <cstdint>
#include <utility>

#include "cuda/std/complex"

#include "backend/utils_internal_gpu/cuScopedResource_gpu.hpp"
#include "backend/utils_internal_gpu/cuTypeTraits_gpu.hpp"
#include "cytnx_error.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace linalg_internal {

    namespace {

      // Threads per block of the reduction kernel. Must be a power of two: the in-block tree
      // merges runs of 1, 2, 4, ... elements.
      constexpr unsigned int kReduceBlockSize = 512;

      // `earlier` precedes `later` in storage order. Both rules below reproduce the CPU
      // implementation (MaxMin_internal.cpp): real elements are replaced only by a strictly
      // better one, so the first of several equal elements is kept; complex elements are ordered
      // by their real part alone and the last of several with equal real part is kept.
      struct PickMax {
        template <class T>
        __device__ T operator()(T earlier, T later) const {
          return later > earlier ? later : earlier;
        }
        template <class F>
        __device__ cuda::std::complex<F> operator()(cuda::std::complex<F> earlier,
                                                    cuda::std::complex<F> later) const {
          return later.real() >= earlier.real() ? later : earlier;
        }
      };
      struct PickMin {
        template <class T>
        __device__ T operator()(T earlier, T later) const {
          return later < earlier ? later : earlier;
        }
        template <class F>
        __device__ cuda::std::complex<F> operator()(cuda::std::complex<F> earlier,
                                                    cuda::std::complex<F> later) const {
          return later.real() <= earlier.real() ? later : earlier;
        }
      };

      // Reduces each block of blockDim.x consecutive input elements to one output element.
      // Requires blockDim.x to be a power of two no larger than kReduceBlockSize.
      template <class T, class Pick>
      __global__ void reduce_blocks_kernel(T *out, const T *in, std::uint64_t element_count) {
        __shared__ T block_values[kReduceBlockSize];

        // Slots past the end of the input are filled with the last element. A repeated element
        // cannot change a maximum or minimum, whereas a fixed filler such as 0 can.
        const std::uint64_t index =
          static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        block_values[threadIdx.x] = in[index < element_count ? index : element_count - 1];
        __syncthreads();

        // Neighbouring runs are merged, so a slot always holds the result for one contiguous run
        // of elements and its right-hand partner the run that follows it; Pick relies on that
        // order. Every step is followed by a barrier: a thread reads a slot that another thread
        // wrote in the previous step, and threads of one warp do not run in lockstep.
        for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
          if (threadIdx.x % (2 * stride) == 0) {
            block_values[threadIdx.x] =
              Pick()(block_values[threadIdx.x], block_values[threadIdx.x + stride]);
          }
          __syncthreads();
        }

        if (threadIdx.x == 0) out[blockIdx.x] = block_values[0];
      }

      // Writes the Pick-reduction of in[0, element_count) to out[0]. element_count must be > 0.
      template <class T, class Pick>
      void reduce_to_scalar(T *out, const T *in, std::uint64_t element_count) {
        // Intermediate levels alternate between two buffers. The kernels run in launch order on
        // one stream, so a level may overwrite the buffer that the level before it read from. The
        // first buffer holds the first level's results and the second the (512 times fewer)
        // results of the second level; every later level fits into one of the two.
        utils_internal::DeviceBuffer<T> level_input;
        utils_internal::DeviceBuffer<T> level_output;
        while (true) {
          const std::uint64_t block_count =
            (element_count + kReduceBlockSize - 1) / kReduceBlockSize;
          if (block_count > 1 && level_output.get() == nullptr) {
            level_output = utils_internal::DeviceBuffer<T>(block_count);
          }
          T *block_results = block_count > 1 ? level_output.get() : out;
          reduce_blocks_kernel<T, Pick>
            <<<static_cast<unsigned int>(block_count), kReduceBlockSize>>>(block_results, in,
                                                                           element_count);
          CYTNX_CHECK_CUDA_LAUNCH(reduce_blocks_kernel);
          if (block_count == 1) break;
          in = block_results;
          element_count = block_count;
          std::swap(level_input, level_output);
        }
        checkCudaErrors(cudaDeviceSynchronize());
      }

      template <class HostType>
      void max_min_gpu(boost::intrusive_ptr<Storage_base> &out,
                       const boost::intrusive_ptr<Storage_base> &ten, char type) {
        using DeviceType = typename utils_internal::ToCudaDType<HostType>::type;
        DeviceType *out_data = static_cast<DeviceType *>(out->data());
        const DeviceType *in_data = static_cast<const DeviceType *>(ten->data());
        if (type == 'x') {
          reduce_to_scalar<DeviceType, PickMax>(out_data, in_data, ten->size());
        } else {
          reduce_to_scalar<DeviceType, PickMin>(out_data, in_data, ten->size());
        }
      }

    }  // namespace

    void cuMaxMin_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                              const boost::intrusive_ptr<Storage_base> &ten,
                              const cytnx_uint64 &Nelem, const char &type) {
      max_min_gpu<cytnx_complex128>(out, ten, type);
    }
    void cuMaxMin_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                              const boost::intrusive_ptr<Storage_base> &ten,
                              const cytnx_uint64 &Nelem, const char &type) {
      max_min_gpu<cytnx_complex64>(out, ten, type);
    }
    void cuMaxMin_internal_d(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten,
                             const cytnx_uint64 &Nelem, const char &type) {
      max_min_gpu<cytnx_double>(out, ten, type);
    }
    void cuMaxMin_internal_f(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten,
                             const cytnx_uint64 &Nelem, const char &type) {
      max_min_gpu<cytnx_float>(out, ten, type);
    }
    void cuMaxMin_internal_u64(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &ten,
                               const cytnx_uint64 &Nelem, const char &type) {
      max_min_gpu<cytnx_uint64>(out, ten, type);
    }
    void cuMaxMin_internal_i64(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &ten,
                               const cytnx_uint64 &Nelem, const char &type) {
      max_min_gpu<cytnx_int64>(out, ten, type);
    }
    void cuMaxMin_internal_u32(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &ten,
                               const cytnx_uint64 &Nelem, const char &type) {
      max_min_gpu<cytnx_uint32>(out, ten, type);
    }
    void cuMaxMin_internal_i32(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &ten,
                               const cytnx_uint64 &Nelem, const char &type) {
      max_min_gpu<cytnx_int32>(out, ten, type);
    }
    void cuMaxMin_internal_u16(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &ten,
                               const cytnx_uint64 &Nelem, const char &type) {
      max_min_gpu<cytnx_uint16>(out, ten, type);
    }
    void cuMaxMin_internal_i16(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &ten,
                               const cytnx_uint64 &Nelem, const char &type) {
      max_min_gpu<cytnx_int16>(out, ten, type);
    }
    void cuMaxMin_internal_b(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten,
                             const cytnx_uint64 &Nelem, const char &type) {
      max_min_gpu<cytnx_bool>(out, ten, type);
    }

  }  // namespace linalg_internal
}  // namespace cytnx
