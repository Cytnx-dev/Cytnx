#include "cuReduce_gpu.hpp"

#include <type_traits>

#include "cuda/std/complex"

#include "Type.hpp"

namespace cytnx {
  namespace utils_internal {

#define _TNinB_REDUCE_ 512

    template <class X>
    __device__ void warp_unroll(X* smem, int tid) {
      X v = smem[tid];
      __syncwarp();
      for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_down_sync(0xFFFFFFFFU, v, offset);
      }
      smem[tid] = v;
    }

    template <class X>
    __device__ void warp_unroll(cuda::std::complex<X>* smem, int tid) {
      cuda::std::complex<X> v = smem[tid];
      X v_real = v.real(), v_imag = v.imag();
      __syncwarp();
      for (int offset = 16; offset > 0; offset /= 2) {
        v_real += __shfl_down_sync(0xFFFFFFFFU, v_real, offset);
        v_imag += __shfl_down_sync(0xFFFFFFFFU, v_imag, offset);
      }
      smem[tid] = cuda::std::complex<X>{v_real, v_imag};
    }

    // require, threads per block to be 32*(2^n), n =0,1,2,3,4,5
    template <class CudaT>
    __global__ void cuReduce_kernel(CudaT* out, CudaT* in, cytnx_uint64 Nelem) {
      __shared__ CudaT sD[_TNinB_REDUCE_];  // allocate share mem for each thread
      sD[threadIdx.x] = 0;

      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        sD[threadIdx.x] = in[blockIdx.x * blockDim.x + threadIdx.x];
      }
      __syncthreads();

      if (blockDim.x >= 1024) {
        if (threadIdx.x < 512) {
          sD[threadIdx.x] += sD[threadIdx.x + 512];
        }
        __syncthreads();
      }
      if (blockDim.x >= 512) {
        if (threadIdx.x < 256) {
          sD[threadIdx.x] += sD[threadIdx.x + 256];
        }
        __syncthreads();
      }
      if (blockDim.x >= 256) {
        if (threadIdx.x < 128) {
          sD[threadIdx.x] += sD[threadIdx.x + 128];
        }
        __syncthreads();
      }
      if (blockDim.x >= 128) {
        if (threadIdx.x < 64) {
          sD[threadIdx.x] += sD[threadIdx.x + 64];
        }
        __syncthreads();
      }
      if (blockDim.x >= 64) {
        if (threadIdx.x < 32) {
          sD[threadIdx.x] += sD[threadIdx.x + 32];
        }
        __syncthreads();
      }

      if (threadIdx.x < 32) warp_unroll(sD, threadIdx.x);
      __syncthreads();

      if (threadIdx.x == 0) out[blockIdx.x] = sD[0];  // write to global for block
    }

    template <class T>
    inline void swap(T*& a, T*& b) {
      T* tmp = a;
      a = b;
      b = tmp;
    }

    template <class CudaT>
    void cuReduce_gpu_generic(CudaT* out, CudaT* in, const cytnx_uint64& Nelem) {
      cytnx_uint64 Nelems = Nelem;
      cytnx_uint64 NBlocks;

      NBlocks = Nelems / _TNinB_REDUCE_;
      if (Nelems % _TNinB_REDUCE_) NBlocks += 1;

      // alloc mem for each block:
      CudaT* dblk;
      checkCudaErrors(cudaMalloc((void**)&dblk, NBlocks * sizeof(*dblk)));
      if (NBlocks == 1) {
        cuReduce_kernel<<<NBlocks, _TNinB_REDUCE_>>>(out, in, Nelems);
      } else {
        cuReduce_kernel<<<NBlocks, _TNinB_REDUCE_>>>(dblk, in, Nelems);
      }
      Nelems = NBlocks;

      while (Nelems > 1) {
        NBlocks = Nelems / _TNinB_REDUCE_;
        if (Nelems % _TNinB_REDUCE_) NBlocks += 1;

        if (NBlocks == 1) {
          cuReduce_kernel<<<NBlocks, _TNinB_REDUCE_>>>(out, dblk, Nelems);
        } else {
          CudaT* dblk2;
          cudaMalloc((void**)&dblk2, NBlocks * sizeof(*dblk2));
          cuReduce_kernel<<<NBlocks, _TNinB_REDUCE_>>>(dblk2, dblk, Nelems);

          swap(dblk2, dblk);
          cudaFree(dblk2);
        }
        Nelems = NBlocks;
      }
      cudaFree(dblk);
    }

    template <class T>
    void cuReduce_gpu(T* out, T* in, const cytnx_uint64& Nelem) {
      if constexpr (std::is_same_v<T, cytnx_complex128> || std::is_same_v<T, cytnx_complex64>) {
        using cuda_complex = cuda::std::complex<typename T::value_type>;
        return cuReduce_gpu_generic(reinterpret_cast<cuda_complex*>(out),
                                    reinterpret_cast<cuda_complex*>(in), Nelem);
      } else {
        return cuReduce_gpu_generic(out, in, Nelem);
      }
    }

    template void cuReduce_gpu<cytnx_complex128>(cytnx_complex128*, cytnx_complex128*,
                                                 const cytnx_uint64&);
    template void cuReduce_gpu<cytnx_complex64>(cytnx_complex64*, cytnx_complex64*,
                                                const cytnx_uint64&);
    template void cuReduce_gpu<cytnx_double>(cytnx_double*, cytnx_double*, const cytnx_uint64&);
    template void cuReduce_gpu<cytnx_float>(cytnx_float*, cytnx_float*, const cytnx_uint64&);
    template void cuReduce_gpu<cytnx_uint64>(cytnx_uint64*, cytnx_uint64*, const cytnx_uint64&);
    template void cuReduce_gpu<cytnx_int64>(cytnx_int64*, cytnx_int64*, const cytnx_uint64&);
    template void cuReduce_gpu<cytnx_uint32>(cytnx_uint32*, cytnx_uint32*, const cytnx_uint64&);
    template void cuReduce_gpu<cytnx_int32>(cytnx_int32*, cytnx_int32*, const cytnx_uint64&);
    template void cuReduce_gpu<cytnx_uint16>(cytnx_uint16*, cytnx_uint16*, const cytnx_uint64&);
    template void cuReduce_gpu<cytnx_int16>(cytnx_int16*, cytnx_int16*, const cytnx_uint64&);

  }  // namespace utils_internal
}  // namespace cytnx
