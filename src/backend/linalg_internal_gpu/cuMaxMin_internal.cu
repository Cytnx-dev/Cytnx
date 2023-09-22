#include "cuMaxMin_internal.hpp"
#include "cytnx_error.hpp"
#include "Type.hpp"
#include "backend/lapack_wrapper.hpp"

namespace cytnx {
  namespace linalg_internal {
    using namespace std;

#define _TNinB_REDUCE_ 512

    template <class X>
    __device__ void warp_unroll_max(X *smem, int tid) {
      X v;
      v = (smem[tid] >= smem[tid + 32]) ? smem[tid] : smem[tid + 32];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid] >= smem[tid + 16]) ? smem[tid] : smem[tid + 16];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid] >= smem[tid + 8]) ? smem[tid] : smem[tid + 8];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid] >= smem[tid + 4]) ? smem[tid] : smem[tid + 4];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid] >= smem[tid + 2]) ? smem[tid] : smem[tid + 2];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid] >= smem[tid + 1]) ? smem[tid] : smem[tid + 1];
      __syncwarp();
      smem[tid] = v;
    }

    __device__ void warp_unroll_max_cd(cuDoubleComplex *smem, int tid) {
      cuDoubleComplex v;
      v = (smem[tid].x >= smem[tid + 32].x) ? smem[tid] : smem[tid + 32];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid].x >= smem[tid + 16].x) ? smem[tid] : smem[tid + 16];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid].x >= smem[tid + 8].x) ? smem[tid] : smem[tid + 8];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid].x >= smem[tid + 4].x) ? smem[tid] : smem[tid + 4];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid].x >= smem[tid + 2].x) ? smem[tid] : smem[tid + 2];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid].x >= smem[tid + 1].x) ? smem[tid] : smem[tid + 1];
      __syncwarp();
      smem[tid] = v;
    }
    __device__ void warp_unroll_max_cf(cuFloatComplex *smem, int tid) {
      cuFloatComplex v;
      v = (smem[tid].x >= smem[tid + 32].x) ? smem[tid] : smem[tid + 32];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid].x >= smem[tid + 16].x) ? smem[tid] : smem[tid + 16];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid].x >= smem[tid + 8].x) ? smem[tid] : smem[tid + 8];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid].x >= smem[tid + 4].x) ? smem[tid] : smem[tid + 4];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid].x >= smem[tid + 2].x) ? smem[tid] : smem[tid + 2];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid].x >= smem[tid + 1].x) ? smem[tid] : smem[tid + 1];
      __syncwarp();
      smem[tid] = v;
    }

    template <class X>
    __device__ void warp_unroll_min(X *smem, int tid) {
      X v;
      v = (smem[tid] <= smem[tid + 32]) ? smem[tid] : smem[tid + 32];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid] <= smem[tid + 16]) ? smem[tid] : smem[tid + 16];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid] <= smem[tid + 8]) ? smem[tid] : smem[tid + 8];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid] <= smem[tid + 4]) ? smem[tid] : smem[tid + 4];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid] <= smem[tid + 2]) ? smem[tid] : smem[tid + 2];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid] <= smem[tid + 1]) ? smem[tid] : smem[tid + 1];
      __syncwarp();
      smem[tid] = v;
    }
    __device__ void warp_unroll_min_cd(cuDoubleComplex *smem, int tid) {
      cuDoubleComplex v;
      v = (smem[tid].x <= smem[tid + 32].x) ? smem[tid] : smem[tid + 32];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid].x <= smem[tid + 16].x) ? smem[tid] : smem[tid + 16];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid].x <= smem[tid + 8].x) ? smem[tid] : smem[tid + 8];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid].x <= smem[tid + 4].x) ? smem[tid] : smem[tid + 4];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid].x <= smem[tid + 2].x) ? smem[tid] : smem[tid + 2];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid].x <= smem[tid + 1].x) ? smem[tid] : smem[tid + 1];
      __syncwarp();
      smem[tid] = v;
    }
    __device__ void warp_unroll_min_cf(cuFloatComplex *smem, int tid) {
      cuFloatComplex v;
      v = (smem[tid].x <= smem[tid + 32].x) ? smem[tid] : smem[tid + 32];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid].x <= smem[tid + 16].x) ? smem[tid] : smem[tid + 16];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid].x <= smem[tid + 8].x) ? smem[tid] : smem[tid + 8];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid].x <= smem[tid + 4].x) ? smem[tid] : smem[tid + 4];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid].x <= smem[tid + 2].x) ? smem[tid] : smem[tid + 2];
      __syncwarp();
      smem[tid] = v;
      v = (smem[tid].x <= smem[tid + 1].x) ? smem[tid] : smem[tid + 1];
      __syncwarp();
      smem[tid] = v;
    }

    // require, threads per block to be 32*(2^n), n =0,1,2,3,4,5
    template <class T>
    __global__ void cuMax_kernel(T *out, T *in, cytnx_uint64 Nelem) {
      __shared__ T sD[_TNinB_REDUCE_];  // allocate share mem for each thread
      sD[threadIdx.x] = 0;

      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        sD[threadIdx.x] = in[blockIdx.x * blockDim.x + threadIdx.x];
      }
      __syncthreads();

      if (blockDim.x >= 1024) {
        if (threadIdx.x < 512) {
          sD[threadIdx.x] =
            (sD[threadIdx.x] >= sD[threadIdx.x + 512]) ? sD[threadIdx.x] : sD[threadIdx.x + 512];
        }
        __syncthreads();
      }
      if (blockDim.x >= 512) {
        if (threadIdx.x < 256) {
          sD[threadIdx.x] =
            (sD[threadIdx.x] >= sD[threadIdx.x + 256]) ? sD[threadIdx.x] : sD[threadIdx.x + 256];
        }
        __syncthreads();
      }
      if (blockDim.x >= 256) {
        if (threadIdx.x < 128) {
          sD[threadIdx.x] =
            (sD[threadIdx.x] >= sD[threadIdx.x + 128]) ? sD[threadIdx.x] : sD[threadIdx.x + 128];
        }
        __syncthreads();
      }
      if (blockDim.x >= 128) {
        if (threadIdx.x < 64) {
          sD[threadIdx.x] =
            (sD[threadIdx.x] >= sD[threadIdx.x + 64]) ? sD[threadIdx.x] : sD[threadIdx.x + 64];
        }
        __syncthreads();
      }

      if (threadIdx.x < 32) warp_unroll_max(sD, threadIdx.x);
      __syncthreads();

      if (threadIdx.x == 0) out[blockIdx.x] = sD[0];  // write to global for block
    }

    // require, threads per block to be 32*(2^n), n =0,1,2,3,4,5
    __global__ void cuMax_kernel(cuDoubleComplex *out, cuDoubleComplex *in, cytnx_uint64 Nelem) {
      __shared__ cuDoubleComplex sD[_TNinB_REDUCE_];  // allocate share mem for each thread
      sD[threadIdx.x] = make_cuDoubleComplex(0, 0);

      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        sD[threadIdx.x] = in[blockIdx.x * blockDim.x + threadIdx.x];
      }
      __syncthreads();

      if (blockDim.x >= 1024) {
        if (threadIdx.x < 512) {
          sD[threadIdx.x] = (sD[threadIdx.x].x >= sD[threadIdx.x + 512].x) ? sD[threadIdx.x]
                                                                           : sD[threadIdx.x + 512];
        }
        __syncthreads();
      }
      if (blockDim.x >= 512) {
        if (threadIdx.x < 256) {
          sD[threadIdx.x] = (sD[threadIdx.x].x >= sD[threadIdx.x + 256].x) ? sD[threadIdx.x]
                                                                           : sD[threadIdx.x + 256];
        }
        __syncthreads();
      }
      if (blockDim.x >= 256) {
        if (threadIdx.x < 128) {
          sD[threadIdx.x] = (sD[threadIdx.x].x >= sD[threadIdx.x + 128].x) ? sD[threadIdx.x]
                                                                           : sD[threadIdx.x + 128];
        }
        __syncthreads();
      }
      if (blockDim.x >= 128) {
        if (threadIdx.x < 64) {
          sD[threadIdx.x] =
            (sD[threadIdx.x].x >= sD[threadIdx.x + 64].x) ? sD[threadIdx.x] : sD[threadIdx.x + 64];
        }
        __syncthreads();
      }

      if (threadIdx.x < 32) warp_unroll_max_cd(sD, threadIdx.x);
      __syncthreads();

      if (threadIdx.x == 0) out[blockIdx.x] = sD[0];  // write to global for block
    }
    // require, threads per block to be 32*(2^n), n =0,1,2,3,4,5
    __global__ void cuMax_kernel(cuFloatComplex *out, cuFloatComplex *in, cytnx_uint64 Nelem) {
      __shared__ cuFloatComplex sD[_TNinB_REDUCE_];  // allocate share mem for each thread
      sD[threadIdx.x] = make_cuFloatComplex(0, 0);

      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        sD[threadIdx.x] = in[blockIdx.x * blockDim.x + threadIdx.x];
      }
      __syncthreads();

      if (blockDim.x >= 1024) {
        if (threadIdx.x < 512) {
          sD[threadIdx.x] = (sD[threadIdx.x].x >= sD[threadIdx.x + 512].x) ? sD[threadIdx.x]
                                                                           : sD[threadIdx.x + 512];
        }
        __syncthreads();
      }
      if (blockDim.x >= 512) {
        if (threadIdx.x < 256) {
          sD[threadIdx.x] = (sD[threadIdx.x].x >= sD[threadIdx.x + 256].x) ? sD[threadIdx.x]
                                                                           : sD[threadIdx.x + 256];
        }
        __syncthreads();
      }
      if (blockDim.x >= 256) {
        if (threadIdx.x < 128) {
          sD[threadIdx.x] = (sD[threadIdx.x].x >= sD[threadIdx.x + 128].x) ? sD[threadIdx.x]
                                                                           : sD[threadIdx.x + 128];
        }
        __syncthreads();
      }
      if (blockDim.x >= 128) {
        if (threadIdx.x < 64) {
          sD[threadIdx.x] =
            (sD[threadIdx.x].x >= sD[threadIdx.x + 64].x) ? sD[threadIdx.x] : sD[threadIdx.x + 64];
        }
        __syncthreads();
      }

      if (threadIdx.x < 32) warp_unroll_max_cf(sD, threadIdx.x);
      __syncthreads();

      if (threadIdx.x == 0) out[blockIdx.x] = sD[0];  // write to global for block
    }

    template <class T>
    __global__ void cuMin_kernel(T *out, T *in, cytnx_uint64 Nelem) {
      __shared__ T sD[_TNinB_REDUCE_];  // allocate share mem for each thread
      sD[threadIdx.x] = 0;

      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        sD[threadIdx.x] = in[blockIdx.x * blockDim.x + threadIdx.x];
      }
      __syncthreads();

      if (blockDim.x >= 1024) {
        if (threadIdx.x < 512) {
          sD[threadIdx.x] =
            (sD[threadIdx.x] <= sD[threadIdx.x + 512]) ? sD[threadIdx.x] : sD[threadIdx.x + 512];
        }
        __syncthreads();
      }
      if (blockDim.x >= 512) {
        if (threadIdx.x < 256) {
          sD[threadIdx.x] =
            (sD[threadIdx.x] <= sD[threadIdx.x + 256]) ? sD[threadIdx.x] : sD[threadIdx.x + 256];
        }
        __syncthreads();
      }
      if (blockDim.x >= 256) {
        if (threadIdx.x < 128) {
          sD[threadIdx.x] =
            (sD[threadIdx.x] <= sD[threadIdx.x + 128]) ? sD[threadIdx.x] : sD[threadIdx.x + 128];
        }
        __syncthreads();
      }
      if (blockDim.x >= 128) {
        if (threadIdx.x < 64) {
          sD[threadIdx.x] =
            (sD[threadIdx.x] <= sD[threadIdx.x + 64]) ? sD[threadIdx.x] : sD[threadIdx.x + 64];
        }
        __syncthreads();
      }

      if (threadIdx.x < 32) warp_unroll_min(sD, threadIdx.x);
      __syncthreads();

      if (threadIdx.x == 0) out[blockIdx.x] = sD[0];  // write to global for block
    }

    // require, threads per block to be 32*(2^n), n =0,1,2,3,4,5
    __global__ void cuMin_kernel(cuDoubleComplex *out, cuDoubleComplex *in, cytnx_uint64 Nelem) {
      __shared__ cuDoubleComplex sD[_TNinB_REDUCE_];  // allocate share mem for each thread
      sD[threadIdx.x] = make_cuDoubleComplex(0, 0);

      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        sD[threadIdx.x] = in[blockIdx.x * blockDim.x + threadIdx.x];
      }
      __syncthreads();

      if (blockDim.x >= 1024) {
        if (threadIdx.x < 512) {
          sD[threadIdx.x] = (sD[threadIdx.x].x <= sD[threadIdx.x + 512].x) ? sD[threadIdx.x]
                                                                           : sD[threadIdx.x + 512];
        }
        __syncthreads();
      }
      if (blockDim.x >= 512) {
        if (threadIdx.x < 256) {
          sD[threadIdx.x] = (sD[threadIdx.x].x <= sD[threadIdx.x + 256].x) ? sD[threadIdx.x]
                                                                           : sD[threadIdx.x + 256];
        }
        __syncthreads();
      }
      if (blockDim.x >= 256) {
        if (threadIdx.x < 128) {
          sD[threadIdx.x] = (sD[threadIdx.x].x <= sD[threadIdx.x + 128].x) ? sD[threadIdx.x]
                                                                           : sD[threadIdx.x + 128];
        }
        __syncthreads();
      }
      if (blockDim.x >= 128) {
        if (threadIdx.x < 64) {
          sD[threadIdx.x] =
            (sD[threadIdx.x].x <= sD[threadIdx.x + 64].x) ? sD[threadIdx.x] : sD[threadIdx.x + 64];
        }
        __syncthreads();
      }

      if (threadIdx.x < 32) warp_unroll_min_cd(sD, threadIdx.x);
      __syncthreads();

      if (threadIdx.x == 0) out[blockIdx.x] = sD[0];  // write to global for block
    }

    // require, threads per block to be 32*(2^n), n =0,1,2,3,4,5
    __global__ void cuMin_kernel(cuFloatComplex *out, cuFloatComplex *in, cytnx_uint64 Nelem) {
      __shared__ cuFloatComplex sD[_TNinB_REDUCE_];  // allocate share mem for each thread
      sD[threadIdx.x] = make_cuFloatComplex(0, 0);

      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        sD[threadIdx.x] = in[blockIdx.x * blockDim.x + threadIdx.x];
      }
      __syncthreads();

      if (blockDim.x >= 1024) {
        if (threadIdx.x < 512) {
          sD[threadIdx.x] = (sD[threadIdx.x].x <= sD[threadIdx.x + 512].x) ? sD[threadIdx.x]
                                                                           : sD[threadIdx.x + 512];
        }
        __syncthreads();
      }
      if (blockDim.x >= 512) {
        if (threadIdx.x < 256) {
          sD[threadIdx.x] = (sD[threadIdx.x].x <= sD[threadIdx.x + 256].x) ? sD[threadIdx.x]
                                                                           : sD[threadIdx.x + 256];
        }
        __syncthreads();
      }
      if (blockDim.x >= 256) {
        if (threadIdx.x < 128) {
          sD[threadIdx.x] = (sD[threadIdx.x].x <= sD[threadIdx.x + 128].x) ? sD[threadIdx.x]
                                                                           : sD[threadIdx.x + 128];
        }
        __syncthreads();
      }
      if (blockDim.x >= 128) {
        if (threadIdx.x < 64) {
          sD[threadIdx.x] =
            (sD[threadIdx.x].x <= sD[threadIdx.x + 64].x) ? sD[threadIdx.x] : sD[threadIdx.x + 64];
        }
        __syncthreads();
      }

      if (threadIdx.x < 32) warp_unroll_min_cf(sD, threadIdx.x);
      __syncthreads();

      if (threadIdx.x == 0) out[blockIdx.x] = sD[0];  // write to global for block
    }
    //=======================
    template <class T>
    void swap(T *&a, T *&b) {
      T *tmp = a;
      a = b;
      b = tmp;
    }

    template <class T>
    void cuMax_gpu_generic(T *out, T *in, const cytnx_uint64 &Nelem) {
      cytnx_uint64 Nelems = Nelem;
      cytnx_uint64 NBlocks;

      NBlocks = Nelems / _TNinB_REDUCE_;
      if (Nelems % _TNinB_REDUCE_) NBlocks += 1;

      // alloc mem for each block:
      T *dblk;
      // std::cout << NBlocks*sizeof(cytnx_double) << std::endl;
      cudaMalloc((void **)&dblk, NBlocks * sizeof(T));

      if (NBlocks == 1) {
        cuMax_kernel<<<NBlocks, _TNinB_REDUCE_>>>(out, in, Nelems);
      } else {
        cuMax_kernel<<<NBlocks, _TNinB_REDUCE_>>>(dblk, in, Nelems);
      }
      Nelems = NBlocks;

      while (Nelems > 1) {
        NBlocks = Nelems / _TNinB_REDUCE_;
        if (Nelems % _TNinB_REDUCE_) NBlocks += 1;

        if (NBlocks == 1) {
          cuMax_kernel<<<NBlocks, _TNinB_REDUCE_>>>(out, dblk, Nelems);
        } else {
          T *dblk2;
          cudaMalloc((void **)&dblk2, NBlocks * sizeof(T));
          // do something:
          cuMax_kernel<<<NBlocks, _TNinB_REDUCE_>>>(dblk2, dblk, Nelems);

          swap(dblk2, dblk);  // swap new data to old data, and free the old
          cudaFree(dblk2);
        }
        Nelems = NBlocks;
      }
      cudaFree(dblk);
    }
    template <class T>
    void cuMin_gpu_generic(T *out, T *in, const cytnx_uint64 &Nelem) {
      cytnx_uint64 Nelems = Nelem;
      cytnx_uint64 NBlocks;

      NBlocks = Nelems / _TNinB_REDUCE_;
      if (Nelems % _TNinB_REDUCE_) NBlocks += 1;

      // alloc mem for each block:
      T *dblk;
      // std::cout << NBlocks*sizeof(cytnx_double) << std::endl;
      cudaMalloc((void **)&dblk, NBlocks * sizeof(T));

      if (NBlocks == 1) {
        cuMin_kernel<<<NBlocks, _TNinB_REDUCE_>>>(out, in, Nelems);
      } else {
        cuMin_kernel<<<NBlocks, _TNinB_REDUCE_>>>(dblk, in, Nelems);
      }
      Nelems = NBlocks;

      while (Nelems > 1) {
        NBlocks = Nelems / _TNinB_REDUCE_;
        if (Nelems % _TNinB_REDUCE_) NBlocks += 1;

        if (NBlocks == 1) {
          cuMin_kernel<<<NBlocks, _TNinB_REDUCE_>>>(out, dblk, Nelems);
        } else {
          T *dblk2;
          cudaMalloc((void **)&dblk2, NBlocks * sizeof(T));
          // do something:
          cuMin_kernel<<<NBlocks, _TNinB_REDUCE_>>>(dblk2, dblk, Nelems);

          swap(dblk2, dblk);  // swap new data to old data, and free the old
          cudaFree(dblk2);
        }
        Nelems = NBlocks;
      }
      cudaFree(dblk);
    }

    void cuMaxMin_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                              const boost::intrusive_ptr<Storage_base> &ten,
                              const cytnx_uint64 &Nelem, const char &type) {
      if (type == 'x')
        cuMax_gpu_generic((cuDoubleComplex *)out->Mem, (cuDoubleComplex *)ten->Mem, ten->len);
      else
        cuMin_gpu_generic((cuDoubleComplex *)out->Mem, (cuDoubleComplex *)ten->Mem, ten->len);
    }
    void cuMaxMin_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                              const boost::intrusive_ptr<Storage_base> &ten,
                              const cytnx_uint64 &Nelem, const char &type) {
      if (type == 'x')
        cuMax_gpu_generic((cuFloatComplex *)out->Mem, (cuFloatComplex *)ten->Mem, ten->len);
      else
        cuMin_gpu_generic((cuFloatComplex *)out->Mem, (cuFloatComplex *)ten->Mem, ten->len);
    }
    void cuMaxMin_internal_d(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten,
                             const cytnx_uint64 &Nelem, const char &type) {
      if (type == 'x')
        cuMax_gpu_generic((cytnx_double *)out->Mem, (cytnx_double *)ten->Mem, ten->len);
      else
        cuMin_gpu_generic((cytnx_double *)out->Mem, (cytnx_double *)ten->Mem, ten->len);
    }
    void cuMaxMin_internal_f(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten,
                             const cytnx_uint64 &Nelem, const char &type) {
      if (type == 'x')
        cuMax_gpu_generic((cytnx_float *)out->Mem, (cytnx_float *)ten->Mem, ten->len);
      else
        cuMin_gpu_generic((cytnx_float *)out->Mem, (cytnx_float *)ten->Mem, ten->len);
    }
    void cuMaxMin_internal_u64(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &ten,
                               const cytnx_uint64 &Nelem, const char &type) {
      if (type == 'x')
        cuMax_gpu_generic((cytnx_uint64 *)out->Mem, (cytnx_uint64 *)ten->Mem, ten->len);
      else
        cuMin_gpu_generic((cytnx_uint64 *)out->Mem, (cytnx_uint64 *)ten->Mem, ten->len);
    }
    void cuMaxMin_internal_i64(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &ten,
                               const cytnx_uint64 &Nelem, const char &type) {
      if (type == 'x')
        cuMax_gpu_generic((cytnx_int64 *)out->Mem, (cytnx_int64 *)ten->Mem, ten->len);
      else
        cuMin_gpu_generic((cytnx_int64 *)out->Mem, (cytnx_int64 *)ten->Mem, ten->len);
    }
    void cuMaxMin_internal_u32(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &ten,
                               const cytnx_uint64 &Nelem, const char &type) {
      if (type == 'x')
        cuMax_gpu_generic((cytnx_uint32 *)out->Mem, (cytnx_uint32 *)ten->Mem, ten->len);
      else
        cuMin_gpu_generic((cytnx_uint32 *)out->Mem, (cytnx_uint32 *)ten->Mem, ten->len);
    }
    void cuMaxMin_internal_i32(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &ten,
                               const cytnx_uint64 &Nelem, const char &type) {
      if (type == 'x')
        cuMax_gpu_generic((cytnx_int32 *)out->Mem, (cytnx_int32 *)ten->Mem, ten->len);
      else
        cuMin_gpu_generic((cytnx_int32 *)out->Mem, (cytnx_int32 *)ten->Mem, ten->len);
    }
    void cuMaxMin_internal_u16(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &ten,
                               const cytnx_uint64 &Nelem, const char &type) {
      if (type == 'x')
        cuMax_gpu_generic((cytnx_uint16 *)out->Mem, (cytnx_uint16 *)ten->Mem, ten->len);
      else
        cuMin_gpu_generic((cytnx_uint16 *)out->Mem, (cytnx_uint16 *)ten->Mem, ten->len);
    }
    void cuMaxMin_internal_i16(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &ten,
                               const cytnx_uint64 &Nelem, const char &type) {
      if (type == 'x')
        cuMax_gpu_generic((cytnx_int16 *)out->Mem, (cytnx_int16 *)ten->Mem, ten->len);
      else
        cuMin_gpu_generic((cytnx_int16 *)out->Mem, (cytnx_int16 *)ten->Mem, ten->len);
    }
    void cuMaxMin_internal_b(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten,
                             const cytnx_uint64 &Nelem, const char &type) {
      if (type == 'x')
        cuMax_gpu_generic((cytnx_bool *)out->Mem, (cytnx_bool *)ten->Mem, ten->len);
      else
        cuMin_gpu_generic((cytnx_bool *)out->Mem, (cytnx_bool *)ten->Mem, ten->len);
    }

  }  // namespace linalg_internal
}  // namespace cytnx
