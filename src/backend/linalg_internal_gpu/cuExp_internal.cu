#include "cuExp_internal.hpp"
#include "../utils_internal_interface.hpp"

// #ifdef UNI_OMP
//     #include <omp.h>
// #endif

namespace cytnx {

  namespace linalg_internal {
    __global__ void cuExp_internal_kernel_d(cytnx_double *out, const cytnx_double *ten,
                                            const cytnx_uint64 Nelem) {
      cytnx_double tmp;
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        tmp = ten[blockIdx.x * blockDim.x + threadIdx.x];
        out[blockIdx.x * blockDim.x + threadIdx.x] = exp(tmp);
      }
      __syncthreads();
    }
    __global__ void cuExp_internal_kernel_f(cytnx_float *out, const cytnx_float *ten,
                                            const cytnx_uint64 Nelem) {
      cytnx_float tmp;
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        tmp = ten[blockIdx.x * blockDim.x + threadIdx.x];
        out[blockIdx.x * blockDim.x + threadIdx.x] = expf(tmp);
      }
      __syncthreads();
    }

    __global__ void cuExp_internal_kernel_cd(cuDoubleComplex *out, const cuDoubleComplex *ten,
                                             const cytnx_uint64 Nelem) {
      cuDoubleComplex tmp;
      double a;
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        a = exp(ten[blockIdx.x * blockDim.x + threadIdx.x].x);
        sincos(ten[blockIdx.x * blockDim.x + threadIdx.x].y, &tmp.y, &tmp.x);
        tmp.x *= a;
        tmp.y *= a;

        out[blockIdx.x * blockDim.x + threadIdx.x] = tmp;
      }
      __syncthreads();
    }

    __global__ void cuExp_internal_kernel_cf(cuFloatComplex *out, const cuFloatComplex *ten,
                                             const cytnx_uint64 Nelem) {
      cuFloatComplex tmp;
      float a;
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        a = expf(ten[blockIdx.x * blockDim.x + threadIdx.x].x);
        sincosf(ten[blockIdx.x * blockDim.x + threadIdx.x].y, &tmp.y, &tmp.x);
        tmp.x *= a;
        tmp.y *= a;

        out[blockIdx.x * blockDim.x + threadIdx.x] = tmp;
      }
      __syncthreads();
    }

  }  // namespace linalg_internal

}  // namespace cytnx

namespace cytnx {
  namespace linalg_internal {

    void cuExp_internal_d(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten,
                          const cytnx_uint64 &Nelem) {
      cytnx_uint32 NBlocks = Nelem / 512;
      if (Nelem % 512) NBlocks += 1;
      cuExp_internal_kernel_d<<<NBlocks, 512>>>((cytnx_double *)out->Mem, (cytnx_double *)ten->Mem,
                                                Nelem);
    }

    void cuExp_internal_f(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten,
                          const cytnx_uint64 &Nelem) {
      cytnx_uint32 NBlocks = Nelem / 512;
      if (Nelem % 512) NBlocks += 1;
      cuExp_internal_kernel_f<<<NBlocks, 512>>>((cytnx_float *)out->Mem, (cytnx_float *)ten->Mem,
                                                Nelem);
    }

    void cuExp_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten,
                           const cytnx_uint64 &Nelem) {
      cytnx_uint32 NBlocks = Nelem / 256;
      if (Nelem % 256) NBlocks += 1;
      cuExp_internal_kernel_cd<<<NBlocks, 256>>>((cuDoubleComplex *)out->Mem,
                                                 (cuDoubleComplex *)ten->Mem, Nelem);
    }

    void cuExp_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten,
                           const cytnx_uint64 &Nelem) {
      cytnx_uint32 NBlocks = Nelem / 256;
      if (Nelem % 256) NBlocks += 1;
      cuExp_internal_kernel_cf<<<NBlocks, 256>>>((cuFloatComplex *)out->Mem,
                                                 (cuFloatComplex *)ten->Mem, Nelem);
    }

  }  // namespace linalg_internal

}  // namespace cytnx
