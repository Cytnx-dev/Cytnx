#include "cuPow_internal.hpp"
#include "../utils_internal_interface.hpp"

// #ifdef UNI_OMP
//     #include <omp.h>
// #endif

namespace cytnx {

  namespace linalg_internal {
    __global__ void cuPow_internal_kernel_d(cytnx_double *out, const cytnx_double *ten,
                                            const cytnx_uint64 Nelem, const double p) {
      cytnx_double tmp;
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        tmp = ten[blockIdx.x * blockDim.x + threadIdx.x];
        out[blockIdx.x * blockDim.x + threadIdx.x] = pow(tmp, p);
      }
      __syncthreads();
    }
    __global__ void cuPow_internal_kernel_f(cytnx_float *out, const cytnx_float *ten,
                                            const cytnx_uint64 Nelem, const double p) {
      cytnx_float tmp;
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        tmp = ten[blockIdx.x * blockDim.x + threadIdx.x];
        out[blockIdx.x * blockDim.x + threadIdx.x] = powf(tmp, p);
      }
      __syncthreads();
    }

    __global__ void cuPow_internal_kernel_cd(cuDoubleComplex *out, const cuDoubleComplex *ten,
                                             const cytnx_uint64 Nelem, const double p) {
      cuDoubleComplex tmp;
      double a;
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        tmp = ten[blockIdx.x * blockDim.x + threadIdx.x];
        a = pow(cuCabs(tmp), p);
        tmp.x = a;
        tmp.y = a;
        a = atan2(tmp.y, tmp.x);
        tmp.x *= cos(p * a);
        tmp.y *= sin(p * a);
        out[blockIdx.x * blockDim.x + threadIdx.x] = tmp;
      }
      __syncthreads();
    }

    __global__ void cuPow_internal_kernel_cf(cuFloatComplex *out, const cuFloatComplex *ten,
                                             const cytnx_uint64 Nelem, const double p) {
      cuFloatComplex tmp;
      float a;
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        tmp = ten[blockIdx.x * blockDim.x + threadIdx.x];
        a = powf(cuCabsf(tmp), p);
        tmp.x = a;
        tmp.y = a;
        a = atan2f(tmp.y, tmp.x);
        tmp.x *= cosf(p * a);
        tmp.y *= sinf(p * a);
        out[blockIdx.x * blockDim.x + threadIdx.x] = tmp;
      }
      __syncthreads();
    }

  }  // namespace linalg_internal

}  // namespace cytnx

namespace cytnx {
  namespace linalg_internal {

    void cuPow_internal_d(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                          const cytnx_double &p) {
      cytnx_uint32 NBlocks = Nelem / 512;
      if (Nelem % 512) NBlocks += 1;
      cuPow_internal_kernel_d<<<NBlocks, 512>>>((cytnx_double *)out->Mem, (cytnx_double *)ten->Mem,
                                                Nelem, p);
    }

    void cuPow_internal_f(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                          const cytnx_double &p) {
      cytnx_uint32 NBlocks = Nelem / 512;
      if (Nelem % 512) NBlocks += 1;
      cuPow_internal_kernel_f<<<NBlocks, 512>>>((cytnx_float *)out->Mem, (cytnx_float *)ten->Mem,
                                                Nelem, p);
    }

    void cuPow_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                           const cytnx_double &p) {
      cytnx_uint32 NBlocks = Nelem / 256;
      if (Nelem % 256) NBlocks += 1;
      cuPow_internal_kernel_cd<<<NBlocks, 256>>>((cuDoubleComplex *)out->Mem,
                                                 (cuDoubleComplex *)ten->Mem, Nelem, p);
    }

    void cuPow_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                           const cytnx_double &p) {
      cytnx_uint32 NBlocks = Nelem / 256;
      if (Nelem % 256) NBlocks += 1;
      cuPow_internal_kernel_cf<<<NBlocks, 256>>>((cuFloatComplex *)out->Mem,
                                                 (cuFloatComplex *)ten->Mem, Nelem, p);
    }

  }  // namespace linalg_internal

}  // namespace cytnx
