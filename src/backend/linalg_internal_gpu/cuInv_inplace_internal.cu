#include "cuInv_inplace_internal.hpp"
#include "backend/utils_internal_interface.hpp"

namespace cytnx {

  namespace linalg_internal {
    __global__ void cuInv_internal_kernel_d(cytnx_double *ten, const cytnx_uint64 Nelem,
                                            const double clip) {
      cytnx_double tmp;
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        tmp = ten[blockIdx.x * blockDim.x + threadIdx.x];
        ten[blockIdx.x * blockDim.x + threadIdx.x] = fabs(tmp) <= clip ? 0. : 1. / tmp;
      }
      __syncthreads();
    }
    __global__ void cuInv_internal_kernel_f(cytnx_float *ten, const cytnx_uint64 Nelem,
                                            const float clip) {
      cytnx_float tmp;
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        tmp = ten[blockIdx.x * blockDim.x + threadIdx.x];
        ten[blockIdx.x * blockDim.x + threadIdx.x] = fabsf(tmp) <= clip ? 0.0f : 1.0f / tmp;
      }
      __syncthreads();
    }

    __global__ void cuInv_internal_kernel_cd(cuDoubleComplex *ten, const cytnx_uint64 Nelem,
                                             const double clipsq) {
      cuDoubleComplex tmp;
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        tmp = ten[blockIdx.x * blockDim.x + threadIdx.x];
        ten[blockIdx.x * blockDim.x + threadIdx.x] = (tmp.x * tmp.x + tmp.y * tmp.y) <= clipsq
                                                       ? make_cuDoubleComplex(0., 0.)
                                                       : cuCdiv(make_cuDoubleComplex(1., 0.), tmp);
      }
      __syncthreads();
    }

    __global__ void cuInv_internal_kernel_cf(cuFloatComplex *ten, const cytnx_uint64 Nelem,
                                             const float clipsq) {
      cuFloatComplex tmp;
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        tmp = ten[blockIdx.x * blockDim.x + threadIdx.x];
        ten[blockIdx.x * blockDim.x + threadIdx.x] =
          (tmp.x * tmp.x + tmp.y * tmp.y) <= clipsq ? make_cuFloatComplex(0.0f, 0.0f)
                                                    : cuCdivf(make_cuFloatComplex(1.0f, 0.0f), tmp);
      }
      __syncthreads();
    }

  }  // namespace linalg_internal

}  // namespace cytnx

namespace cytnx {
  namespace linalg_internal {

    void cuInv_inplace_internal_d(boost::intrusive_ptr<Storage_base> &ten,
                                  const cytnx_uint64 &Nelem, const double &clip) {
      cytnx_uint32 NBlocks = Nelem / 512;
      if (Nelem % 512) NBlocks += 1;
      cuInv_internal_kernel_d<<<NBlocks, 512>>>((cytnx_double *)ten->data(), Nelem, clip);
    }

    void cuInv_inplace_internal_f(boost::intrusive_ptr<Storage_base> &ten,
                                  const cytnx_uint64 &Nelem, const double &clip) {
      cytnx_uint32 NBlocks = Nelem / 512;
      if (Nelem % 512) NBlocks += 1;
      cuInv_internal_kernel_f<<<NBlocks, 512>>>((cytnx_float *)ten->data(), Nelem, clip);
    }

    void cuInv_inplace_internal_cd(boost::intrusive_ptr<Storage_base> &ten,
                                   const cytnx_uint64 &Nelem, const double &clip) {
      cytnx_uint32 NBlocks = Nelem / 256;
      if (Nelem % 256) NBlocks += 1;
      const double clipsq = (clip < 0. ? -1. : clip * clip);  // because std::norm is squared norm
      cuInv_internal_kernel_cd<<<NBlocks, 256>>>((cuDoubleComplex *)ten->data(), Nelem, clipsq);
    }

    void cuInv_inplace_internal_cf(boost::intrusive_ptr<Storage_base> &ten,
                                   const cytnx_uint64 &Nelem, const double &clip) {
      cytnx_uint32 NBlocks = Nelem / 256;
      if (Nelem % 256) NBlocks += 1;
      const double clipsq = (clip < 0. ? -1. : clip * clip);  // because std::norm is squared norm
      cuInv_internal_kernel_cf<<<NBlocks, 256>>>((cuFloatComplex *)ten->data(), Nelem, clipsq);
    }

  }  // namespace linalg_internal

}  // namespace cytnx
