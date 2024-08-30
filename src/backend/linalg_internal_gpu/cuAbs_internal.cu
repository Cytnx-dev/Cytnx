#include "cuAbs_internal.hpp"
#include "../utils_internal_interface.hpp"

// #include "cytnx_error.hpp"
// #include "utils/backend/lapack_wrapper.hpp"

#ifdef UNI_OMP
  #include <omp.h>
#endif

namespace cytnx {
  namespace linalg_internal {

    __global__ void cuAbs_kernel(cytnx_double *out, const cuDoubleComplex *ptr,
                                 const cytnx_uint64 Nelem) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCabs(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }
    __global__ void cuAbs_kernel(cytnx_float *out, const cuFloatComplex *ptr,
                                 const cytnx_uint64 Nelem) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          cuCabsf(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }
    __global__ void cuAbs_kernel(cytnx_double *out, const double *ptr, const cytnx_uint64 Nelem) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          fabs(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }
    __global__ void cuAbs_kernel(cytnx_float *out, const float *ptr, const cytnx_uint64 Nelem) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          fabsf(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }
    // [NOTE] no unsigned version!
    __global__ void cuAbs_kernel(cytnx_int64 *out, const cytnx_int64 *ptr,
                                 const cytnx_uint64 Nelem) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          llabs(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }
    __global__ void cuAbs_kernel(cytnx_int32 *out, const cytnx_int32 *ptr,
                                 const cytnx_uint64 Nelem) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          labs(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }
    __global__ void cuAbs_kernel(cytnx_int16 *out, const cytnx_int16 *ptr,
                                 const cytnx_uint64 Nelem) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          labs(ptr[blockIdx.x * blockDim.x + threadIdx.x]);
      }
      __syncthreads();
    }

    void cuAbs_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten,
                           const cytnx_uint64 &Nelem) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cuDoubleComplex *_ten = (cuDoubleComplex *)ten->Mem;

      cytnx_uint32 NBlocks = Nelem / 512;
      if (Nelem % 512) NBlocks += 1;

      cuAbs_kernel<<<NBlocks, 512>>>(_out, _ten, Nelem);
    }

    void cuAbs_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten,
                           const cytnx_uint64 &Nelem) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cuFloatComplex *_ten = (cuFloatComplex *)ten->Mem;

      cytnx_uint32 NBlocks = Nelem / 512;
      if (Nelem % 512) NBlocks += 1;

      cuAbs_kernel<<<NBlocks, 512>>>(_out, _ten, Nelem);
    }

    void cuAbs_internal_d(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten,
                          const cytnx_uint64 &Nelem) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_ten = (cytnx_double *)ten->Mem;

      cytnx_uint32 NBlocks = Nelem / 512;
      if (Nelem % 512) NBlocks += 1;

      cuAbs_kernel<<<NBlocks, 512>>>(_out, _ten, Nelem);
    }

    void cuAbs_internal_f(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten,
                          const cytnx_uint64 &Nelem) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_float *_ten = (cytnx_float *)ten->Mem;

      cytnx_uint32 NBlocks = Nelem / 512;
      if (Nelem % 512) NBlocks += 1;

      cuAbs_kernel<<<NBlocks, 512>>>(_out, _ten, Nelem);
    }

    void cuAbs_internal_i64(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &ten,
                            const cytnx_uint64 &Nelem) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_int64 *_ten = (cytnx_int64 *)ten->Mem;

      cytnx_uint32 NBlocks = Nelem / 512;
      if (Nelem % 512) NBlocks += 1;

      cuAbs_kernel<<<NBlocks, 512>>>(_out, _ten, Nelem);
    }

    void cuAbs_internal_i32(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &ten,
                            const cytnx_uint64 &Nelem) {
      cytnx_int32 *_out = (cytnx_int32 *)out->Mem;
      cytnx_int32 *_ten = (cytnx_int32 *)ten->Mem;

      cytnx_uint32 NBlocks = Nelem / 512;
      if (Nelem % 512) NBlocks += 1;

      cuAbs_kernel<<<NBlocks, 512>>>(_out, _ten, Nelem);
    }

    void cuAbs_internal_i16(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &ten,
                            const cytnx_uint64 &Nelem) {
      cytnx_int16 *_out = (cytnx_int16 *)out->Mem;
      cytnx_int16 *_ten = (cytnx_int16 *)ten->Mem;

      cytnx_uint32 NBlocks = Nelem / 512;
      if (Nelem % 512) NBlocks += 1;

      cuAbs_kernel<<<NBlocks, 512>>>(_out, _ten, Nelem);
    }

    void cuAbs_internal_pass(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten,
                             const cytnx_uint64 &Nelem) {
      cytnx_error_msg(
        true, "[ERROR][Abs_internal_gpu] Called pass, which should not be called by frontend.%s",
        "\n");
    }
  }  // namespace linalg_internal
}  // namespace cytnx
