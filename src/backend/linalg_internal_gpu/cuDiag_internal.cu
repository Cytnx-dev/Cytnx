#include "cuDiag_internal.hpp"
#include "backend/utils_internal_interface.hpp"

namespace cytnx {

  namespace linalg_internal {

    template <class T>
    __global__ void cuDiag_internal_kernel(T *out, const T *ten, const cytnx_uint64 L) {
      if (blockIdx.x * blockDim.x + threadIdx.x < L) {
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          ten[(blockIdx.x * blockDim.x + threadIdx.x) * L + blockIdx.x * blockDim.x + threadIdx.x];
      }
      __syncthreads();
    }

    template <class T>
    __global__ void cuDiag_internal_getdiag_kernel(T *out, const T *ten, const cytnx_uint64 L) {
      if (blockIdx.x * blockDim.x + threadIdx.x < L) {
        out[(blockIdx.x * blockDim.x + threadIdx.x) * L + blockIdx.x * blockDim.x + threadIdx.x] =
          ten[blockIdx.x * blockDim.x + threadIdx.x];
      }
      __syncthreads();
    }

  }  // namespace linalg_internal

}  // namespace cytnx

namespace cytnx {
  namespace linalg_internal {

    void cuDiag_internal_b(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                           const bool &isrank2) {
      cytnx_uint32 NBlocks = L / 512;
      if (L % 512) NBlocks += 1;
      if (isrank2)
        cuDiag_internal_kernel<<<NBlocks, 512>>>((cytnx_bool *)out->data(),
                                                 (cytnx_bool *)ten->data(), L);
      else
        cuDiag_internal_getdiag_kernel<<<NBlocks, 512>>>((cytnx_bool *)out->data(),
                                                         (cytnx_bool *)ten->data(), L);
    }

    void cuDiag_internal_i16(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                             const bool &isrank2) {
      cytnx_uint32 NBlocks = L / 512;
      if (L % 512) NBlocks += 1;
      if (isrank2)
        cuDiag_internal_kernel<<<NBlocks, 512>>>((cytnx_int16 *)out->data(),
                                                 (cytnx_int16 *)ten->data(), L);
      else
        cuDiag_internal_getdiag_kernel<<<NBlocks, 512>>>((cytnx_int16 *)out->data(),
                                                         (cytnx_int16 *)ten->data(), L);
    }

    void cuDiag_internal_u16(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                             const bool &isrank2) {
      cytnx_uint32 NBlocks = L / 512;
      if (L % 512) NBlocks += 1;
      if (isrank2)
        cuDiag_internal_kernel<<<NBlocks, 512>>>((cytnx_uint16 *)out->data(),
                                                 (cytnx_uint16 *)ten->data(), L);
      else
        cuDiag_internal_getdiag_kernel<<<NBlocks, 512>>>((cytnx_uint16 *)out->data(),
                                                         (cytnx_uint16 *)ten->data(), L);
    }

    void cuDiag_internal_i32(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                             const bool &isrank2) {
      cytnx_uint32 NBlocks = L / 512;
      if (L % 512) NBlocks += 1;
      if (isrank2)
        cuDiag_internal_kernel<<<NBlocks, 512>>>((cytnx_int32 *)out->data(),
                                                 (cytnx_int32 *)ten->data(), L);
      else
        cuDiag_internal_getdiag_kernel<<<NBlocks, 512>>>((cytnx_int32 *)out->data(),
                                                         (cytnx_int32 *)ten->data(), L);
    }

    void cuDiag_internal_u32(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                             const bool &isrank2) {
      cytnx_uint32 NBlocks = L / 512;
      if (L % 512) NBlocks += 1;
      if (isrank2)
        cuDiag_internal_kernel<<<NBlocks, 512>>>((cytnx_uint32 *)out->data(),
                                                 (cytnx_uint32 *)ten->data(), L);
      else
        cuDiag_internal_getdiag_kernel<<<NBlocks, 512>>>((cytnx_uint32 *)out->data(),
                                                         (cytnx_uint32 *)ten->data(), L);
    }

    void cuDiag_internal_i64(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                             const bool &isrank2) {
      cytnx_uint32 NBlocks = L / 512;
      if (L % 512) NBlocks += 1;
      if (isrank2)
        cuDiag_internal_kernel<<<NBlocks, 512>>>((cytnx_int64 *)out->data(),
                                                 (cytnx_int64 *)ten->data(), L);
      else
        cuDiag_internal_getdiag_kernel<<<NBlocks, 512>>>((cytnx_int64 *)out->data(),
                                                         (cytnx_int64 *)ten->data(), L);
    }

    void cuDiag_internal_u64(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                             const bool &isrank2) {
      cytnx_uint32 NBlocks = L / 512;
      if (L % 512) NBlocks += 1;
      if (isrank2)
        cuDiag_internal_kernel<<<NBlocks, 512>>>((cytnx_uint64 *)out->data(),
                                                 (cytnx_uint64 *)ten->data(), L);
      else
        cuDiag_internal_getdiag_kernel<<<NBlocks, 512>>>((cytnx_uint64 *)out->data(),
                                                         (cytnx_uint64 *)ten->data(), L);
    }

    void cuDiag_internal_d(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                           const bool &isrank2) {
      cytnx_uint32 NBlocks = L / 512;
      if (L % 512) NBlocks += 1;
      if (isrank2)
        cuDiag_internal_kernel<<<NBlocks, 512>>>((cytnx_double *)out->data(),
                                                 (cytnx_double *)ten->data(), L);
      else
        cuDiag_internal_getdiag_kernel<<<NBlocks, 512>>>((cytnx_double *)out->data(),
                                                         (cytnx_double *)ten->data(), L);
    }

    void cuDiag_internal_f(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                           const bool &isrank2) {
      cytnx_uint32 NBlocks = L / 512;
      if (L % 512) NBlocks += 1;
      if (isrank2)
        cuDiag_internal_kernel<<<NBlocks, 512>>>((cytnx_float *)out->data(),
                                                 (cytnx_float *)ten->data(), L);
      else
        cuDiag_internal_getdiag_kernel<<<NBlocks, 512>>>((cytnx_float *)out->data(),
                                                         (cytnx_float *)ten->data(), L);
    }

    void cuDiag_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                            const bool &isrank2) {
      cytnx_uint32 NBlocks = L / 256;
      if (L % 256) NBlocks += 1;
      if (isrank2)
        cuDiag_internal_kernel<<<NBlocks, 256>>>((cuDoubleComplex *)out->data(),
                                                 (cuDoubleComplex *)ten->data(), L);
      else
        cuDiag_internal_getdiag_kernel<<<NBlocks, 256>>>((cuDoubleComplex *)out->data(),
                                                         (cuDoubleComplex *)ten->data(), L);
    }

    void cuDiag_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                            const bool &isrank2) {
      cytnx_uint32 NBlocks = L / 256;
      if (L % 256) NBlocks += 1;
      if (isrank2)
        cuDiag_internal_kernel<<<NBlocks, 256>>>((cuFloatComplex *)out->data(),
                                                 (cuFloatComplex *)ten->data(), L);
      else
        cuDiag_internal_getdiag_kernel<<<NBlocks, 256>>>((cuFloatComplex *)out->data(),
                                                         (cuFloatComplex *)ten->data(), L);
    }

  }  // namespace linalg_internal

}  // namespace cytnx
