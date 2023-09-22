#include "cuMatvec_internal.hpp"
#include "cytnx_error.hpp"
#include "Type.hpp"
#include "backend/lapack_wrapper.hpp"

namespace cytnx {

  namespace linalg_internal {

    template <typename UniType>
    __global__ void cuMatVec_kernel(UniType *out, const UniType *inl, const UniType *inr,
                                    cytnx_int64 Ml, cytnx_int64 Nr) {
      UniType tmp = 0;
      cytnx_uint64 sid = blockIdx.x * blockDim.x + threadIdx.x;
      if (sid < cytnx_uint64(Ml)) {
        for (cytnx_int64 c = 0; c < Nr; c++) {
          tmp += inl[sid * Nr + c] * inr[c];
        }
        out[sid] = tmp;
      }
    }

    /// cuMatvec
    void cuMatvec_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                              const boost::intrusive_ptr<Storage_base> &inl,
                              const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                              const cytnx_int64 &Nr) {
      // create handles:
      cublasHandle_t cublasH = NULL;
      checkCudaErrors(cublasCreate(&cublasH));
      cytnx_complex128 alpha = cytnx_complex128(1, 0), beta = cytnx_complex128(0, 0);

      cuDoubleComplex *_out = (cuDoubleComplex *)out->Mem;
      cuDoubleComplex *_inl = (cuDoubleComplex *)inl->Mem;
      cuDoubleComplex *_inr = (cuDoubleComplex *)inr->Mem;

      // query working space :
      cytnx_int32 blsMl = Ml, blsNr = Nr;
      // checkCudaErrors(cublasZgemv(cublasH, CUBLAS_OP_T, blsMl, blsNr, (cuDoubleComplex *)&alpha,
      //                             _inl, blsMl, _inr, 1, (cuDoubleComplex *)&beta, _out, 1));
      checkCudaErrors(cublasZgemv(cublasH, CUBLAS_OP_T, blsNr, blsMl, (cuDoubleComplex *)&alpha,
                                  _inl, blsMl, _inr, 1, (cuDoubleComplex *)&beta, _out, 1));

      cublasDestroy(cublasH);
    }
    void cuMatvec_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                              const boost::intrusive_ptr<Storage_base> &inl,
                              const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                              const cytnx_int64 &Nr) {
      // create handles:
      cublasHandle_t cublasH = NULL;
      checkCudaErrors(cublasCreate(&cublasH));
      cytnx_complex64 alpha = cytnx_complex64(1, 0), beta = cytnx_complex64(0, 0);

      cuFloatComplex *_out = (cuFloatComplex *)out->Mem;
      cuFloatComplex *_inl = (cuFloatComplex *)inl->Mem;
      cuFloatComplex *_inr = (cuFloatComplex *)inr->Mem;

      // query working space :
      cytnx_int32 blsMl = Ml, blsNr = Nr;
      // checkCudaErrors(cublasCgemv(cublasH, CUBLAS_OP_T, blsMl, blsNr, (cuFloatComplex *)&alpha,
      //                             _inl, blsMl, _inr, 1, (cuFloatComplex *)&beta, _out, 1));
      checkCudaErrors(cublasCgemv(cublasH, CUBLAS_OP_T, blsNr, blsMl, (cuFloatComplex *)&alpha,
                                  _inl, blsMl, _inr, 1, (cuFloatComplex *)&beta, _out, 1));

      cublasDestroy(cublasH);
    }

    void cuMatvec_internal_d(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &inl,
                             const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                             const cytnx_int64 &Nr) {
      // create handles:
      cublasHandle_t cublasH = NULL;
      checkCudaErrors(cublasCreate(&cublasH));
      cytnx_double alpha = 1, beta = 0;

      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_inl = (cytnx_double *)inl->Mem;
      cytnx_double *_inr = (cytnx_double *)inr->Mem;

      // query working space :
      cytnx_int32 blsMl = Ml, blsNr = Nr;
      // checkCudaErrors(cublasDgemv(cublasH, CUBLAS_OP_T, blsMl, blsNr, &alpha, _inl, blsMl, _inr,
      // 1, &beta, _out, 1));
      checkCudaErrors(cublasDgemv(cublasH, CUBLAS_OP_T, blsNr, blsMl, &alpha, _inl, blsNr, _inr, 1,
                                  &beta, _out, 1));

      cublasDestroy(cublasH);
    }
    void cuMatvec_internal_f(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &inl,
                             const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                             const cytnx_int64 &Nr) {
      // create handles:
      cublasHandle_t cublasH = NULL;
      checkCudaErrors(cublasCreate(&cublasH));
      cytnx_float alpha = 1, beta = 0;

      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_float *_inl = (cytnx_float *)inl->Mem;
      cytnx_float *_inr = (cytnx_float *)inr->Mem;

      // query working space :
      cytnx_int32 blsMl = Ml, blsNr = Nr;
      // checkCudaErrors(cublasSgemv(cublasH, CUBLAS_OP_T, blsMl, blsNr, &alpha, _inr, blsMl, _inl,
      // 1,
      //                             &beta, _out, 1));
      checkCudaErrors(cublasSgemv(cublasH, CUBLAS_OP_T, blsNr, blsMl, &alpha, _inr, blsMl, _inl, 1,
                                  &beta, _out, 1));

      cublasDestroy(cublasH);
    }
    void cuMatvec_internal_i64(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &inl,
                               const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                               const cytnx_int64 &Nr) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_int64 *_inl = (cytnx_int64 *)inl->Mem;
      cytnx_int64 *_inr = (cytnx_int64 *)inr->Mem;

      cytnx_uint64 Nblocks = (cytnx_uint64(Ml)) / 512;
      if ((cytnx_uint64(Ml)) % 512) Nblocks += 1;

      cuMatVec_kernel<<<Nblocks, 512>>>(_out, _inl, _inr, Ml, Nr);
    }
    void cuMatvec_internal_u64(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &inl,
                               const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                               const cytnx_int64 &Nr) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->Mem;
      cytnx_uint64 *_inl = (cytnx_uint64 *)inl->Mem;
      cytnx_uint64 *_inr = (cytnx_uint64 *)inr->Mem;

      cytnx_uint64 Nblocks = (cytnx_uint64(Ml)) / 512;
      if ((cytnx_uint64(Ml)) % 512) Nblocks += 1;

      cuMatVec_kernel<<<Nblocks, 512>>>(_out, _inl, _inr, Ml, Nr);
    }
    void cuMatvec_internal_i32(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &inl,
                               const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                               const cytnx_int64 &Nr) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_int64 *_inl = (cytnx_int64 *)inl->Mem;
      cytnx_int64 *_inr = (cytnx_int64 *)inr->Mem;

      cytnx_uint64 Nblocks = (cytnx_uint64(Ml)) / 512;
      if ((cytnx_uint64(Ml)) % 512) Nblocks += 1;

      cuMatVec_kernel<<<Nblocks, 512>>>(_out, _inl, _inr, Ml, Nr);
    }
    void cuMatvec_internal_u32(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &inl,
                               const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                               const cytnx_int64 &Nr) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->Mem;
      cytnx_uint64 *_inl = (cytnx_uint64 *)inl->Mem;
      cytnx_uint64 *_inr = (cytnx_uint64 *)inr->Mem;

      cytnx_uint64 Nblocks = (cytnx_uint64(Ml)) / 512;
      if ((cytnx_uint64(Ml)) % 512) Nblocks += 1;

      cuMatVec_kernel<<<Nblocks, 512>>>(_out, _inl, _inr, Ml, Nr);
    }
    void cuMatvec_internal_i16(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &inl,
                               const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                               const cytnx_int64 &Nr) {
      cytnx_int16 *_out = (cytnx_int16 *)out->Mem;
      cytnx_int16 *_inl = (cytnx_int16 *)inl->Mem;
      cytnx_int16 *_inr = (cytnx_int16 *)inr->Mem;

      cytnx_uint64 Nblocks = (cytnx_uint64(Ml)) / 512;
      if ((cytnx_uint64(Ml)) % 512) Nblocks += 1;

      cuMatVec_kernel<<<Nblocks, 512>>>(_out, _inl, _inr, Ml, Nr);
    }
    void cuMatvec_internal_u16(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &inl,
                               const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                               const cytnx_int64 &Nr) {
      cytnx_uint16 *_out = (cytnx_uint16 *)out->Mem;
      cytnx_uint16 *_inl = (cytnx_uint16 *)inl->Mem;
      cytnx_uint16 *_inr = (cytnx_uint16 *)inr->Mem;

      cytnx_uint64 Nblocks = (cytnx_uint64(Ml)) / 512;
      if ((cytnx_uint64(Ml)) % 512) Nblocks += 1;

      cuMatVec_kernel<<<Nblocks, 512>>>(_out, _inl, _inr, Ml, Nr);
    }
    void cuMatvec_internal_b(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &inl,
                             const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                             const cytnx_int64 &Nr) {
      cytnx_bool *_out = (cytnx_bool *)out->Mem;
      cytnx_bool *_inl = (cytnx_bool *)inl->Mem;
      cytnx_bool *_inr = (cytnx_bool *)inr->Mem;

      cytnx_uint64 Nblocks = (cytnx_uint64(Ml)) / 512;
      if ((cytnx_uint64(Ml)) % 512) Nblocks += 1;

      cuMatVec_kernel<<<Nblocks, 512>>>(_out, _inl, _inr, Ml, Nr);
    }

  }  // namespace linalg_internal
}  // namespace cytnx
