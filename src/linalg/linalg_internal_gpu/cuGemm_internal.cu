#include "cuGemm_internal.hpp"
#include "cytnx_error.hpp"
#include "Type.hpp"
#include "backend/lapack_wrapper.hpp"

namespace cytnx {

  namespace linalg_internal {
    void cuGemm_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &inl,
                            const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                            const cytnx_int64 &Comm, const cytnx_int64 &Nr, const Scalar &a,
                            const Scalar &b) {
      // create handles:
      cublasHandle_t cublasH = NULL;
      checkCudaErrors(cublasCreate(&cublasH));
      cytnx_complex128 alpha = complex128(a), beta = complex128(b);

      cuDoubleComplex *_out = (cuDoubleComplex *)out->Mem;
      cuDoubleComplex *_inl = (cuDoubleComplex *)inl->Mem;
      cuDoubleComplex *_inr = (cuDoubleComplex *)inr->Mem;

      // query working space :
      cytnx_int32 blsMl = Ml, blsNr = Nr, blsComm = Comm;
      checkCudaErrors(cublasZgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, blsNr, blsMl, blsComm,
                                  (cuDoubleComplex *)&alpha, _inr, blsNr, _inl, blsComm,
                                  (cuDoubleComplex *)&beta, _out, blsNr));

      cublasDestroy(cublasH);
    }
    void cuGemm_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &inl,
                            const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                            const cytnx_int64 &Comm, const cytnx_int64 &Nr, const Scalar &a,
                            const Scalar &b) {
      // create handles:
      cublasHandle_t cublasH = NULL;
      checkCudaErrors(cublasCreate(&cublasH));
      cytnx_complex64 alpha = complex64(a), beta = complex64(b);

      cuFloatComplex *_out = (cuFloatComplex *)out->Mem;
      cuFloatComplex *_inl = (cuFloatComplex *)inl->Mem;
      cuFloatComplex *_inr = (cuFloatComplex *)inr->Mem;

      // query working space :
      cytnx_int32 blsMl = Ml, blsNr = Nr, blsComm = Comm;
      checkCudaErrors(cublasCgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, blsNr, blsMl, blsComm,
                                  (cuFloatComplex *)&alpha, _inr, blsNr, _inl, blsComm,
                                  (cuFloatComplex *)&beta, _out, blsNr));

      cublasDestroy(cublasH);
    }

    void cuGemm_internal_d(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &inl,
                           const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                           const cytnx_int64 &Comm, const cytnx_int64 &Nr, const Scalar &a,
                           const Scalar &b) {
      // create handles:
      cublasHandle_t cublasH = NULL;
      checkCudaErrors(cublasCreate(&cublasH));
      cytnx_double alpha = double(a), beta = double(b);

      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_inl = (cytnx_double *)inl->Mem;
      cytnx_double *_inr = (cytnx_double *)inr->Mem;

      // query working space :
      cytnx_int32 blsMl = Ml, blsNr = Nr, blsComm = Comm;
      checkCudaErrors(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, blsNr, blsMl, blsComm, &alpha,
                                  _inr, blsNr, _inl, blsComm, &beta, _out, blsNr));

      cublasDestroy(cublasH);
    }
    void cuGemm_internal_f(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &inl,
                           const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                           const cytnx_int64 &Comm, const cytnx_int64 &Nr, const Scalar &a,
                           const Scalar &b) {
      // create handles:
      cublasHandle_t cublasH = NULL;
      checkCudaErrors(cublasCreate(&cublasH));
      cytnx_float alpha = float(a), beta = float(b);

      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_float *_inl = (cytnx_float *)inl->Mem;
      cytnx_float *_inr = (cytnx_float *)inr->Mem;

      // query working space :
      cytnx_int32 blsMl = Ml, blsNr = Nr, blsComm = Comm;
      checkCudaErrors(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, blsNr, blsMl, blsComm, &alpha,
                                  _inr, blsNr, _inl, blsComm, &beta, _out, blsNr));

      cublasDestroy(cublasH);
    }

  }  // namespace linalg_internal
}  // namespace cytnx
