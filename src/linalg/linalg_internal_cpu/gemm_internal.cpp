#include "gemm_internal.hpp"
#include "cytnx_error.hpp"
#include "lapack_wrapper.hpp"

#ifdef UNI_OMP
  #include <omp.h>
#endif

namespace cytnx {
  namespace linalg_internal {

    void gemm_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &inl,
                            const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                            const cytnx_int64 &Comm, const cytnx_int64 &Nr) {
      cytnx_complex128 *_out = (cytnx_complex128 *)out->Mem;
      cytnx_complex128 *_inl = (cytnx_complex128 *)inl->Mem;
      cytnx_complex128 *_inr = (cytnx_complex128 *)inr->Mem;

      cytnx_complex128 alpha = cytnx_complex128(1, 0), beta = cytnx_complex128(0, 0);
      blas_int blsMl = Ml, blsNr = Nr, blsComm = Comm;
      zgemm((char *)"N", (char *)"N", &blsNr, &blsMl, &blsComm, &alpha, _inr, &blsNr, _inl,
            &blsComm, &beta, _out, &blsNr);
    }

    void gemm_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &inl,
                            const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                            const cytnx_int64 &Comm, const cytnx_int64 &Nr) {
      cytnx_complex64 *_out = (cytnx_complex64 *)out->Mem;
      cytnx_complex64 *_inl = (cytnx_complex64 *)inl->Mem;
      cytnx_complex64 *_inr = (cytnx_complex64 *)inr->Mem;

      cytnx_complex64 alpha = cytnx_complex64(1, 0), beta = cytnx_complex64(0, 0);
      blas_int blsMl = Ml, blsNr = Nr, blsComm = Comm;
      cgemm((char *)"N", (char *)"N", &blsNr, &blsMl, &blsComm, &alpha, _inr, &blsNr, _inl,
            &blsComm, &beta, _out, &blsNr);
    }

    void gemm_internal_d(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &inl,
                           const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                           const cytnx_int64 &Comm, const cytnx_int64 &Nr) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_inl = (cytnx_double *)inl->Mem;
      cytnx_double *_inr = (cytnx_double *)inr->Mem;

      cytnx_double alpha = 1, beta = 0;
      blas_int blsMl = Ml, blsNr = Nr, blsComm = Comm;
      dgemm((char *)"N", (char *)"N", &blsNr, &blsMl, &blsComm, &alpha, _inr, &blsNr, _inl,
            &blsComm, &beta, _out, &blsNr);
    }

    void gemm_internal_f(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &inl,
                           const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                           const cytnx_int64 &Comm, const cytnx_int64 &Nr) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_float *_inl = (cytnx_float *)inl->Mem;
      cytnx_float *_inr = (cytnx_float *)inr->Mem;

      cytnx_float alpha = 1, beta = 0;
      blas_int blsMl = Ml, blsNr = Nr, blsComm = Comm;
      sgemm((char *)"N", (char *)"N", &blsNr, &blsMl, &blsComm, &alpha, _inr, &blsNr, _inl,
            &blsComm, &beta, _out, &blsNr);
    }

  }  // namespace linalg_internal

};  // namespace cytnx
