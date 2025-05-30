#include "Gemm_internal.hpp"
#include "cytnx_error.hpp"
#include "backend/lapack_wrapper.hpp"

namespace cytnx {
  namespace linalg_internal {

    void Gemm_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &inl,
                          const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                          const cytnx_int64 &Comm, const cytnx_int64 &Nr, const Scalar &a,
                          const Scalar &b) {
      cytnx_complex128 *_out = (cytnx_complex128 *)out->data();
      cytnx_complex128 *_inl = (cytnx_complex128 *)inl->data();
      cytnx_complex128 *_inr = (cytnx_complex128 *)inr->data();

      cytnx_complex128 alpha = complex128(a), beta = complex128(b);
      blas_int blsMl = Ml, blsNr = Nr, blsComm = Comm;
      zgemm((char *)"N", (char *)"N", &blsNr, &blsMl, &blsComm, &alpha, _inr, &blsNr, _inl,
            &blsComm, &beta, _out, &blsNr);
    }

    void Gemm_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &inl,
                          const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                          const cytnx_int64 &Comm, const cytnx_int64 &Nr, const Scalar &a,
                          const Scalar &b) {
      cytnx_complex64 *_out = (cytnx_complex64 *)out->data();
      cytnx_complex64 *_inl = (cytnx_complex64 *)inl->data();
      cytnx_complex64 *_inr = (cytnx_complex64 *)inr->data();

      cytnx_complex64 alpha = complex64(a), beta = complex64(b);
      blas_int blsMl = Ml, blsNr = Nr, blsComm = Comm;
      cgemm((char *)"N", (char *)"N", &blsNr, &blsMl, &blsComm, &alpha, _inr, &blsNr, _inl,
            &blsComm, &beta, _out, &blsNr);
    }

    void Gemm_internal_d(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &inl,
                         const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                         const cytnx_int64 &Comm, const cytnx_int64 &Nr, const Scalar &a,
                         const Scalar &b) {
      cytnx_double *_out = (cytnx_double *)out->data();
      cytnx_double *_inl = (cytnx_double *)inl->data();
      cytnx_double *_inr = (cytnx_double *)inr->data();

      cytnx_double alpha = double(a), beta = double(b);
      blas_int blsMl = Ml, blsNr = Nr, blsComm = Comm;
      dgemm((char *)"N", (char *)"N", &blsNr, &blsMl, &blsComm, &alpha, _inr, &blsNr, _inl,
            &blsComm, &beta, _out, &blsNr);
    }

    void Gemm_internal_f(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &inl,
                         const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                         const cytnx_int64 &Comm, const cytnx_int64 &Nr, const Scalar &a,
                         const Scalar &b) {
      cytnx_float *_out = (cytnx_float *)out->data();
      cytnx_float *_inl = (cytnx_float *)inl->data();
      cytnx_float *_inr = (cytnx_float *)inr->data();

      cytnx_float alpha = float(a), beta = float(b);
      blas_int blsMl = Ml, blsNr = Nr, blsComm = Comm;
      sgemm((char *)"N", (char *)"N", &blsNr, &blsMl, &blsComm, &alpha, _inr, &blsNr, _inl,
            &blsComm, &beta, _out, &blsNr);
    }

  }  // namespace linalg_internal

};  // namespace cytnx
