#include "Matmul_internal.hpp"
#include "cytnx_error.hpp"
#include "backend/lapack_wrapper.hpp"

namespace cytnx {
  namespace linalg_internal {

    template <class T1>
    void Matmul_driver(T1 *out, const T1 *inl, const T1 *inr, const cytnx_int64 &Ml,
                       const cytnx_int64 &Comm, const cytnx_int64 &Nr) {
      for (cytnx_uint64 n = 0; n < cytnx_uint64(Ml) * Nr; n++) {
        cytnx_int64 i = n % Nr;
        cytnx_int64 j = n / Nr;
        out[j * Nr + i] = 0;
        for (cytnx_int64 c = 0; c < Comm; c++) {
          // std::cout << inl[j*Comm+c] << " " << inr[c*Nr+i] << std::endl;
          out[j * Nr + i] += inl[j * Comm + c] * inr[c * Nr + i];
        }
      }
    }

    void Matmul_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &inl,
                            const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                            const cytnx_int64 &Comm, const cytnx_int64 &Nr) {
      cytnx_complex128 *_out = (cytnx_complex128 *)out->data();
      cytnx_complex128 *_inl = (cytnx_complex128 *)inl->data();
      cytnx_complex128 *_inr = (cytnx_complex128 *)inr->data();

      cytnx_complex128 alpha = cytnx_complex128(1, 0), beta = cytnx_complex128(0, 0);
      blas_int blsMl = Ml, blsNr = Nr, blsComm = Comm;
      zgemm((char *)"N", (char *)"N", &blsNr, &blsMl, &blsComm, &alpha, _inr, &blsNr, _inl,
            &blsComm, &beta, _out, &blsNr);
    }

    void Matmul_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &inl,
                            const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                            const cytnx_int64 &Comm, const cytnx_int64 &Nr) {
      cytnx_complex64 *_out = (cytnx_complex64 *)out->data();
      cytnx_complex64 *_inl = (cytnx_complex64 *)inl->data();
      cytnx_complex64 *_inr = (cytnx_complex64 *)inr->data();

      cytnx_complex64 alpha = cytnx_complex64(1, 0), beta = cytnx_complex64(0, 0);
      blas_int blsMl = Ml, blsNr = Nr, blsComm = Comm;
      cgemm((char *)"N", (char *)"N", &blsNr, &blsMl, &blsComm, &alpha, _inr, &blsNr, _inl,
            &blsComm, &beta, _out, &blsNr);
    }

    void Matmul_internal_d(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &inl,
                           const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                           const cytnx_int64 &Comm, const cytnx_int64 &Nr) {
      cytnx_double *_out = (cytnx_double *)out->data();
      cytnx_double *_inl = (cytnx_double *)inl->data();
      cytnx_double *_inr = (cytnx_double *)inr->data();

      cytnx_double alpha = 1, beta = 0;
      blas_int blsMl = Ml, blsNr = Nr, blsComm = Comm;
      dgemm((char *)"N", (char *)"N", &blsNr, &blsMl, &blsComm, &alpha, _inr, &blsNr, _inl,
            &blsComm, &beta, _out, &blsNr);
    }

    void Matmul_internal_f(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &inl,
                           const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                           const cytnx_int64 &Comm, const cytnx_int64 &Nr) {
      cytnx_float *_out = (cytnx_float *)out->data();
      cytnx_float *_inl = (cytnx_float *)inl->data();
      cytnx_float *_inr = (cytnx_float *)inr->data();

      cytnx_float alpha = 1, beta = 0;
      blas_int blsMl = Ml, blsNr = Nr, blsComm = Comm;
      sgemm((char *)"N", (char *)"N", &blsNr, &blsMl, &blsComm, &alpha, _inr, &blsNr, _inl,
            &blsComm, &beta, _out, &blsNr);
    }

    void Matmul_internal_i64(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &inl,
                             const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                             const cytnx_int64 &Comm, const cytnx_int64 &Nr) {
      cytnx_int64 *_out = (cytnx_int64 *)out->data();
      cytnx_int64 *_inl = (cytnx_int64 *)inl->data();
      cytnx_int64 *_inr = (cytnx_int64 *)inr->data();
      Matmul_driver(_out, _inl, _inr, Ml, Comm, Nr);
    }

    void Matmul_internal_u64(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &inl,
                             const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                             const cytnx_int64 &Comm, const cytnx_int64 &Nr) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->data();
      cytnx_uint64 *_inl = (cytnx_uint64 *)inl->data();
      cytnx_uint64 *_inr = (cytnx_uint64 *)inr->data();
      Matmul_driver(_out, _inl, _inr, Ml, Comm, Nr);
    }

    void Matmul_internal_i32(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &inl,
                             const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                             const cytnx_int64 &Comm, const cytnx_int64 &Nr) {
      cytnx_int64 *_out = (cytnx_int64 *)out->data();
      cytnx_int64 *_inl = (cytnx_int64 *)inl->data();
      cytnx_int64 *_inr = (cytnx_int64 *)inr->data();
      Matmul_driver(_out, _inl, _inr, Ml, Comm, Nr);
    }

    void Matmul_internal_u32(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &inl,
                             const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                             const cytnx_int64 &Comm, const cytnx_int64 &Nr) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->data();
      cytnx_uint64 *_inl = (cytnx_uint64 *)inl->data();
      cytnx_uint64 *_inr = (cytnx_uint64 *)inr->data();
      Matmul_driver(_out, _inl, _inr, Ml, Comm, Nr);
    }
    void Matmul_internal_i16(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &inl,
                             const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                             const cytnx_int64 &Comm, const cytnx_int64 &Nr) {
      cytnx_int16 *_out = (cytnx_int16 *)out->data();
      cytnx_int16 *_inl = (cytnx_int16 *)inl->data();
      cytnx_int16 *_inr = (cytnx_int16 *)inr->data();
      Matmul_driver(_out, _inl, _inr, Ml, Comm, Nr);
    }

    void Matmul_internal_u16(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &inl,
                             const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                             const cytnx_int64 &Comm, const cytnx_int64 &Nr) {
      cytnx_uint16 *_out = (cytnx_uint16 *)out->data();
      cytnx_uint16 *_inl = (cytnx_uint16 *)inl->data();
      cytnx_uint16 *_inr = (cytnx_uint16 *)inr->data();
      Matmul_driver(_out, _inl, _inr, Ml, Comm, Nr);
    }
    void Matmul_internal_b(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &inl,
                           const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                           const cytnx_int64 &Comm, const cytnx_int64 &Nr) {
      cytnx_bool *_out = (cytnx_bool *)out->data();
      cytnx_bool *_inl = (cytnx_bool *)inl->data();
      cytnx_bool *_inr = (cytnx_bool *)inr->data();
      Matmul_driver(_out, _inl, _inr, Ml, Comm, Nr);
    }

  }  // namespace linalg_internal

};  // namespace cytnx
