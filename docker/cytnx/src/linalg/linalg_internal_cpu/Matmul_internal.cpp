#include "linalg/linalg_internal_cpu/Matmul_internal.hpp"
#include "cytnx_error.hpp"
#include "utils/lapack_wrapper.hpp"

#ifdef UNI_OMP
  #include <omp.h>
#endif

namespace cytnx {
  namespace linalg_internal {

    template <class T1>
    void Matmul_driver(T1 *out, const T1 *inl, const T1 *inr, const cytnx_int32 &Ml,
                       const cytnx_int32 &Comm, const cytnx_int32 &Nr) {
#ifdef UNI_OMP
  #pragma omp parallel for
#endif
      for (cytnx_uint64 n = 0; n < cytnx_uint64(Ml) * Nr; n++) {
        cytnx_int32 i = n % Nr;
        cytnx_int32 j = n / Nr;
        out[j * Nr + i] = 0;
        for (cytnx_int32 c = 0; c < Comm; c++) {
          // std::cout << inl[j*Comm+c] << " " << inr[c*Nr+i] << std::endl;
          out[j * Nr + i] += inl[j * Comm + c] * inr[c * Nr + i];
        }
      }
    }

    void Matmul_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &inl,
                            const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int32 &Ml,
                            const cytnx_int32 &Comm, const cytnx_int32 &Nr) {
      cytnx_complex128 *_out = (cytnx_complex128 *)out->Mem;
      cytnx_complex128 *_inl = (cytnx_complex128 *)inl->Mem;
      cytnx_complex128 *_inr = (cytnx_complex128 *)inr->Mem;

      cytnx_complex128 alpha = cytnx_complex128(1, 0), beta = cytnx_complex128(0, 0);
      zgemm((char *)"N", (char *)"N", &Nr, &Ml, &Comm, &alpha, _inr, &Nr, _inl, &Comm, &beta, _out,
            &Nr);
    }

    void Matmul_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &inl,
                            const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int32 &Ml,
                            const cytnx_int32 &Comm, const cytnx_int32 &Nr) {
      cytnx_complex64 *_out = (cytnx_complex64 *)out->Mem;
      cytnx_complex64 *_inl = (cytnx_complex64 *)inl->Mem;
      cytnx_complex64 *_inr = (cytnx_complex64 *)inr->Mem;

      cytnx_complex64 alpha = cytnx_complex64(1, 0), beta = cytnx_complex64(0, 0);
      cgemm((char *)"N", (char *)"N", &Nr, &Ml, &Comm, &alpha, _inr, &Nr, _inl, &Comm, &beta, _out,
            &Nr);
    }

    void Matmul_internal_d(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &inl,
                           const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int32 &Ml,
                           const cytnx_int32 &Comm, const cytnx_int32 &Nr) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_inl = (cytnx_double *)inl->Mem;
      cytnx_double *_inr = (cytnx_double *)inr->Mem;

      cytnx_double alpha = 1, beta = 0;
      dgemm((char *)"N", (char *)"N", &Nr, &Ml, &Comm, &alpha, _inr, &Nr, _inl, &Comm, &beta, _out,
            &Nr);
    }

    void Matmul_internal_f(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &inl,
                           const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int32 &Ml,
                           const cytnx_int32 &Comm, const cytnx_int32 &Nr) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_float *_inl = (cytnx_float *)inl->Mem;
      cytnx_float *_inr = (cytnx_float *)inr->Mem;

      cytnx_float alpha = 1, beta = 0;
      sgemm((char *)"N", (char *)"N", &Nr, &Ml, &Comm, &alpha, _inr, &Nr, _inl, &Comm, &beta, _out,
            &Nr);
    }

    void Matmul_internal_i64(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &inl,
                             const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int32 &Ml,
                             const cytnx_int32 &Comm, const cytnx_int32 &Nr) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_int64 *_inl = (cytnx_int64 *)inl->Mem;
      cytnx_int64 *_inr = (cytnx_int64 *)inr->Mem;
      Matmul_driver(_out, _inl, _inr, Ml, Comm, Nr);
    }

    void Matmul_internal_u64(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &inl,
                             const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int32 &Ml,
                             const cytnx_int32 &Comm, const cytnx_int32 &Nr) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->Mem;
      cytnx_uint64 *_inl = (cytnx_uint64 *)inl->Mem;
      cytnx_uint64 *_inr = (cytnx_uint64 *)inr->Mem;
      Matmul_driver(_out, _inl, _inr, Ml, Comm, Nr);
    }

    void Matmul_internal_i32(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &inl,
                             const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int32 &Ml,
                             const cytnx_int32 &Comm, const cytnx_int32 &Nr) {
      cytnx_int32 *_out = (cytnx_int32 *)out->Mem;
      cytnx_int32 *_inl = (cytnx_int32 *)inl->Mem;
      cytnx_int32 *_inr = (cytnx_int32 *)inr->Mem;
      Matmul_driver(_out, _inl, _inr, Ml, Comm, Nr);
    }

    void Matmul_internal_u32(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &inl,
                             const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int32 &Ml,
                             const cytnx_int32 &Comm, const cytnx_int32 &Nr) {
      cytnx_uint32 *_out = (cytnx_uint32 *)out->Mem;
      cytnx_uint32 *_inl = (cytnx_uint32 *)inl->Mem;
      cytnx_uint32 *_inr = (cytnx_uint32 *)inr->Mem;
      Matmul_driver(_out, _inl, _inr, Ml, Comm, Nr);
    }
    void Matmul_internal_i16(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &inl,
                             const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int32 &Ml,
                             const cytnx_int32 &Comm, const cytnx_int32 &Nr) {
      cytnx_int16 *_out = (cytnx_int16 *)out->Mem;
      cytnx_int16 *_inl = (cytnx_int16 *)inl->Mem;
      cytnx_int16 *_inr = (cytnx_int16 *)inr->Mem;
      Matmul_driver(_out, _inl, _inr, Ml, Comm, Nr);
    }

    void Matmul_internal_u16(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &inl,
                             const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int32 &Ml,
                             const cytnx_int32 &Comm, const cytnx_int32 &Nr) {
      cytnx_uint16 *_out = (cytnx_uint16 *)out->Mem;
      cytnx_uint16 *_inl = (cytnx_uint16 *)inl->Mem;
      cytnx_uint16 *_inr = (cytnx_uint16 *)inr->Mem;
      Matmul_driver(_out, _inl, _inr, Ml, Comm, Nr);
    }
    void Matmul_internal_b(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &inl,
                           const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int32 &Ml,
                           const cytnx_int32 &Comm, const cytnx_int32 &Nr) {
      cytnx_bool *_out = (cytnx_bool *)out->Mem;
      cytnx_bool *_inl = (cytnx_bool *)inl->Mem;
      cytnx_bool *_inr = (cytnx_bool *)inr->Mem;
      Matmul_driver(_out, _inl, _inr, Ml, Comm, Nr);
    }

  }  // namespace linalg_internal

};  // namespace cytnx
