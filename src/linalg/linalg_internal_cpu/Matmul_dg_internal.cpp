#include "Matmul_dg_internal.hpp"
#include "cytnx_error.hpp"
#include "lapack_wrapper.hpp"

#ifdef UNI_OMP
  #include <omp.h>
#endif

namespace cytnx {
  namespace linalg_internal {

    template <class T1>
    void Matmul_dg_diagL_driver(T1 *out, const T1 *inl, const T1 *inr, const cytnx_int64 &Ml,
                                const cytnx_int64 &Comm, const cytnx_int64 &Nr) {
#ifdef UNI_OMP
  #pragma omp parallel for
#endif
      for (cytnx_uint64 n = 0; n < cytnx_uint64(Ml) * Nr; n++) {
        out[n] = inl[(n / Nr)] * inr[n];
      }
    }

    template <class T1>
    void Matmul_dg_diagR_driver(T1 *out, const T1 *inl, const T1 *inr, const cytnx_int64 &Ml,
                                const cytnx_int64 &Comm, const cytnx_int64 &Nr) {
#ifdef UNI_OMP
  #pragma omp parallel for
#endif
      for (cytnx_uint64 n = 0; n < cytnx_uint64(Ml) * Nr; n++) {
        out[n] = inl[n] * inr[n % Nr];
      }
    }

    void Matmul_dg_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &inl,
                               const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                               const cytnx_int64 &Comm, const cytnx_int64 &Nr, const int &diag_L) {
      cytnx_complex128 *_out = (cytnx_complex128 *)out->Mem;
      cytnx_complex128 *_inl = (cytnx_complex128 *)inl->Mem;
      cytnx_complex128 *_inr = (cytnx_complex128 *)inr->Mem;

      blas_int blsMl = Ml, blsNr = Nr;
      blas_int blsONE = 1;
      if (diag_L) {
        for (cytnx_int64 i = 0; i < Ml; i++)
          zaxpy(&blsNr, &_inl[i], &_inr[i * Nr], &blsONE, &_out[i * Nr], &blsONE);
      } else {
        Matmul_dg_diagR_driver(_out, _inl, _inr, Ml, Comm, Nr);
      }
    }

    void Matmul_dg_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &inl,
                               const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                               const cytnx_int64 &Comm, const cytnx_int64 &Nr, const int &diag_L) {
      cytnx_complex64 *_out = (cytnx_complex64 *)out->Mem;
      cytnx_complex64 *_inl = (cytnx_complex64 *)inl->Mem;
      cytnx_complex64 *_inr = (cytnx_complex64 *)inr->Mem;

      blas_int blsMl = Ml, blsNr = Nr;
      blas_int blsONE = 1;
      if (diag_L) {
        for (cytnx_int64 i = 0; i < Ml; i++)
          caxpy(&blsNr, &_inl[i], &_inr[i * Nr], &blsONE, &_out[i * Nr], &blsONE);
      } else {
        Matmul_dg_diagR_driver(_out, _inl, _inr, Ml, Comm, Nr);
      }
    }

    void Matmul_dg_internal_d(boost::intrusive_ptr<Storage_base> &out,
                              const boost::intrusive_ptr<Storage_base> &inl,
                              const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                              const cytnx_int64 &Comm, const cytnx_int64 &Nr, const int &diag_L) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_inl = (cytnx_double *)inl->Mem;
      cytnx_double *_inr = (cytnx_double *)inr->Mem;

      blas_int blsMl = Ml, blsNr = Nr;
      blas_int blsONE = 1;
      if (diag_L) {
        for (cytnx_int64 i = 0; i < Ml; i++)
          daxpy(&blsNr, &_inl[i], &_inr[i * Nr], &blsONE, &_out[i * Nr], &blsONE);
      } else {
        Matmul_dg_diagR_driver(_out, _inl, _inr, Ml, Comm, Nr);
      }
    }

    void Matmul_dg_internal_f(boost::intrusive_ptr<Storage_base> &out,
                              const boost::intrusive_ptr<Storage_base> &inl,
                              const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                              const cytnx_int64 &Comm, const cytnx_int64 &Nr, const int &diag_L) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_float *_inl = (cytnx_float *)inl->Mem;
      cytnx_float *_inr = (cytnx_float *)inr->Mem;

      blas_int blsMl = Ml, blsNr = Nr;
      blas_int blsONE = 1;
      if (diag_L) {
        for (cytnx_int64 i = 0; i < Ml; i++)
          saxpy(&blsNr, &_inl[i], &_inr[i * Nr], &blsONE, &_out[i * Nr], &blsONE);
      } else {
        Matmul_dg_diagR_driver(_out, _inl, _inr, Ml, Comm, Nr);
      }
    }

    void Matmul_dg_internal_i64(boost::intrusive_ptr<Storage_base> &out,
                                const boost::intrusive_ptr<Storage_base> &inl,
                                const boost::intrusive_ptr<Storage_base> &inr,
                                const cytnx_int64 &Ml, const cytnx_int64 &Comm,
                                const cytnx_int64 &Nr, const int &diag_L) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_int64 *_inl = (cytnx_int64 *)inl->Mem;
      cytnx_int64 *_inr = (cytnx_int64 *)inr->Mem;
      if (diag_L) {
        Matmul_dg_diagL_driver(_out, _inl, _inr, Ml, Comm, Nr);
      } else {
        Matmul_dg_diagR_driver(_out, _inl, _inr, Ml, Comm, Nr);
      }
    }

    void Matmul_dg_internal_u64(boost::intrusive_ptr<Storage_base> &out,
                                const boost::intrusive_ptr<Storage_base> &inl,
                                const boost::intrusive_ptr<Storage_base> &inr,
                                const cytnx_int64 &Ml, const cytnx_int64 &Comm,
                                const cytnx_int64 &Nr, const int &diag_L) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->Mem;
      cytnx_uint64 *_inl = (cytnx_uint64 *)inl->Mem;
      cytnx_uint64 *_inr = (cytnx_uint64 *)inr->Mem;
      if (diag_L) {
        Matmul_dg_diagL_driver(_out, _inl, _inr, Ml, Comm, Nr);
      } else {
        Matmul_dg_diagR_driver(_out, _inl, _inr, Ml, Comm, Nr);
      }
    }

    void Matmul_dg_internal_i32(boost::intrusive_ptr<Storage_base> &out,
                                const boost::intrusive_ptr<Storage_base> &inl,
                                const boost::intrusive_ptr<Storage_base> &inr,
                                const cytnx_int64 &Ml, const cytnx_int64 &Comm,
                                const cytnx_int64 &Nr, const int &diag_L) {
      cytnx_int64 *_out = (cytnx_int64 *)out->Mem;
      cytnx_int64 *_inl = (cytnx_int64 *)inl->Mem;
      cytnx_int64 *_inr = (cytnx_int64 *)inr->Mem;
      if (diag_L) {
        Matmul_dg_diagL_driver(_out, _inl, _inr, Ml, Comm, Nr);
      } else {
        Matmul_dg_diagR_driver(_out, _inl, _inr, Ml, Comm, Nr);
      }
    }

    void Matmul_dg_internal_u32(boost::intrusive_ptr<Storage_base> &out,
                                const boost::intrusive_ptr<Storage_base> &inl,
                                const boost::intrusive_ptr<Storage_base> &inr,
                                const cytnx_int64 &Ml, const cytnx_int64 &Comm,
                                const cytnx_int64 &Nr, const int &diag_L) {
      cytnx_uint64 *_out = (cytnx_uint64 *)out->Mem;
      cytnx_uint64 *_inl = (cytnx_uint64 *)inl->Mem;
      cytnx_uint64 *_inr = (cytnx_uint64 *)inr->Mem;
      if (diag_L) {
        Matmul_dg_diagL_driver(_out, _inl, _inr, Ml, Comm, Nr);
      } else {
        Matmul_dg_diagR_driver(_out, _inl, _inr, Ml, Comm, Nr);
      }
    }
    void Matmul_dg_internal_i16(boost::intrusive_ptr<Storage_base> &out,
                                const boost::intrusive_ptr<Storage_base> &inl,
                                const boost::intrusive_ptr<Storage_base> &inr,
                                const cytnx_int64 &Ml, const cytnx_int64 &Comm,
                                const cytnx_int64 &Nr, const int &diag_L) {
      cytnx_int16 *_out = (cytnx_int16 *)out->Mem;
      cytnx_int16 *_inl = (cytnx_int16 *)inl->Mem;
      cytnx_int16 *_inr = (cytnx_int16 *)inr->Mem;
      if (diag_L) {
        Matmul_dg_diagL_driver(_out, _inl, _inr, Ml, Comm, Nr);
      } else {
        Matmul_dg_diagR_driver(_out, _inl, _inr, Ml, Comm, Nr);
      }
    }

    void Matmul_dg_internal_u16(boost::intrusive_ptr<Storage_base> &out,
                                const boost::intrusive_ptr<Storage_base> &inl,
                                const boost::intrusive_ptr<Storage_base> &inr,
                                const cytnx_int64 &Ml, const cytnx_int64 &Comm,
                                const cytnx_int64 &Nr, const int &diag_L) {
      cytnx_uint16 *_out = (cytnx_uint16 *)out->Mem;
      cytnx_uint16 *_inl = (cytnx_uint16 *)inl->Mem;
      cytnx_uint16 *_inr = (cytnx_uint16 *)inr->Mem;
      if (diag_L) {
        Matmul_dg_diagL_driver(_out, _inl, _inr, Ml, Comm, Nr);
      } else {
        Matmul_dg_diagR_driver(_out, _inl, _inr, Ml, Comm, Nr);
      }
    }
    void Matmul_dg_internal_b(boost::intrusive_ptr<Storage_base> &out,
                              const boost::intrusive_ptr<Storage_base> &inl,
                              const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                              const cytnx_int64 &Comm, const cytnx_int64 &Nr, const int &diag_L) {
      cytnx_bool *_out = (cytnx_bool *)out->Mem;
      cytnx_bool *_inl = (cytnx_bool *)inl->Mem;
      cytnx_bool *_inr = (cytnx_bool *)inr->Mem;
      if (diag_L) {
        Matmul_dg_diagL_driver(_out, _inl, _inr, Ml, Comm, Nr);
      } else {
        Matmul_dg_diagR_driver(_out, _inl, _inr, Ml, Comm, Nr);
      }
    }

  }  // namespace linalg_internal

};  // namespace cytnx
