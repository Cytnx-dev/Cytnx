#include "Lstsq_internal.hpp"
#include "cytnx_error.hpp"
#include "lapack_wrapper.hpp"

namespace cytnx {
  namespace linalg_internal {
    void Lstsq_internal_d(boost::intrusive_ptr<Storage_base> &in,
                          boost::intrusive_ptr<Storage_base> &b,
                          boost::intrusive_ptr<Storage_base> &s,
                          boost::intrusive_ptr<Storage_base> &r, const cytnx_int64 &M,
                          const cytnx_int64 &N, const cytnx_int64 &nrhs, const cytnx_float &rcond) {
      lapack_int info, lda, ldb;
      lda = N;
      ldb = nrhs;
      info = LAPACKE_dgelsd(LAPACK_ROW_MAJOR, (lapack_int)M, (lapack_int)N, (lapack_int)nrhs,
                            (double *)in->Mem, lda, (double *)b->Mem, ldb, (double *)s->Mem,
                            (double)rcond, (lapack_int *)r->Mem);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'dgelsd': Lapack INFO = ", info);
    }

    void Lstsq_internal_f(boost::intrusive_ptr<Storage_base> &in,
                          boost::intrusive_ptr<Storage_base> &b,
                          boost::intrusive_ptr<Storage_base> &s,
                          boost::intrusive_ptr<Storage_base> &r, const cytnx_int64 &M,
                          const cytnx_int64 &N, const cytnx_int64 &nrhs, const cytnx_float &rcond) {
      lapack_int info, lda, ldb;
      lda = N;
      ldb = nrhs;
      info = LAPACKE_sgelsd(LAPACK_ROW_MAJOR, (lapack_int)M, (lapack_int)N, (lapack_int)nrhs,
                            (float *)in->Mem, lda, (float *)b->Mem, ldb, (float *)s->Mem,
                            (float)rcond, (lapack_int *)r->Mem);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'sgelsd': Lapack INFO = ", info);
    }

    void Lstsq_internal_cf(boost::intrusive_ptr<Storage_base> &in,
                           boost::intrusive_ptr<Storage_base> &b,
                           boost::intrusive_ptr<Storage_base> &s,
                           boost::intrusive_ptr<Storage_base> &r, const cytnx_int64 &M,
                           const cytnx_int64 &N, const cytnx_int64 &nrhs,
                           const cytnx_float &rcond) {
      lapack_int info, lda, ldb;
      lda = N;
      ldb = nrhs;
      info = LAPACKE_cgelsd(LAPACK_ROW_MAJOR, (lapack_int)M, (lapack_int)N, (lapack_int)nrhs,
                            (lapack_complex_float *)in->Mem, lda, (lapack_complex_float *)b->Mem,
                            ldb, (float *)s->Mem, (float)rcond, (lapack_int *)r->Mem);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'cgelsd': Lapack INFO = ", info);
    }

    void Lstsq_internal_cd(boost::intrusive_ptr<Storage_base> &in,
                           boost::intrusive_ptr<Storage_base> &b,
                           boost::intrusive_ptr<Storage_base> &s,
                           boost::intrusive_ptr<Storage_base> &r, const cytnx_int64 &M,
                           const cytnx_int64 &N, const cytnx_int64 &nrhs,
                           const cytnx_float &rcond) {
      lapack_int info, lda, ldb;
      lda = N;
      ldb = nrhs;
      info = LAPACKE_zgelsd(LAPACK_ROW_MAJOR, (lapack_int)M, (lapack_int)N, (lapack_int)nrhs,
                            (lapack_complex_double *)in->Mem, lda, (lapack_complex_double *)b->Mem,
                            ldb, (double *)s->Mem, (double)rcond, (lapack_int *)r->Mem);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'zgelsd': Lapack INFO = ", info);
    }
  }  // namespace linalg_internal
}  // namespace cytnx