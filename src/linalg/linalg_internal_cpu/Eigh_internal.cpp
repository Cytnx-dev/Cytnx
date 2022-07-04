#include "Eigh_internal.hpp"
#include "cytnx_error.hpp"
#include "lapack_wrapper.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// Eigh
    void Eigh_internal_cd(const boost::intrusive_ptr<Storage_base> &in,
                          boost::intrusive_ptr<Storage_base> &e,
                          boost::intrusive_ptr<Storage_base> &v, const cytnx_int64 &L) {
      char jobs = 'N';

      cytnx_complex128 *tA;
      if (v->dtype != Type.Void) {
        tA = (cytnx_complex128 *)v->Mem;
        memcpy(v->Mem, in->Mem, sizeof(cytnx_complex128) * cytnx_uint64(L) * L);
        jobs = 'V';
      } else {
        tA = (cytnx_complex128 *)malloc(cytnx_uint64(L) * L * sizeof(cytnx_complex128));
        memcpy(tA, in->Mem, sizeof(cytnx_complex128) * cytnx_uint64(L) * L);
      }

      lapack_int ldA = L;
      lapack_int info;

      info = LAPACKE_zheev(LAPACK_COL_MAJOR, jobs, 'U', L, (lapack_complex_double *)tA, ldA,
                           (cytnx_double *)e->Mem);
      cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'zheev': Lapack INFO = ", info);

      if (v->dtype == Type.Void) free(tA);
    }
    void Eigh_internal_cf(const boost::intrusive_ptr<Storage_base> &in,
                          boost::intrusive_ptr<Storage_base> &e,
                          boost::intrusive_ptr<Storage_base> &v, const cytnx_int64 &L) {
      char jobs = 'N';

      cytnx_complex64 *tA;
      if (v->dtype != Type.Void) {
        tA = (cytnx_complex64 *)v->Mem;
        memcpy(v->Mem, in->Mem, sizeof(cytnx_complex64) * cytnx_uint64(L) * L);
        jobs = 'V';
      } else {
        tA = (cytnx_complex64 *)malloc(cytnx_uint64(L) * L * sizeof(cytnx_complex64));
        memcpy(tA, in->Mem, sizeof(cytnx_complex64) * cytnx_uint64(L) * L);
      }

      lapack_int ldA = L;
      lapack_int info;

      info = LAPACKE_cheev(LAPACK_COL_MAJOR, jobs, 'U', L, (lapack_complex_float *)tA, ldA,
                           (cytnx_float *)e->Mem);
      cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'cheev': Lapack INFO = ", info);
      if (v->dtype == Type.Void) free(tA);
    }
    void Eigh_internal_d(const boost::intrusive_ptr<Storage_base> &in,
                         boost::intrusive_ptr<Storage_base> &e,
                         boost::intrusive_ptr<Storage_base> &v, const cytnx_int64 &L) {
      char jobs = 'N';

      cytnx_double *tA;
      if (v->dtype != Type.Void) {
        tA = (cytnx_double *)v->Mem;
        memcpy(v->Mem, in->Mem, sizeof(cytnx_double) * cytnx_uint64(L) * L);
        jobs = 'V';
      } else {
        tA = (cytnx_double *)malloc(cytnx_uint64(L) * L * sizeof(cytnx_double));
        memcpy(tA, in->Mem, sizeof(cytnx_double) * cytnx_uint64(L) * L);
      }

      lapack_int ldA = L;
      lapack_int info;
      info = LAPACKE_dsyev(LAPACK_COL_MAJOR, jobs, 'U', L, tA, ldA, (cytnx_double *)e->Mem);

      cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'dsyev': Lapack INFO = ", info);

      if (v->dtype == Type.Void) free(tA);
    }
    void Eigh_internal_f(const boost::intrusive_ptr<Storage_base> &in,
                         boost::intrusive_ptr<Storage_base> &e,
                         boost::intrusive_ptr<Storage_base> &v, const cytnx_int64 &L) {
      char jobs = 'N';

      cytnx_float *tA;
      if (v->dtype != Type.Void) {
        tA = (cytnx_float *)v->Mem;
        memcpy(v->Mem, in->Mem, sizeof(cytnx_float) * cytnx_uint64(L) * L);
        jobs = 'V';
      } else {
        tA = (cytnx_float *)malloc(cytnx_uint64(L) * L * sizeof(cytnx_float));
        memcpy(tA, in->Mem, sizeof(cytnx_float) * cytnx_uint64(L) * L);
      }

      lapack_int ldA = L;
      lapack_int info;
      info = LAPACKE_ssyev(LAPACK_COL_MAJOR, jobs, 'U', L, tA, ldA, (cytnx_float *)e->Mem);

      cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'ssyev': Lapack INFO = ", info);
      if (v->dtype == Type.Void) free(tA);
    }

  }  // namespace linalg_internal
}  // namespace cytnx
