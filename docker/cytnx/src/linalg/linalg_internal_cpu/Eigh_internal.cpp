#include "linalg/linalg_internal_cpu/Eigh_internal.hpp"
#include "cytnx_error.hpp"
#include "utils/lapack_wrapper.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// Eigh
    void Eigh_internal_cd(const boost::intrusive_ptr<Storage_base> &in,
                          boost::intrusive_ptr<Storage_base> &e,
                          boost::intrusive_ptr<Storage_base> &v, const cytnx_int32 &L) {
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

      cytnx_int32 ldA = L;
      cytnx_int32 lwork = -1;
      cytnx_complex128 worktest;
      cytnx_int32 info;
      cytnx_double *rwork = (cytnx_double *)malloc((3 * L + 1) * sizeof(cytnx_double));

      zheev(&jobs, (char *)"U", &L, tA, &ldA, (cytnx_double *)e->Mem, &worktest, &lwork, rwork,
            &info);

      cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'dsyev': Lapack INFO = ", info);

      lwork = (cytnx_int32)(worktest.real());
      cytnx_complex128 *work = (cytnx_complex128 *)malloc(sizeof(cytnx_complex128) * lwork);
      zheev(&jobs, (char *)"U", &L, tA, &ldA, (cytnx_double *)e->Mem, work, &lwork, rwork, &info);

      cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'dsyev': Lapack INFO = ", info);
      free(rwork);
      free(work);
      if (v->dtype == Type.Void) free(tA);
    }
    void Eigh_internal_cf(const boost::intrusive_ptr<Storage_base> &in,
                          boost::intrusive_ptr<Storage_base> &e,
                          boost::intrusive_ptr<Storage_base> &v, const cytnx_int32 &L) {
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

      cytnx_int32 ldA = L;
      cytnx_int32 lwork = -1;
      cytnx_complex64 worktest;
      cytnx_int32 info;
      cytnx_float *rwork = (cytnx_float *)malloc((3 * L + 1) * sizeof(cytnx_float));

      cheev(&jobs, (char *)"U", &L, tA, &ldA, (cytnx_float *)e->Mem, &worktest, &lwork, rwork,
            &info);

      cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'dsyev': Lapack INFO = ", info);

      lwork = (cytnx_int32)(worktest.real());
      cytnx_complex64 *work = (cytnx_complex64 *)malloc(sizeof(cytnx_complex64) * lwork);
      cheev(&jobs, (char *)"U", &L, tA, &ldA, (cytnx_float *)e->Mem, work, &lwork, rwork, &info);

      cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'dsyev': Lapack INFO = ", info);
      free(rwork);
      free(work);
      if (v->dtype == Type.Void) free(tA);
    }
    void Eigh_internal_d(const boost::intrusive_ptr<Storage_base> &in,
                         boost::intrusive_ptr<Storage_base> &e,
                         boost::intrusive_ptr<Storage_base> &v, const cytnx_int32 &L) {
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

      cytnx_int32 ldA = L;
      cytnx_int32 lwork = -1;
      cytnx_double worktest;
      cytnx_int32 info;
      dsyev(&jobs, (char *)"U", &L, tA, &ldA, (cytnx_double *)e->Mem, &worktest, &lwork, &info);

      cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'dsyev': Lapack INFO = ", info);

      lwork = (cytnx_int32)worktest;
      cytnx_double *work = (cytnx_double *)malloc(sizeof(cytnx_double) * lwork);
      dsyev(&jobs, (char *)"U", &L, tA, &ldA, (cytnx_double *)e->Mem, work, &lwork, &info);

      cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'dsyev': Lapack INFO = ", info);

      free(work);
      if (v->dtype == Type.Void) free(tA);
    }
    void Eigh_internal_f(const boost::intrusive_ptr<Storage_base> &in,
                         boost::intrusive_ptr<Storage_base> &e,
                         boost::intrusive_ptr<Storage_base> &v, const cytnx_int32 &L) {
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

      cytnx_int32 ldA = L;
      cytnx_int32 lwork = -1;
      cytnx_float worktest;
      cytnx_int32 info;
      ssyev(&jobs, (char *)"U", &L, tA, &ldA, (cytnx_float *)e->Mem, &worktest, &lwork, &info);

      cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'dsyev': Lapack INFO = ", info);

      lwork = (cytnx_int32)worktest;
      cytnx_float *work = (cytnx_float *)malloc(sizeof(cytnx_float) * lwork);
      ssyev(&jobs, (char *)"U", &L, tA, &ldA, (cytnx_float *)e->Mem, work, &lwork, &info);

      cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'dsyev': Lapack INFO = ", info);

      free(work);
      if (v->dtype == Type.Void) free(tA);
    }

  }  // namespace linalg_internal
}  // namespace cytnx
