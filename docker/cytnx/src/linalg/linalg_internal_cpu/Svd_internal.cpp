#include "linalg/linalg_internal_cpu/Svd_internal.hpp"
#include "cytnx_error.hpp"
#include "utils/lapack_wrapper.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// Svd
    void Svd_internal_cd(const boost::intrusive_ptr<Storage_base> &in,
                         boost::intrusive_ptr<Storage_base> &U,
                         boost::intrusive_ptr<Storage_base> &vT,
                         boost::intrusive_ptr<Storage_base> &S, const cytnx_int32 &M,
                         const cytnx_int32 &N) {
      char jobu[1], jobv[1];

      // if U and vT are NULL ptr, then it will not be computed.
      jobu[0] = (U->dtype == Type.Void) ? 'N' : 'S';
      jobv[0] = (vT->dtype == Type.Void) ? 'N' : 'S';

      cytnx_complex128 *Mij = (cytnx_complex128 *)malloc(M * N * sizeof(cytnx_complex128));
      memcpy(Mij, in->Mem, M * N * sizeof(cytnx_complex128));
      cytnx_int32 min = std::min(M, N);
      cytnx_int32 ldA = N, ldu = N, ldvT = min;
      cytnx_int32 lwork = -1;
      cytnx_complex128 worktest;
      cytnx_int32 info;

      cytnx_double *rwork =
        (cytnx_double *)malloc(std::max((cytnx_int32)1, 5 * min) * sizeof(cytnx_double));
      zgesvd(jobv, jobu, &N, &M, Mij, &ldA, (cytnx_double *)S->Mem, (cytnx_complex128 *)vT->Mem,
             &ldu, (cytnx_complex128 *)U->Mem, &ldvT, &worktest, &lwork, rwork, &info);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'zgesvd': Lapack INFO = ", info);

      lwork = (cytnx_int32)(worktest.real());
      cytnx_complex128 *work = (cytnx_complex128 *)malloc(lwork * sizeof(cytnx_complex128));
      zgesvd(jobv, jobu, &N, &M, Mij, &ldA, (cytnx_double *)S->Mem, (cytnx_complex128 *)vT->Mem,
             &ldu, (cytnx_complex128 *)U->Mem, &ldvT, work, &lwork, rwork, &info);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'zgesvd': Lapack INFO = ", info);

      free(rwork);
      free(work);
      free(Mij);
    }
    void Svd_internal_cf(const boost::intrusive_ptr<Storage_base> &in,
                         boost::intrusive_ptr<Storage_base> &U,
                         boost::intrusive_ptr<Storage_base> &vT,
                         boost::intrusive_ptr<Storage_base> &S, const cytnx_int32 &M,
                         const cytnx_int32 &N) {
      char jobu[1], jobv[1];

      // if U and vT are NULL ptr, then it will not be computed.
      jobu[0] = (U->dtype == Type.Void) ? 'N' : 'S';
      jobv[0] = (vT->dtype == Type.Void) ? 'N' : 'S';

      cytnx_complex64 *Mij = (cytnx_complex64 *)malloc(M * N * sizeof(cytnx_complex64));
      memcpy(Mij, in->Mem, M * N * sizeof(cytnx_complex64));
      cytnx_int32 min = std::min(M, N);
      cytnx_int32 ldA = N, ldu = N, ldvT = min;
      cytnx_int32 lwork = -1;
      cytnx_complex64 worktest;
      cytnx_int32 info;

      cytnx_float *rwork =
        (cytnx_float *)malloc(std::max((cytnx_int32)1, 5 * min) * sizeof(cytnx_float));
      cgesvd(jobv, jobu, &N, &M, Mij, &ldA, (cytnx_float *)S->Mem, (cytnx_complex64 *)vT->Mem, &ldu,
             (cytnx_complex64 *)U->Mem, &ldvT, &worktest, &lwork, rwork, &info);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'cgesvd': Lapack INFO = ", info);

      lwork = (cytnx_int32)(worktest.real());
      cytnx_complex64 *work = (cytnx_complex64 *)malloc(lwork * sizeof(cytnx_complex64));
      cgesvd(jobv, jobu, &N, &M, Mij, &ldA, (cytnx_float *)S->Mem, (cytnx_complex64 *)vT->Mem, &ldu,
             (cytnx_complex64 *)U->Mem, &ldvT, work, &lwork, rwork, &info);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'cgesvd': Lapack INFO = ", info);

      free(rwork);
      free(work);
      free(Mij);
    }
    void Svd_internal_d(const boost::intrusive_ptr<Storage_base> &in,
                        boost::intrusive_ptr<Storage_base> &U,
                        boost::intrusive_ptr<Storage_base> &vT,
                        boost::intrusive_ptr<Storage_base> &S, const cytnx_int32 &M,
                        const cytnx_int32 &N) {
      char jobu[1], jobv[1];

      jobu[0] = (U->dtype == Type.Void) ? 'N' : 'S';
      jobv[0] = (vT->dtype == Type.Void) ? 'N' : 'S';

      cytnx_double *Mij = (cytnx_double *)malloc(M * N * sizeof(cytnx_double));
      memcpy(Mij, in->Mem, M * N * sizeof(cytnx_double));
      cytnx_int32 min = std::min(M, N);
      cytnx_int32 ldA = N, ldu = N, ldvT = min;
      cytnx_int32 lwork = -1;
      cytnx_double worktest;
      cytnx_int32 info;

      dgesvd(jobv, jobu, &N, &M, Mij, &ldA, (cytnx_double *)S->Mem, (cytnx_double *)vT->Mem, &ldu,
             (cytnx_double *)U->Mem, &ldvT, &worktest, &lwork, &info);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'dgesvd': Lapack INFO = ", info);

      lwork = (cytnx_int32)worktest;
      cytnx_double *work = (cytnx_double *)malloc(lwork * sizeof(cytnx_double));
      dgesvd(jobv, jobu, &N, &M, Mij, &ldA, (cytnx_double *)S->Mem, (cytnx_double *)vT->Mem, &ldu,
             (cytnx_double *)U->Mem, &ldvT, work, &lwork, &info);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'dgesvd': Lapack INFO = ", info);

      free(work);
      free(Mij);
    }
    void Svd_internal_f(const boost::intrusive_ptr<Storage_base> &in,
                        boost::intrusive_ptr<Storage_base> &U,
                        boost::intrusive_ptr<Storage_base> &vT,
                        boost::intrusive_ptr<Storage_base> &S, const cytnx_int32 &M,
                        const cytnx_int32 &N) {
      char jobu[1], jobv[1];

      jobu[0] = (U->dtype == Type.Void) ? 'N' : 'S';
      jobv[0] = (vT->dtype == Type.Void) ? 'N' : 'S';

      cytnx_float *Mij = (cytnx_float *)malloc(M * N * sizeof(cytnx_float));
      memcpy(Mij, in->Mem, M * N * sizeof(cytnx_float));
      cytnx_int32 min = std::min(M, N);
      cytnx_int32 ldA = N, ldu = N, ldvT = min;
      cytnx_int32 lwork = -1;
      cytnx_float worktest;
      cytnx_int32 info;

      sgesvd(jobv, jobu, &N, &M, Mij, &ldA, (cytnx_float *)S->Mem, (cytnx_float *)vT->Mem, &ldu,
             (cytnx_float *)U->Mem, &ldvT, &worktest, &lwork, &info);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'sgesvd': Lapack INFO = ", info);

      lwork = (cytnx_int32)worktest;
      cytnx_float *work = (cytnx_float *)malloc(lwork * sizeof(cytnx_float));
      sgesvd(jobv, jobu, &N, &M, Mij, &ldA, (cytnx_float *)S->Mem, (cytnx_float *)vT->Mem, &ldu,
             (cytnx_float *)U->Mem, &ldvT, work, &lwork, &info);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'sgesvd': Lapack INFO = ", info);

      free(work);
      free(Mij);
    }

  }  // namespace linalg_internal
}  // namespace cytnx
