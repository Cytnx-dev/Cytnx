#include "Sdd_internal.hpp"
#include "cytnx_error.hpp"
#include "backend/lapack_wrapper.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// Sdd
    void Sdd_internal_cd(const boost::intrusive_ptr<Storage_base> &in,
                         boost::intrusive_ptr<Storage_base> &U,
                         boost::intrusive_ptr<Storage_base> &vT,
                         boost::intrusive_ptr<Storage_base> &S, const cytnx_int64 &M,
                         const cytnx_int64 &N) {
      // char jobu, jobv;

      // // if U and vT are NULL ptr, then it will not be computed.
      // jobu = (U->dtype == Type.Void) ? 'N' : 'S';
      // jobv = (vT->dtype == Type.Void) ? 'N' : 'S';

      lapack_int min = std::min(M, N);
      lapack_int max = std::max(M, N);
      lapack_int ldA = N, ldu = N, ldvT = min;
      lapack_int info;

      cytnx_complex128 *Mij = (cytnx_complex128 *)malloc(M * N * sizeof(cytnx_complex128));
      memcpy(Mij, in->Mem, M * N * sizeof(cytnx_complex128));

      char jobz = 'S';
      if (U->dtype == Type.Void and vT->dtype == Type.Void) {
        jobz = 'N';
      }
      void *UMem =
        (U->Mem ? U->Mem : (jobz == 'S' ? malloc(max * max * sizeof(cytnx_complex128)) : NULL));
      void *vTMem =
        (vT->Mem ? vT->Mem : (jobz == 'S' ? malloc(max * max * sizeof(cytnx_complex128)) : NULL));

      // double *superb = (double *)malloc(sizeof(double) * (min - 1));

      // info = LAPACKE_zgesvd(LAPACK_COL_MAJOR, jobv, jobu, N, M, (lapack_complex_double *)Mij,
      // ldA,
      //                       (cytnx_double *)S->Mem, (lapack_complex_double *)vT->Mem, ldu,
      //                       (lapack_complex_double *)U->Mem, ldvT, superb);
      info = LAPACKE_zgesdd(LAPACK_COL_MAJOR, jobz, N, M, (lapack_complex_double *)Mij, ldA,
                            (cytnx_double *)S->Mem, (lapack_complex_double *)vTMem, ldu,
                            (lapack_complex_double *)UMem, ldvT);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'zgesvd': Lapack INFO = ", info);

      free(Mij);
      // free(superb);
      if (UMem != nullptr and U->dtype == Type.Void) {
        free(UMem);
      }
      if (vTMem != nullptr and vT->dtype == Type.Void) {
        free(vTMem);
      }
    }
    void Sdd_internal_cf(const boost::intrusive_ptr<Storage_base> &in,
                         boost::intrusive_ptr<Storage_base> &U,
                         boost::intrusive_ptr<Storage_base> &vT,
                         boost::intrusive_ptr<Storage_base> &S, const cytnx_int64 &M,
                         const cytnx_int64 &N) {
      // char jobu, jobv;

      // jobu = (U->dtype == Type.Void) ? 'N' : 'S';
      // jobv = (vT->dtype == Type.Void) ? 'N' : 'S';

      lapack_int min = std::min(M, N);
      lapack_int max = std::max(M, N);
      lapack_int ldA = N, ldu = N, ldvT = min;
      lapack_int info;

      cytnx_complex64 *Mij = (cytnx_complex64 *)malloc(M * N * sizeof(cytnx_complex64));
      memcpy(Mij, in->Mem, M * N * sizeof(cytnx_complex64));

      char jobz = 'S';
      if (U->dtype == Type.Void and vT->dtype == Type.Void) {
        jobz = 'N';
      }
      void *UMem =
        (U->Mem ? U->Mem : (jobz == 'S' ? malloc(max * max * sizeof(cytnx_complex64)) : NULL));
      void *vTMem =
        (vT->Mem ? vT->Mem : (jobz == 'S' ? malloc(max * max * sizeof(cytnx_complex64)) : NULL));

      // double *superb = (double *)malloc(sizeof(double) * (min - 1));
      // info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, jobv, jobu, N, M, Mij, ldA, (cytnx_double *)S->Mem,
      //                       (cytnx_double *)vT->Mem, ldu, (cytnx_double *)U->Mem, ldvT, superb);
      info = LAPACKE_cgesdd(LAPACK_COL_MAJOR, jobz, N, M, (lapack_complex_float *)Mij, ldA,
                            (cytnx_float *)S->Mem, (lapack_complex_float *)vTMem, ldu,
                            (lapack_complex_float *)UMem, ldvT);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'dgesvd': Lapack INFO = ", info);

      // free(superb);
      if (UMem != nullptr and U->dtype == Type.Void) {
        free(UMem);
      }
      if (vTMem != nullptr and vT->dtype == Type.Void) {
        free(vTMem);
      }
      free(Mij);
    }
    void Sdd_internal_d(const boost::intrusive_ptr<Storage_base> &in,
                        boost::intrusive_ptr<Storage_base> &U,
                        boost::intrusive_ptr<Storage_base> &vT,
                        boost::intrusive_ptr<Storage_base> &S, const cytnx_int64 &M,
                        const cytnx_int64 &N) {
      // char jobu, jobv;

      // jobu = (U->dtype == Type.Void) ? 'N' : 'S';
      // jobv = (vT->dtype == Type.Void) ? 'N' : 'S';

      lapack_int min = std::min(M, N);
      lapack_int max = std::max(M, N);
      lapack_int ldA = N, ldu = N, ldvT = min;
      lapack_int info;

      cytnx_double *Mij = (cytnx_double *)malloc(M * N * sizeof(cytnx_double));
      memcpy(Mij, in->Mem, M * N * sizeof(cytnx_double));

      char jobz = 'S';
      if (U->dtype == Type.Void and vT->dtype == Type.Void) {
        jobz = 'N';
      }
      void *UMem =
        (U->Mem ? U->Mem : (jobz == 'S' ? malloc(max * max * sizeof(cytnx_double)) : NULL));
      void *vTMem =
        (vT->Mem ? vT->Mem : (jobz == 'S' ? malloc(max * max * sizeof(cytnx_double)) : NULL));

      // double *superb = (double *)malloc(sizeof(double) * (min - 1));
      // info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, jobv, jobu, N, M, Mij, ldA, (cytnx_double *)S->Mem,
      //                       (cytnx_double *)vT->Mem, ldu, (cytnx_double *)U->Mem, ldvT, superb);
      info = LAPACKE_dgesdd(LAPACK_COL_MAJOR, jobz, N, M, (cytnx_double *)Mij, ldA,
                            (cytnx_double *)S->Mem, (cytnx_double *)vTMem, ldu,
                            (cytnx_double *)UMem, ldvT);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'dgesvd': Lapack INFO = ", info);

      // free(superb);
      if (UMem != nullptr and U->dtype == Type.Void) {
        free(UMem);
      }
      if (vTMem != nullptr and vT->dtype == Type.Void) {
        free(vTMem);
      }
      free(Mij);
    }
    void Sdd_internal_f(const boost::intrusive_ptr<Storage_base> &in,
                        boost::intrusive_ptr<Storage_base> &U,
                        boost::intrusive_ptr<Storage_base> &vT,
                        boost::intrusive_ptr<Storage_base> &S, const cytnx_int64 &M,
                        const cytnx_int64 &N) {
      // char jobu, jobv;

      // jobu = (U->dtype == Type.Void) ? 'N' : 'S';
      // jobv = (vT->dtype == Type.Void) ? 'N' : 'S';

      lapack_int min = std::min(M, N);
      lapack_int max = std::max(M, N);
      lapack_int ldA = N, ldu = N, ldvT = min;
      lapack_int info;

      cytnx_float *Mij = (cytnx_float *)malloc(M * N * sizeof(cytnx_float));
      memcpy(Mij, in->Mem, M * N * sizeof(cytnx_float));

      char jobz = 'S';
      if (U->dtype == Type.Void and vT->dtype == Type.Void) {
        jobz = 'N';
      }
      void *UMem =
        (U->Mem ? U->Mem : (jobz == 'S' ? malloc(max * max * sizeof(cytnx_float)) : NULL));
      void *vTMem =
        (vT->Mem ? vT->Mem : (jobz == 'S' ? malloc(max * max * sizeof(cytnx_float)) : NULL));

      // double *superb = (double *)malloc(sizeof(double) * (min - 1));
      // info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, jobv, jobu, N, M, Mij, ldA, (cytnx_double *)S->Mem,
      //                       (cytnx_double *)vT->Mem, ldu, (cytnx_double *)U->Mem, ldvT, superb);
      info =
        LAPACKE_sgesdd(LAPACK_COL_MAJOR, jobz, N, M, (cytnx_float *)Mij, ldA, (cytnx_float *)S->Mem,
                       (cytnx_float *)vTMem, ldu, (cytnx_float *)UMem, ldvT);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'dgesvd': Lapack INFO = ", info);

      // free(superb);
      if (UMem != nullptr and U->dtype == Type.Void) {
        free(UMem);
      }
      if (vTMem != nullptr and vT->dtype == Type.Void) {
        free(vTMem);
      }
      free(Mij);
    }

  }  // namespace linalg_internal
}  // namespace cytnx
