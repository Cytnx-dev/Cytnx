#include "Svd_internal.hpp"
#include "cytnx_error.hpp"
#include "lapack_wrapper.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// Svd
    void Svd_internal_cd(const boost::intrusive_ptr<Storage_base> &in,
                         boost::intrusive_ptr<Storage_base> &U,
                         boost::intrusive_ptr<Storage_base> &vT,
                         boost::intrusive_ptr<Storage_base> &S, const cytnx_int64 &M,
                         const cytnx_int64 &N) {
      // char jobu, jobv;

      // // if U and vT are NULL ptr, then it will not be computed.
      // jobu = (U->dtype == Type.Void) ? 'N' : 'S';
      // jobv = (vT->dtype == Type.Void) ? 'N' : 'S';

      cytnx_complex128 *Mij = (cytnx_complex128 *)malloc(M * N * sizeof(cytnx_complex128));
      memcpy(Mij, in->Mem, M * N * sizeof(cytnx_complex128));

      lapack_int min = std::min(M, N);
      lapack_int ldA = N, ldu = N, ldvT = min;
      lapack_int info;

      void* UMem=(U->Mem?U->Mem:malloc(min * min * sizeof(cytnx_complex128)));
      void* vTMem=(vT->Mem?vT->Mem:malloc(min * min * sizeof(cytnx_complex128)));
      char jobz = 'S';
      if(U->dtype == Type.Void and vT->dtype == Type.Void){
        jobz = 'N';
      }

      // double *superb = (double *)malloc(sizeof(double) * (min - 1));

      // info = LAPACKE_zgesvd(LAPACK_COL_MAJOR, jobv, jobu, N, M, (lapack_complex_double *)Mij, ldA,
      //                       (cytnx_double *)S->Mem, (lapack_complex_double *)vT->Mem, ldu,
      //                       (lapack_complex_double *)U->Mem, ldvT, superb);
      info = LAPACKE_zgesdd(LAPACK_COL_MAJOR, jobz, N, M, (lapack_complex_double *)Mij, ldA,
                            (cytnx_double *)S->Mem, (lapack_complex_double *)vTMem, ldu,
                            (lapack_complex_double *)UMem, ldvT);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'zgesvd': Lapack INFO = ", info);

      free(Mij);
      // free(superb);
      if(U->dtype == Type.Void){
        free(UMem);
      }
      if(vT->dtype == Type.Void){
        free(vTMem);
      }
    }
    void Svd_internal_cf(const boost::intrusive_ptr<Storage_base> &in,
                         boost::intrusive_ptr<Storage_base> &U,
                         boost::intrusive_ptr<Storage_base> &vT,
                         boost::intrusive_ptr<Storage_base> &S, const cytnx_int64 &M,
                         const cytnx_int64 &N) {
      char jobu, jobv;

      // if U and vT are NULL ptr, then it will not be computed.
      jobu = (U->dtype == Type.Void) ? 'N' : 'S';
      jobv = (vT->dtype == Type.Void) ? 'N' : 'S';

      cytnx_complex64 *Mij = (cytnx_complex64 *)malloc(M * N * sizeof(cytnx_complex64));
      memcpy(Mij, in->Mem, M * N * sizeof(cytnx_complex64));

      lapack_int min = std::min(M, N);
      lapack_int ldA = N, ldu = N, ldvT = min;
      lapack_int info;
      float *superb = (float *)malloc(sizeof(float) * (min - 1));

      info = LAPACKE_cgesvd(LAPACK_COL_MAJOR, jobv, jobu, N, M, (lapack_complex_float *)Mij, ldA,
                            (cytnx_float *)S->Mem, (lapack_complex_float *)vT->Mem, ldu,
                            (lapack_complex_float *)U->Mem, ldvT, superb);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'cgesvd': Lapack INFO = ", info);

      free(Mij);
      free(superb);
    }
    void Svd_internal_d(const boost::intrusive_ptr<Storage_base> &in,
                        boost::intrusive_ptr<Storage_base> &U,
                        boost::intrusive_ptr<Storage_base> &vT,
                        boost::intrusive_ptr<Storage_base> &S, const cytnx_int64 &M,
                        const cytnx_int64 &N) {
      // char jobu, jobv;

      // jobu = (U->dtype == Type.Void) ? 'N' : 'S';
      // jobv = (vT->dtype == Type.Void) ? 'N' : 'S';

      // cytnx_double *Mij = (cytnx_double *)malloc(M * N * sizeof(cytnx_double));
      // memcpy(Mij, in->Mem, M * N * sizeof(cytnx_double));

      lapack_int min = std::min(M, N);
      lapack_int ldA = N, ldu = N, ldvT = min;
      lapack_int info;

      char jobz = 'S';
      if(U->dtype == Type.Void and vT->dtype == Type.Void){
        jobz = 'N';
      }
      void* UMem=(U->Mem?U->Mem:(jobz=='S'?malloc(min * min * sizeof(cytnx_double)):NULL));
      void* vTMem=(vT->Mem?vT->Mem:(jobz=='S'?malloc(min * min * sizeof(cytnx_double)):NULL));
      

      // double *superb = (double *)malloc(sizeof(double) * (min - 1));
      // info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, jobv, jobu, N, M, Mij, ldA, (cytnx_double *)S->Mem,
      //                       (cytnx_double *)vT->Mem, ldu, (cytnx_double *)U->Mem, ldvT, superb);
      info = LAPACKE_dgesdd(LAPACK_COL_MAJOR, jobz, N, M, (cytnx_double *)in->Mem, ldA, (cytnx_double *)S->Mem,
                            (cytnx_double *)vTMem, ldu, (cytnx_double *)UMem, ldvT);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'dgesvd': Lapack INFO = ", info);

      // free(superb);
      if(U->dtype == Type.Void){
        free(UMem);
      }
      if(vT->dtype == Type.Void){
        free(vTMem);
      }
      // free(Mij);
    }
    void Svd_internal_f(const boost::intrusive_ptr<Storage_base> &in,
                        boost::intrusive_ptr<Storage_base> &U,
                        boost::intrusive_ptr<Storage_base> &vT,
                        boost::intrusive_ptr<Storage_base> &S, const cytnx_int64 &M,
                        const cytnx_int64 &N) {
      char jobu, jobv;

      jobu = (U->dtype == Type.Void) ? 'N' : 'S';
      jobv = (vT->dtype == Type.Void) ? 'N' : 'S';

      cytnx_float *Mij = (cytnx_float *)malloc(M * N * sizeof(cytnx_float));
      memcpy(Mij, in->Mem, M * N * sizeof(cytnx_float));

      lapack_int min = std::min(M, N);
      lapack_int ldA = N, ldu = N, ldvT = min;
      lapack_int info;

      float *superb = (float *)malloc(sizeof(float) * (min - 1));
      info = LAPACKE_sgesvd(LAPACK_COL_MAJOR, jobv, jobu, N, M, Mij, ldA, (cytnx_float *)S->Mem,
                            (cytnx_float *)vT->Mem, ldu, (cytnx_float *)U->Mem, ldvT, superb);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'sgesvd': Lapack INFO = ", info);

      free(superb);
      free(Mij);
    }

  }  // namespace linalg_internal
}  // namespace cytnx
