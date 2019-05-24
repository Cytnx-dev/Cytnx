#include "linalg/linalg_internal_cpu/Svd_internal.hpp"
#include "tor10_error.hpp"
#include "utils/lapack_wrapper.h"

namespace tor10{

    namespace linalg_internal{

        /// Svd
        void Svd_internal_cd(const boost::intrusive_ptr<Storage_base> &in, boost::intrusive_ptr<Storage_base> &U, boost::intrusive_ptr<Storage_base> &vT, boost::intrusive_ptr<Storage_base> &S, const tor10_int32 &M, const tor10_int32 &N){

            char jobu[1], jobv[1];

            // if U and vT are NULL ptr, then it will not be computed. 
            jobu[0] = ( U->dtype == tor10type.Void  ) ? 'N' : 'S';
            jobv[0] = ( vT->dtype == tor10type.Void ) ? 'N' : 'S';

            tor10_complex128* Mij = (tor10_complex128*)malloc(M * N * sizeof(tor10_complex128));
            memcpy(Mij, in->Mem, M * N * sizeof(tor10_complex128));
            tor10_int32 min = std::min(M, N);
            tor10_int32 ldA = N, ldu = N, ldvT = min;
            tor10_int32 lwork = -1;
            tor10_complex128 worktest;
            tor10_int32 info;

            tor10_double* rwork = (tor10_double*) malloc(std::max((tor10_int32)1, 5*min) * sizeof(tor10_double));
            zgesvd(jobv, jobu, &N, &M, Mij, &ldA, (tor10_double*)S->Mem, (tor10_complex128*)vT->Mem, &ldu, (tor10_complex128*)U->Mem, &ldvT, &worktest, &lwork, rwork, &info);

            tor10_error_msg(info != 0, "%s %d", "Error in Lapack function 'zgesvd': Lapack INFO = ", info);

            lwork = (tor10_int32)(worktest.real());
            tor10_complex128 *work = (tor10_complex128*)malloc(lwork*sizeof(tor10_complex128));
            zgesvd(jobv, jobu, &N, &M, Mij, &ldA, (tor10_double*)S->Mem, (tor10_complex128*)vT->Mem, &ldu, (tor10_complex128*)U->Mem, &ldvT, work, &lwork,rwork, &info);

            tor10_error_msg(info != 0, "%s %d", "Error in Lapack function 'zgesvd': Lapack INFO = ", info);

            free(rwork);
            free(work);
            free(Mij);




        }
        void Svd_internal_cf(const boost::intrusive_ptr<Storage_base> &in, boost::intrusive_ptr<Storage_base> &U, boost::intrusive_ptr<Storage_base> &vT, boost::intrusive_ptr<Storage_base> &S, const tor10_int32 &M, const tor10_int32 &N){
            char jobu[1], jobv[1];

            // if U and vT are NULL ptr, then it will not be computed. 
            jobu[0] = ( U->dtype == tor10type.Void  ) ? 'N' : 'S';
            jobv[0] = ( vT->dtype == tor10type.Void ) ? 'N' : 'S';

            tor10_complex64* Mij = (tor10_complex64*)malloc(M * N * sizeof(tor10_complex64));
            memcpy(Mij, in->Mem, M * N * sizeof(tor10_complex64));
            tor10_int32 min = std::min(M, N);
            tor10_int32 ldA = N, ldu = N, ldvT = min;
            tor10_int32 lwork = -1;
            tor10_complex64 worktest;
            tor10_int32 info;

            tor10_float* rwork = (tor10_float*) malloc(std::max((tor10_int32)1, 5*min) * sizeof(tor10_float));
            cgesvd(jobv, jobu, &N, &M, Mij, &ldA, (tor10_float*)S->Mem, (tor10_complex64*)vT->Mem, &ldu, (tor10_complex64*)U->Mem, &ldvT, &worktest, &lwork, rwork, &info);

            tor10_error_msg(info != 0, "%s %d", "Error in Lapack function 'cgesvd': Lapack INFO = ", info);

            lwork = (tor10_int32)(worktest.real());
            tor10_complex64 *work = (tor10_complex64*)malloc(lwork*sizeof(tor10_complex64));
            cgesvd(jobv, jobu, &N, &M, Mij, &ldA, (tor10_float*)S->Mem, (tor10_complex64*)vT->Mem, &ldu, (tor10_complex64*)U->Mem, &ldvT, work, &lwork,rwork, &info);

            tor10_error_msg(info != 0, "%s %d", "Error in Lapack function 'cgesvd': Lapack INFO = ", info);

            free(rwork);
            free(work);
            free(Mij);




        }
        void Svd_internal_d( const boost::intrusive_ptr<Storage_base> &in, boost::intrusive_ptr<Storage_base> &U, boost::intrusive_ptr<Storage_base> &vT, boost::intrusive_ptr<Storage_base> &S, const tor10_int32 &M, const tor10_int32 &N){
            char jobu[1], jobv[1];

            jobu[0] = (U->dtype ==tor10type.Void) ? 'N': 'S';
            jobv[0] = (vT->dtype==tor10type.Void) ? 'N': 'S';

            tor10_double* Mij = (tor10_double*)malloc(M * N * sizeof(tor10_double));
            memcpy(Mij,in->Mem, M * N * sizeof(tor10_double));
            tor10_int32 min = std::min(M, N);
            tor10_int32 ldA = N, ldu = N, ldvT = min;
            tor10_int32 lwork = -1;
            tor10_double worktest;
            tor10_int32 info;

            dgesvd(jobv, jobu, &N, &M, Mij, &ldA, (tor10_double*)S->Mem, (tor10_double*)vT->Mem, &ldu, (tor10_double*)U->Mem, &ldvT, &worktest, &lwork, &info);

            tor10_error_msg(info != 0, "%s %d", "Error in Lapack function 'dgesvd': Lapack INFO = ", info);

            lwork = (tor10_int32)worktest;
            tor10_double *work = (tor10_double*)malloc(lwork*sizeof(tor10_double));
            dgesvd(jobv, jobu, &N, &M, Mij, &ldA, (tor10_double*)S->Mem, (tor10_double*)vT->Mem, &ldu, (tor10_double*)U->Mem, &ldvT, work, &lwork, &info);

            tor10_error_msg(info != 0, "%s %d", "Error in Lapack function 'dgesvd': Lapack INFO = ", info);

            free(work);
            free(Mij);

        }
        void Svd_internal_f( const boost::intrusive_ptr<Storage_base> &in, boost::intrusive_ptr<Storage_base> &U, boost::intrusive_ptr<Storage_base> &vT, boost::intrusive_ptr<Storage_base> &S, const tor10_int32 &M, const tor10_int32 &N){

            char jobu[1], jobv[1];

            jobu[0] = (U->dtype ==tor10type.Void) ? 'N': 'S';
            jobv[0] = (vT->dtype==tor10type.Void) ? 'N': 'S';

            tor10_float* Mij = (tor10_float*)malloc(M * N * sizeof(tor10_float));
            memcpy(Mij,in->Mem, M * N * sizeof(tor10_float));
            tor10_int32 min = std::min(M, N);
            tor10_int32 ldA = N, ldu = N, ldvT = min;
            tor10_int32 lwork = -1;
            tor10_float worktest;
            tor10_int32 info;

            sgesvd(jobv, jobu, &N, &M, Mij, &ldA, (tor10_float*)S->Mem, (tor10_float*)vT->Mem, &ldu, (tor10_float*)U->Mem, &ldvT, &worktest, &lwork, &info);

            tor10_error_msg(info != 0, "%s %d", "Error in Lapack function 'sgesvd': Lapack INFO = ", info);

            lwork = (tor10_int32)worktest;
            tor10_float *work = (tor10_float*)malloc(lwork*sizeof(tor10_float));
            sgesvd(jobv, jobu, &N, &M, Mij, &ldA, (tor10_float*)S->Mem, (tor10_float*)vT->Mem, &ldu, (tor10_float*)U->Mem, &ldvT, work, &lwork, &info);

            tor10_error_msg(info != 0, "%s %d", "Error in Lapack function 'sgesvd': Lapack INFO = ", info);

            free(work);
            free(Mij);


        }


    }//linalg_internal 
}//tor10



