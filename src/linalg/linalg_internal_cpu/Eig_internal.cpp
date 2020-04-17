#include "Eig_internal.hpp"
#include "cytnx_error.hpp"
#include "lapack_wrapper.hpp"

namespace cytnx{

    namespace linalg_internal{

        /// Eig
        void Eig_internal_cd(const boost::intrusive_ptr<Storage_base> &in, boost::intrusive_ptr<Storage_base> &e, boost::intrusive_ptr<Storage_base> &v, const cytnx_int32 &L){
            char jobs = 'N';

            cytnx_complex128 *tA;
            cytnx_complex128 *buffer_A = (cytnx_complex128*)malloc(cytnx_uint64(L)*L*sizeof(cytnx_complex128));
            memcpy(buffer_A,in->Mem,sizeof(cytnx_complex128)*cytnx_uint64(L)*L);
            if(v->dtype!=Type.Void){
                tA = (cytnx_complex128*)v->Mem;
                jobs = 'V';
            }


            cytnx_int32 ldA = L;
            cytnx_int32 lwork = -1;
            cytnx_double *rwork = (cytnx_double*)malloc(sizeof(cytnx_double)*2*L);
            cytnx_complex128 workspace = 0;
            cytnx_int32 info;
            cytnx_int32 ONE = 1;


            /// query lwork
            zgeev(&jobs, (char*)"N", &L, buffer_A, &ldA, (cytnx_complex128*)e->Mem, tA,&L,nullptr, &ONE, &workspace, &lwork, rwork, &info);

            cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'zgeev': Lapack INFO = ", info);

            lwork = cytnx_int32(workspace.real());
            cytnx_complex128* work= (cytnx_complex128*)malloc(sizeof(cytnx_complex128)*lwork);
            zgeev(&jobs, (char*)"N", &L, buffer_A, &ldA, (cytnx_complex128*)e->Mem, tA,&L,nullptr, &ONE,work, &lwork, rwork, &info);

            cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'zgeev': Lapack INFO = ", info);
            free(work);
            free(buffer_A);
            free(rwork);

        }
        void Eig_internal_cf(const boost::intrusive_ptr<Storage_base> &in, boost::intrusive_ptr<Storage_base> &e, boost::intrusive_ptr<Storage_base> &v, const cytnx_int32 &L){
            char jobs = 'N';

            cytnx_complex64 *tA;
            cytnx_complex64 *buffer_A = (cytnx_complex64*)malloc(cytnx_uint64(L)*L*sizeof(cytnx_complex64));
            memcpy(buffer_A,in->Mem,sizeof(cytnx_complex64)*cytnx_uint64(L)*L);
            if(v->dtype!=Type.Void){
                tA = (cytnx_complex64*)v->Mem;
                jobs = 'V';
            }


            cytnx_int32 ldA = L;
            cytnx_int32 lwork = -1;
            cytnx_float *rwork = (cytnx_float*)malloc(sizeof(cytnx_float)*2*L);
            cytnx_complex64 workspace = 0;
            cytnx_int32 info;
            cytnx_int32 ONE = 1;

            /// query lwork
            cgeev(&jobs, (char*)"N", &L, buffer_A, &ldA, (cytnx_complex64*)e->Mem, tA,&L,nullptr, &ONE,&workspace, &lwork, rwork, &info);

            cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'cgeev': Lapack INFO = ", info);
            lwork = cytnx_int32(workspace.real());
            cytnx_complex64* work= (cytnx_complex64*)malloc(sizeof(cytnx_complex64)*lwork);
            cgeev(&jobs, (char*)"N", &L, buffer_A, &ldA, (cytnx_complex64*)e->Mem, tA,&L,nullptr, &ONE,work, &lwork, rwork, &info);

            cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'cgeev': Lapack INFO = ", info);
            free(work);
            free(buffer_A);
            free(rwork);
        }

        void Eig_internal_d( const boost::intrusive_ptr<Storage_base> &in, boost::intrusive_ptr<Storage_base> &e, boost::intrusive_ptr<Storage_base> &v, const cytnx_int32 &L){
            /*
            char jobs = 'N';

            cytnx_double *tA;
            cytnx_double *buffer_A = (cytnx_double*)malloc(cytnx_uint64(L)*L*sizeof(cytnx_double));
            memcpy(buffer_A,in->Mem,sizeof(cytnx_double)*cytnx_uint64(L)*L);
            cytnx_double *e_real = (cytnx_double*)malloc(cytnx_uint64(L)*sizeof(cytnx_double));
            cytnx_double *e_imag = (cytnx_double*)malloc(cytnx_uint64(L)*sizeof(cytnx_double));
            
            if(v->dtype!=Type.Void){
                tA = (cytnx_double*)v->Mem; 
                jobs = 'V';
            }


            cytnx_int32 ldA = L;
            cytnx_int32 lwork = -1;
            cytnx_double *rwork = (cytnx_double*)malloc(sizeof(cytnx_double)*2*L);
            cytnx_double workspace = 0;
            cytnx_int32 info;
            cytnx_int32 ONE = 1;
            cytnx_int32 TWO = 2;
            /// query lwork
            dgeev(&jobs, (char*)"N", &L, buffer_A, &ldA, e_real, e_imag, tA, &L,nullptr, &ONE,&workspace, &lwork,  &info);

            cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'dgeev': Lapack INFO = ", info);
            lwork = cytnx_int32(workspace);
            cytnx_double* work= (cytnx_double*)malloc(sizeof(cytnx_double)*lwork);
            dgeev(&jobs, (char*)"N", &L, buffer_A, &ldA, e_real, e_imag, tA, &L,nullptr, &ONE,work, &lwork,  &info);

            cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'dgeev': Lapack INFO = ", info);

            // copy the real and imag to e output:
            dcopy(L,e_real, &ONE, (cytnx_double*)e->Mem, &TWO);
            dcopy(L,e_imag, &ONE, &(((cytnx_double*)e->Mem)[1]), &TWO);

            free(work);
            free(buffer_A);
            free(rwork);
            free(e_real);
            free(e_imag);
            */
        }
        void Eig_internal_f( const boost::intrusive_ptr<Storage_base> &in, boost::intrusive_ptr<Storage_base> &e, boost::intrusive_ptr<Storage_base> &v, const cytnx_int32 &L){
            /*
            char jobs = 'N';

            cytnx_float *tA;
            cytnx_float *buffer_A = (cytnx_float*)malloc(cytnx_uint64(L)*L*sizeof(cytnx_float));
            memcpy(buffer_A,in->Mem,sizeof(cytnx_float)*cytnx_uint64(L)*L);
            cytnx_float *e_real = (cytnx_float*)malloc(cytnx_uint64(L)*sizeof(cytnx_float));
            cytnx_float *e_imag = (cytnx_float*)malloc(cytnx_uint64(L)*sizeof(cytnx_float));
            
            if(v->dtype!=Type.Void){
                tA = (cytnx_float*)v->Mem; 
                jobs = 'V';
            }


            cytnx_int32 ldA = L;
            cytnx_int32 lwork = -1;
            cytnx_float *rwork = (cytnx_float*)malloc(sizeof(cytnx_float)*2*L);
            cytnx_float workspace = 0;
            cytnx_int32 info;
            cytnx_int32 ONE = 1;
            cytnx_int32 TWO = 2;
            /// query lwork
            sgeev(&jobs, (char*)"N", &L, buffer_A, &ldA, e_real, e_imag, tA, &L,nullptr, &ONE,&workspace, &lwork,  &info);

            cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'sgeev': Lapack INFO = ", info);
            lwork = cytnx_int32(workspace);
            cytnx_float* work= (cytnx_float*)malloc(sizeof(cytnx_float)*lwork);
            sgeev(&jobs, (char*)"N", &L, buffer_A, &ldA, e_real, e_imag, tA, &L,nullptr, &ONE,work, &lwork,  &info);

            cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'sgeev': Lapack INFO = ", info);

            // copy the real and imag to e output:
            scopy(L,e_real, &ONE, (cytnx_float*)e->Mem, &TWO);
            scopy(L,e_imag, &ONE, &(((cytnx_float*)e->Mem)[1]), &TWO);

            free(work);
            free(buffer_A);
            free(rwork);
            free(e_real);
            free(e_imag);
            */
        }


    }//linalg_internal 
}//cytnx



