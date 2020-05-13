#include "QR_internal.hpp"
#include "cytnx_error.hpp"
#include "lapack_wrapper.hpp"

namespace cytnx{

    namespace linalg_internal{

        template<class T>
        void GetUpTri(T* out, const T* elem, const cytnx_uint64 &M, const cytnx_uint64 &N){
            cytnx_uint64 min = M < N? M : N;
            for(cytnx_uint64 i=0;i<min;i++){
                memcpy(out+i*N+i, elem + i*N+i, (N-i)*sizeof(T)); 
            }
        }


        /// QR
        void QR_internal_cd(const boost::intrusive_ptr<Storage_base> &in, boost::intrusive_ptr<Storage_base> &Q, boost::intrusive_ptr<Storage_base> &R, boost::intrusive_ptr<Storage_base> &tau, const cytnx_int32 &M, const cytnx_int32 &N){
            // Q should be the same shape as in
            // tau should be the min(M,N)

            cytnx_complex128 *pQ = (cytnx_complex128*)Q->Mem;
            cytnx_complex128 *pR = (cytnx_complex128*)R->Mem;
            cytnx_complex128 *ptau = (cytnx_complex128*)tau->Mem;


            //cytnx_complex128* Mij = (cytnx_complex128*)malloc(M * N * sizeof(cytnx_complex128));
            memcpy(pQ, in->Mem, M * N * sizeof(cytnx_complex128));

            
            cytnx_int32 min = std::min(M, N);
            cytnx_int32 ldA = N;
            cytnx_int32 lwork;
            cytnx_complex128 worktest;
            cytnx_int32 info;
            cytnx_int32 K = N;
            cytnx_complex128 *work;


            //cytnx_double* rwork = (cytnx_double*) malloc(std::max((cytnx_int32)1, 5*min) * sizeof(cytnx_double));
            //query lwork & alloc
            lwork = -1;
            zgelqf(&N, &M, pQ, &ldA, ptau, &worktest, &lwork, &info);
            lwork = (cytnx_int32)worktest.real();
            work = (cytnx_complex128*)malloc(sizeof(cytnx_complex128)*lwork);
            
            // call linalg:
            zgelqf(&N,&M, pQ, &ldA, ptau, work, &lwork, &info);
            cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'zgelqf': Lapack INFO = ", info);
            free(work);
            
            //getR:
            GetUpTri(pR,pQ,M,N);
            
            //getQ:
            //query lwork & alloc
            lwork = -1;
            cytnx_int32 col = M<N?N:M;
            zunglq(&N, &col, &K, pQ, &ldA, ptau, &worktest, &lwork, &info);
            lwork = (cytnx_int32)worktest.real();
            work = (cytnx_complex128*)malloc(sizeof(cytnx_complex128)*lwork);
            
            // call linalg:
            zunglq(&N,&col, &K, pQ, &ldA, ptau, work, &lwork, &info);
            cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'zunglq': Lapack INFO = ", info);
            free(work);


        }
        void QR_internal_cf(const boost::intrusive_ptr<Storage_base> &in, boost::intrusive_ptr<Storage_base> &Q, boost::intrusive_ptr<Storage_base> &R, boost::intrusive_ptr<Storage_base> &tau, const cytnx_int32 &M, const cytnx_int32 &N){
            // Q should be the same shape as in
            // tau should be the min(M,N)

            cytnx_complex64 *pQ = (cytnx_complex64*)Q->Mem;
            cytnx_complex64 *pR = (cytnx_complex64*)R->Mem;
            cytnx_complex64 *ptau = (cytnx_complex64*)tau->Mem;


            //cytnx_complex128* Mij = (cytnx_complex128*)malloc(M * N * sizeof(cytnx_complex128));
            memcpy(pQ, in->Mem, M * N * sizeof(cytnx_complex64));

            
            cytnx_int32 min = std::min(M, N);
            cytnx_int32 ldA = N;
            cytnx_int32 lwork;
            cytnx_complex64 worktest;
            cytnx_int32 info;
            cytnx_int32 K = N;
            cytnx_complex64 *work;


            //cytnx_double* rwork = (cytnx_double*) malloc(std::max((cytnx_int32)1, 5*min) * sizeof(cytnx_double));
            //query lwork & alloc
            lwork = -1;
            cgelqf(&N, &M, pQ, &ldA, ptau, &worktest, &lwork, &info);
            lwork = (cytnx_int32)worktest.real();
            work = (cytnx_complex64*)malloc(sizeof(cytnx_complex64)*lwork);
            
            // call linalg:
            cgelqf(&N,&M, pQ, &ldA, ptau, work, &lwork, &info);
            cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'cgelqf': Lapack INFO = ", info);
            free(work);
            
            //getR:
            GetUpTri(pR,pQ,M,N);
            
            //getQ:
            //query lwork & alloc
            lwork = -1;
            cytnx_int32 col = M<N?N:M;
            cunglq(&N, &col, &K, pQ, &ldA, ptau, &worktest, &lwork, &info);
            lwork = (cytnx_int32)worktest.real();
            work = (cytnx_complex64*)malloc(sizeof(cytnx_complex64)*lwork);
            
            // call linalg:
            cunglq(&N,&col, &K, pQ, &ldA, ptau, work, &lwork, &info);
            cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'cunglq': Lapack INFO = ", info);
            free(work);




        }
        void QR_internal_d(const boost::intrusive_ptr<Storage_base> &in, boost::intrusive_ptr<Storage_base> &Q, boost::intrusive_ptr<Storage_base> &R, boost::intrusive_ptr<Storage_base> &tau, const cytnx_int32 &M, const cytnx_int32 &N){
            // Q should be the same shape as in
            // tau should be the min(M,N)

            cytnx_double *pQ = (cytnx_double*)Q->Mem;
            cytnx_double *pR = (cytnx_double*)R->Mem;
            cytnx_double *ptau = (cytnx_double*)tau->Mem;


            //cytnx_complex128* Mij = (cytnx_complex128*)malloc(M * N * sizeof(cytnx_complex128));
            memcpy(pQ, in->Mem, M * N * sizeof(cytnx_double));

            
            cytnx_int32 min = std::min(M, N);
            cytnx_int32 ldA = N;
            cytnx_int32 lwork;
            cytnx_double worktest;
            cytnx_int32 info;
            cytnx_int32 K = N;
            cytnx_double *work;


            //cytnx_double* rwork = (cytnx_double*) malloc(std::max((cytnx_int32)1, 5*min) * sizeof(cytnx_double));
            //query lwork & alloc
            lwork = -1;
            dgelqf(&N, &M, pQ, &ldA, ptau, &worktest, &lwork, &info);
            lwork = (cytnx_int32)worktest;
            work = (cytnx_double*)malloc(sizeof(cytnx_double)*lwork);
            
            // call linalg:
            dgelqf(&N,&M, pQ, &ldA, ptau, work, &lwork, &info);
            cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'dgelqf': Lapack INFO = ", info);
            free(work);
            
            //getR:
            GetUpTri(pR,pQ,M,N);
            
            //getQ:
            //query lwork & alloc
            lwork = -1;
            cytnx_int32 col = M<N?N:M;
            dorglq(&N, &col, &K, pQ, &ldA, ptau, &worktest, &lwork, &info);
            lwork = (cytnx_int32)worktest;
            work = (cytnx_double*)malloc(sizeof(cytnx_double)*lwork);
            
            // call linalg:
            dorglq(&N,&col, &K, pQ, &ldA, ptau, work, &lwork, &info);
            cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'dorglq': Lapack INFO = ", info);
            free(work);


        }
        void QR_internal_f(const boost::intrusive_ptr<Storage_base> &in, boost::intrusive_ptr<Storage_base> &Q, boost::intrusive_ptr<Storage_base> &R, boost::intrusive_ptr<Storage_base> &tau, const cytnx_int32 &M, const cytnx_int32 &N){
            // Q should be the same shape as in
            // tau should be the min(M,N)

            cytnx_float *pQ = (cytnx_float*)Q->Mem;
            cytnx_float *pR = (cytnx_float*)R->Mem;
            cytnx_float *ptau = (cytnx_float*)tau->Mem;


            //cytnx_complex128* Mij = (cytnx_complex128*)malloc(M * N * sizeof(cytnx_complex128));
            memcpy(pQ, in->Mem, M * N * sizeof(cytnx_float));

            
            cytnx_int32 min = std::min(M, N);
            cytnx_int32 ldA = N;
            cytnx_int32 lwork;
            cytnx_float worktest;
            cytnx_int32 info;
            cytnx_int32 K = N;
            cytnx_float *work;


            //cytnx_float* rwork = (cytnx_float*) malloc(std::max((cytnx_int32)1, 5*min) * sizeof(cytnx_float));
            //query lwork & alloc
            lwork = -1;
            sgelqf(&N, &M, pQ, &ldA, ptau, &worktest, &lwork, &info);
            lwork = (cytnx_int32)worktest;
            work = (cytnx_float*)malloc(sizeof(cytnx_float)*lwork);
            
            // call linalg:
            sgelqf(&N,&M, pQ, &ldA, ptau, work, &lwork, &info);
            cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'sgelqf': Lapack INFO = ", info);
            free(work);
            
            //getR:
            GetUpTri(pR,pQ,M,N);
            
            //getQ:
            //query lwork & alloc
            lwork = -1;
            cytnx_int32 col = M<N?N:M;
            sorglq(&N, &col, &K, pQ, &ldA, ptau, &worktest, &lwork, &info);
            lwork = (cytnx_int32)worktest;
            work = (cytnx_float*)malloc(sizeof(cytnx_float)*lwork);
            
            // call linalg:
            sorglq(&N,&col, &K, pQ, &ldA, ptau, work, &lwork, &info);
            cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'sorglq': Lapack INFO = ", info);
            free(work);


        }


    }//linalg_internal 
}//cytnx



