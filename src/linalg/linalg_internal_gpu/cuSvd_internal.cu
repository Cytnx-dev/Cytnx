#include "linalg/linalg_internal_gpu/cuSvd_internal.hpp"
#include "cytnx_error.hpp"
#include "Type.hpp"
#include "lapack_wrapper.hpp"

namespace cytnx{

    namespace linalg_internal{

        /// cuSvd
        void cuSvd_internal_cd(const boost::intrusive_ptr<Storage_base> &in, boost::intrusive_ptr<Storage_base> &U, boost::intrusive_ptr<Storage_base> &vT, boost::intrusive_ptr<Storage_base> &S, const cytnx_int32 &M, const cytnx_int32 &N){
            signed char jobu, jobv;

            // if U and vT are NULL ptr, then it will not be computed. 
            jobu = ( U->dtype == Type.Void ) ? 'N' : 'S';
            jobv = ( vT->dtype == Type.Void ) ? 'N' : 'S';


            // create handles:
            cusolverDnHandle_t cusolverH = NULL;
            checkCudaErrors(cusolverDnCreate(&cusolverH));

            cuDoubleComplex* Mij;
            checkCudaErrors(cudaMalloc((void**)&Mij,M * N * sizeof(cuDoubleComplex)));
            checkCudaErrors(cudaMemcpy(Mij,in->Mem,sizeof(cytnx_complex128)*M*N,cudaMemcpyDeviceToDevice));

            cytnx_int32 min = std::min(M, N);
            cytnx_int32 ldA = N, ldu = N, ldvT = min;
            cytnx_int32 lwork = 0;

            // query working space :
            checkCudaErrors(cusolverDnZgesvd_bufferSize(cusolverH, N, M, &lwork));


            // allocate working space:
            cuDoubleComplex *work;
            cytnx_double *rwork=NULL;
            checkCudaErrors(cudaMalloc((void**)&work,lwork*sizeof(cuDoubleComplex)));
            //checkCudaErrors(cudaMalloc((void**)&rwork,(min-1)*sizeof(cytnx_double64)));    

            cytnx_int32 *devinfo ;
            checkCudaErrors(cudaMalloc((void**)&devinfo,sizeof(cytnx_int32)));
            checkCudaErrors(cudaMemset(devinfo,0,sizeof(cytnx_int32)));

            cytnx_int32 info;
            /// compute:
            checkCudaErrors(cusolverDnZgesvd(cusolverH,jobv,jobu,N,M,Mij,ldA,(cytnx_double*)S->Mem,(cuDoubleComplex*)vT->Mem,ldu,(cuDoubleComplex*)U->Mem,ldvT,work,lwork,rwork,devinfo));

            // get info
            checkCudaErrors(cudaMemcpy(&info,devinfo,sizeof(cytnx_int32),cudaMemcpyDeviceToHost));

            cytnx_error_msg(info != 0, "%s %d", "Error in cuBlas function 'cusolverDnZgesvd': cuBlas INFO = ", info);


            cudaFree(work);
            cudaFree(Mij);
            cudaFree(devinfo);
            cusolverDnDestroy(cusolverH);


        }
        void cuSvd_internal_cf(const boost::intrusive_ptr<Storage_base> &in, boost::intrusive_ptr<Storage_base> &U, boost::intrusive_ptr<Storage_base> &vT, boost::intrusive_ptr<Storage_base> &S, const cytnx_int32 &M, const cytnx_int32 &N){
            signed char jobu, jobv;

            // if U and vT are NULL ptr, then it will not be computed. 
            jobu = ( U->dtype == Type.Void ) ? 'N' : 'S';
            jobv = ( vT->dtype == Type.Void ) ? 'N' : 'S';


            // create handles:
            cusolverDnHandle_t cusolverH = NULL;
            checkCudaErrors(cusolverDnCreate(&cusolverH));

            cuFloatComplex* Mij;
            checkCudaErrors(cudaMalloc((void**)&Mij,M * N * sizeof(cuFloatComplex)));
            checkCudaErrors(cudaMemcpy(Mij,in->Mem,sizeof(cytnx_complex128)*M*N,cudaMemcpyDeviceToDevice));

            cytnx_int32 min = std::min(M, N);
            cytnx_int32 ldA = N, ldu = N, ldvT = min;
            cytnx_int32 lwork = 0;

            // query working space :
            checkCudaErrors(cusolverDnCgesvd_bufferSize(cusolverH, N, M, &lwork));


            // allocate working space:
            cuFloatComplex *work;
            cytnx_float *rwork=NULL;
            checkCudaErrors(cudaMalloc((void**)&work,lwork*sizeof(cuFloatComplex)));
            //checkCudaErrors(cudaMalloc((void**)&rwork,(min-1)*sizeof(cytnx_float64)));    

            cytnx_int32 *devinfo ;
            checkCudaErrors(cudaMalloc((void**)&devinfo,sizeof(cytnx_int32)));
            checkCudaErrors(cudaMemset(devinfo,0,sizeof(cytnx_int32)));

            cytnx_int32 info;
            /// compute:
            checkCudaErrors(cusolverDnCgesvd(cusolverH,jobv,jobu,N,M,Mij,ldA,(cytnx_float*)S->Mem,(cuFloatComplex*)vT->Mem,ldu,(cuFloatComplex*)U->Mem,ldvT,work,lwork,rwork,devinfo));

            // get info
            checkCudaErrors(cudaMemcpy(&info,devinfo,sizeof(cytnx_int32),cudaMemcpyDeviceToHost));

            cytnx_error_msg(info != 0, "%s %d", "Error in cuBlas function 'cusolverDnCgesvd': cuBlas INFO = ", info);


            cudaFree(work);
            cudaFree(Mij);
            cudaFree(devinfo);
            cusolverDnDestroy(cusolverH);
        }
        void cuSvd_internal_d( const boost::intrusive_ptr<Storage_base> &in, boost::intrusive_ptr<Storage_base> &U, boost::intrusive_ptr<Storage_base> &vT, boost::intrusive_ptr<Storage_base> &S, const cytnx_int32 &M, const cytnx_int32 &N){
            signed char jobu, jobv;

            // if U and vT are NULL ptr, then it will not be computed. 
            jobu = ( U->dtype == Type.Void ) ? 'N' : 'S';
            jobv = ( vT->dtype == Type.Void ) ? 'N' : 'S';


            // create handles:
            cusolverDnHandle_t cusolverH = NULL;
            checkCudaErrors(cusolverDnCreate(&cusolverH));

            cytnx_double* Mij;
            checkCudaErrors(cudaMalloc((void**)&Mij,M * N * sizeof(cytnx_double)));
            checkCudaErrors(cudaMemcpy(Mij,in->Mem,sizeof(cytnx_double)*M*N,cudaMemcpyDeviceToDevice));

            cytnx_int32 min = std::min(M, N);
            cytnx_int32 ldA = N, ldu = N, ldvT = min;
            cytnx_int32 lwork = 0;

            // query working space :
            checkCudaErrors(cusolverDnDgesvd_bufferSize(cusolverH, N, M, &lwork));


            // allocate working space:
            cytnx_double *work;
            cytnx_double *rwork=NULL;
            checkCudaErrors(cudaMalloc((void**)&work,lwork*sizeof(cytnx_double)));
            //checkCudaErrors(cudaMalloc((void**)&rwork,(min-1)*sizeof(cytnx_double64)));    

            cytnx_int32 *devinfo ;
            checkCudaErrors(cudaMalloc((void**)&devinfo,sizeof(cytnx_int32)));
            checkCudaErrors(cudaMemset(devinfo,0,sizeof(cytnx_int32)));

            cytnx_int32 info;
            /// compute:
            checkCudaErrors(cusolverDnDgesvd(cusolverH,jobv,jobu,N,M,Mij,ldA,(cytnx_double*)S->Mem,(cytnx_double*)vT->Mem,ldu,(cytnx_double*)U->Mem,ldvT,work,lwork,rwork,devinfo));

            // get info
            checkCudaErrors(cudaMemcpy(&info,devinfo,sizeof(cytnx_int32),cudaMemcpyDeviceToHost));

            cytnx_error_msg(info != 0, "%s %d", "Error in cuBlas function 'cusolverDnDgesvd': cuBlas INFO = ", info);


            cudaFree(work);
            cudaFree(Mij);
            cudaFree(devinfo);
            cusolverDnDestroy(cusolverH);
        }
        void cuSvd_internal_f( const boost::intrusive_ptr<Storage_base> &in, boost::intrusive_ptr<Storage_base> &U, boost::intrusive_ptr<Storage_base> &vT, boost::intrusive_ptr<Storage_base> &S, const cytnx_int32 &M, const cytnx_int32 &N){
            signed char jobu, jobv;

            // if U and vT are NULL ptr, then it will not be computed. 
            jobu = ( U->dtype == Type.Void ) ? 'N' : 'S';
            jobv = ( vT->dtype == Type.Void ) ? 'N' : 'S';


            // create handles:
            cusolverDnHandle_t cusolverH = NULL;
            checkCudaErrors(cusolverDnCreate(&cusolverH));

            cytnx_float* Mij;
            checkCudaErrors(cudaMalloc((void**)&Mij,M * N * sizeof(cytnx_float)));
            checkCudaErrors(cudaMemcpy(Mij,in->Mem,sizeof(cytnx_float)*M*N,cudaMemcpyDeviceToDevice));

            cytnx_int32 min = std::min(M, N);
            cytnx_int32 ldA = N, ldu = N, ldvT = min;
            cytnx_int32 lwork = 0;

            // query working space :
            checkCudaErrors(cusolverDnSgesvd_bufferSize(cusolverH, N, M, &lwork));


            // allocate working space:
            cytnx_float *work;
            cytnx_float *rwork=NULL;
            checkCudaErrors(cudaMalloc((void**)&work,lwork*sizeof(cytnx_float)));
            //checkCudaErrors(cudaMalloc((void**)&rwork,(min-1)*sizeof(cytnx_float64)));    

            cytnx_int32 *devinfo ;
            checkCudaErrors(cudaMalloc((void**)&devinfo,sizeof(cytnx_int32)));
            checkCudaErrors(cudaMemset(devinfo,0,sizeof(cytnx_int32)));

            cytnx_int32 info;
            /// compute:
            checkCudaErrors(cusolverDnSgesvd(cusolverH,jobv,jobu,N,M,Mij,ldA,(cytnx_float*)S->Mem,(cytnx_float*)vT->Mem,ldu,(cytnx_float*)U->Mem,ldvT,work,lwork,rwork,devinfo));

            // get info
            checkCudaErrors(cudaMemcpy(&info,devinfo,sizeof(cytnx_int32),cudaMemcpyDeviceToHost));

            cytnx_error_msg(info != 0, "%s %d", "Error in cuBlas function 'cusolverDnSgesvd': cuBlas INFO = ", info);


            cudaFree(work);
            cudaFree(Mij);
            cudaFree(devinfo);
            cusolverDnDestroy(cusolverH);

        }


    }//linalg_internal 
}//cytnx



