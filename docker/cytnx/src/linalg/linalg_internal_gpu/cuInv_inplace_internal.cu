#include "linalg/linalg_internal_gpu/cuInv_inplace_internal.hpp"
#include "cytnx_error.hpp"
#include "Type.hpp"
#include "utils/lapack_wrapper.hpp"


namespace cytnx{
    namespace linalg_internal{

        void cuInv_inplace_internal_d(boost::intrusive_ptr<Storage_base> &ten, const cytnx_int32 &L){

            // create handles:
            cusolverDnHandle_t cusolverH = NULL;
            checkCudaErrors(cusolverDnCreate(&cusolverH));


            cytnx_int32 *ipiv;
            cytnx_int32 info;
            cytnx_int32 lwork=0;
            cytnx_double *d_work = NULL;
            cytnx_int32 *devinfo ;
            checkCudaErrors(cudaMalloc((void**)&ipiv, (L+1)*sizeof(cytnx_int32)));
            checkCudaErrors(cudaMalloc((void**)&devinfo, sizeof(cytnx_int32)));
            //trf:
            checkCudaErrors(cusolverDnDgetrf_bufferSize(cusolverH,L,L,(cytnx_double*)ten->Mem,L,&lwork));
            checkCudaErrors(cudaMalloc((void**)&d_work,sizeof(cytnx_double)*lwork));

            checkCudaErrors(cusolverDnDgetrf(cusolverH,L,L,(cytnx_double*)ten->Mem,L,d_work,ipiv,devinfo));            
            checkCudaErrors(cudaMemcpy(&info,devinfo,sizeof(cytnx_int32),cudaMemcpyDeviceToHost));

            cytnx_error_msg(info != 0, "%s %d", "ERROR in cuSolver function 'cusolverDnDgetrf': cuBlas INFO = ", info);


            // trs AX = B with B = I
            lwork =0;
            cytnx_double *d_I;
            checkCudaErrors(cudaMalloc((void**)&d_I,sizeof(cytnx_double)*L*L));
            cytnx_double *h_I = (cytnx_double*)calloc(L*L,sizeof(cytnx_double));
            for(cytnx_uint64 i=0;i<L;i++)
                h_I[i*L+i] = 1;

            checkCudaErrors(cudaMemcpy(d_I,h_I,sizeof(cytnx_double)*L*L,cudaMemcpyHostToDevice));
            
            checkCudaErrors(cusolverDnDgetrs(cusolverH,CUBLAS_OP_N,L,L,(cytnx_double*)ten->Mem,L,ipiv,d_I,L,devinfo));
            checkCudaErrors(cudaMemcpy(&info,devinfo,sizeof(cytnx_int32),cudaMemcpyDeviceToHost));

            cytnx_error_msg(info != 0, "%s %d", "ERROR in cuSolver function 'cusolverDnDgetrs': cuBlas INFO = ", info);
    
            checkCudaErrors(cudaMemcpy(ten->Mem,d_I,sizeof(cytnx_double)*L*L,cudaMemcpyDeviceToDevice));
            
            cudaFree(d_I);
            cudaFree(d_work);
            cudaFree(devinfo);
            cudaFree(ipiv);
            free(h_I);
            cusolverDnDestroy(cusolverH);

        }
        void cuInv_inplace_internal_f(boost::intrusive_ptr<Storage_base> &ten, const cytnx_int32 &L){

            // create handles:
            cusolverDnHandle_t cusolverH = NULL;
            checkCudaErrors(cusolverDnCreate(&cusolverH));


            cytnx_int32 *ipiv;
            cytnx_int32 info;
            cytnx_int32 lwork=0;
            cytnx_float *d_work = NULL;
            cytnx_int32 *devinfo ;
            checkCudaErrors(cudaMalloc((void**)&ipiv, (L+1)*sizeof(cytnx_int32)));
            checkCudaErrors(cudaMalloc((void**)&devinfo, sizeof(cytnx_int32)));
            //trf:
            checkCudaErrors(cusolverDnSgetrf_bufferSize(cusolverH,L,L,(cytnx_float*)ten->Mem,L,&lwork));
            checkCudaErrors(cudaMalloc((void**)&d_work,sizeof(cytnx_float)*lwork));

            checkCudaErrors(cusolverDnSgetrf(cusolverH,L,L,(cytnx_float*)ten->Mem,L,d_work,ipiv,devinfo));            
            checkCudaErrors(cudaMemcpy(&info,devinfo,sizeof(cytnx_int32),cudaMemcpyDeviceToHost));

            cytnx_error_msg(info != 0, "%s %d", "ERROR in cuSolver function 'cusolverDnSgetrf': cuBlas INFO = ", info);


            // trs AX = B with B = I
            lwork =0;
            cytnx_float *d_I;
            checkCudaErrors(cudaMalloc((void**)&d_I,sizeof(cytnx_float)*L*L));
            cytnx_float *h_I = (cytnx_float*)calloc(L*L,sizeof(cytnx_float));
            for(cytnx_uint64 i=0;i<L;i++)
                h_I[i*L+i] = 1;

            checkCudaErrors(cudaMemcpy(d_I,h_I,sizeof(cytnx_float)*L*L,cudaMemcpyHostToDevice));
            
            checkCudaErrors(cusolverDnSgetrs(cusolverH,CUBLAS_OP_N,L,L,(cytnx_float*)ten->Mem,L,ipiv,d_I,L,devinfo));
            checkCudaErrors(cudaMemcpy(&info,devinfo,sizeof(cytnx_int32),cudaMemcpyDeviceToHost));

            cytnx_error_msg(info != 0, "%s %d", "ERROR in cuSolver function 'cusolverDnSgetrs': cuBlas INFO = ", info);
    
            checkCudaErrors(cudaMemcpy(ten->Mem,d_I,sizeof(cytnx_float)*L*L,cudaMemcpyDeviceToDevice));
            
            cudaFree(d_I);
            cudaFree(d_work);
            cudaFree(devinfo);
            cudaFree(ipiv);
            free(h_I);
            cusolverDnDestroy(cusolverH);
        }
        void cuInv_inplace_internal_cd(boost::intrusive_ptr<Storage_base> &ten,  const cytnx_int32 &L){

            // create handles:
            cusolverDnHandle_t cusolverH = NULL;
            checkCudaErrors(cusolverDnCreate(&cusolverH));


            cytnx_int32 *ipiv;
            cytnx_int32 info;
            cytnx_int32 lwork=0;
            cytnx_complex128 *d_work = NULL;
            cytnx_int32 *devinfo ;
            checkCudaErrors(cudaMalloc((void**)&ipiv, (L+1)*sizeof(cytnx_int32)));
            checkCudaErrors(cudaMalloc((void**)&devinfo, sizeof(cytnx_int32)));
            //trf:
            checkCudaErrors(cusolverDnZgetrf_bufferSize(cusolverH,L,L,(cuDoubleComplex*)ten->Mem,L,&lwork));
            checkCudaErrors(cudaMalloc((void**)&d_work,sizeof(cytnx_complex128)*lwork));

            checkCudaErrors(cusolverDnZgetrf(cusolverH,L,L,(cuDoubleComplex*)ten->Mem,L,(cuDoubleComplex*)d_work,ipiv,devinfo));            
            checkCudaErrors(cudaMemcpy(&info,devinfo,sizeof(cytnx_int32),cudaMemcpyDeviceToHost));

            cytnx_error_msg(info != 0, "%s %d", "ERROR in cuSolver function 'cusolverDnZgetrf': cuBlas INFO = ", info);


            // trs AX = B with B = I
            lwork =0;
            cytnx_complex128 *d_I;
            checkCudaErrors(cudaMalloc((void**)&d_I,sizeof(cytnx_complex128)*L*L));
            cytnx_complex128 *h_I = (cytnx_complex128*)calloc(L*L,sizeof(cytnx_complex128));
            for(cytnx_uint64 i=0;i<L;i++)
                h_I[i*L+i] = cytnx_complex128(1,0);

            checkCudaErrors(cudaMemcpy(d_I,h_I,sizeof(cytnx_complex128)*L*L,cudaMemcpyHostToDevice));
            
            checkCudaErrors(cusolverDnZgetrs(cusolverH,CUBLAS_OP_N,L,L,(cuDoubleComplex*)ten->Mem,L,ipiv,(cuDoubleComplex*)d_I,L,devinfo));
            checkCudaErrors(cudaMemcpy(&info,devinfo,sizeof(cytnx_int32),cudaMemcpyDeviceToHost));

            cytnx_error_msg(info != 0, "%s %d", "ERROR in cuSolver function 'cusolverDnZgetrs': cuBlas INFO = ", info);
    
            checkCudaErrors(cudaMemcpy(ten->Mem,d_I,sizeof(cytnx_complex128)*L*L,cudaMemcpyDeviceToDevice));
            
            cudaFree(d_I);
            cudaFree(d_work);
            cudaFree(devinfo);
            cudaFree(ipiv);
            free(h_I);
            cusolverDnDestroy(cusolverH);
        }

        void cuInv_inplace_internal_cf(boost::intrusive_ptr<Storage_base> &ten,  const cytnx_int32 &L){

            // create handles:
            cusolverDnHandle_t cusolverH = NULL;
            checkCudaErrors(cusolverDnCreate(&cusolverH));


            cytnx_int32 *ipiv;
            cytnx_int32 info;
            cytnx_int32 lwork=0;
            cytnx_complex64 *d_work = NULL;
            cytnx_int32 *devinfo ;
            checkCudaErrors(cudaMalloc((void**)&ipiv, (L+1)*sizeof(cytnx_int32)));
            checkCudaErrors(cudaMalloc((void**)&devinfo, sizeof(cytnx_int32)));
            //trf:
            checkCudaErrors(cusolverDnCgetrf_bufferSize(cusolverH,L,L,(cuFloatComplex*)ten->Mem,L,&lwork));
            checkCudaErrors(cudaMalloc((void**)&d_work,sizeof(cytnx_complex64)*lwork));

            checkCudaErrors(cusolverDnCgetrf(cusolverH,L,L,(cuFloatComplex*)ten->Mem,L,(cuFloatComplex*)d_work,ipiv,devinfo));            
            checkCudaErrors(cudaMemcpy(&info,devinfo,sizeof(cytnx_int32),cudaMemcpyDeviceToHost));

            cytnx_error_msg(info != 0, "%s %d", "ERROR in cuSolver function 'cusolverDnCgetrf': cuBlas INFO = ", info);


            // trs AX = B with B = I
            lwork =0;
            cytnx_complex64 *d_I;
            checkCudaErrors(cudaMalloc((void**)&d_I,sizeof(cytnx_complex64)*L*L));
            cytnx_complex64 *h_I = (cytnx_complex64*)calloc(L*L,sizeof(cytnx_complex64));
            for(cytnx_uint64 i=0;i<L;i++)
                h_I[i*L+i] = cytnx_complex64(1,0);

            checkCudaErrors(cudaMemcpy(d_I,h_I,sizeof(cytnx_complex64)*L*L,cudaMemcpyHostToDevice));
            
            checkCudaErrors(cusolverDnCgetrs(cusolverH,CUBLAS_OP_N,L,L,(cuFloatComplex*)ten->Mem,L,ipiv,(cuFloatComplex*)d_I,L,devinfo));
            checkCudaErrors(cudaMemcpy(&info,devinfo,sizeof(cytnx_int32),cudaMemcpyDeviceToHost));

            cytnx_error_msg(info != 0, "%s %d", "ERROR in cuSolver function 'cusolverDnCgetrs': cuBlas INFO = ", info);
    
            checkCudaErrors(cudaMemcpy(ten->Mem,d_I,sizeof(cytnx_complex64)*L*L,cudaMemcpyDeviceToDevice));
            
            cudaFree(d_I);
            cudaFree(d_work);
            cudaFree(devinfo);
            cudaFree(ipiv);
            free(h_I);
            cusolverDnDestroy(cusolverH);
        }



    }// namespace linalg_internal

}// namespace cytnx


