#include "cuReduce_gpu.hpp";
#include "utils/complex_arithmetic.hpp";

namespace cytnx{
    namespace utils_internal{

        #define _TNinB_REDUCE_ 512


        template <class X> 
        __device__ void warp_unroll(volatile X *smem,int thidx){

          smem[thidx]+=smem[thidx + 32];
          smem[thidx]+=smem[thidx + 16];
          smem[thidx]+=smem[thidx + 8 ];
          smem[thidx]+=smem[thidx + 4 ];
          smem[thidx]+=smem[thidx + 2 ];
          smem[thidx]+=smem[thidx + 1 ];

        }


        // require, threads per block to be 32*(2^n), n =0,1,2,3,4,5 
        template<class T>
        __global__ void cuReduce_kernel(T* out, T* in, cytnx_uint64 Nelem){
            __shared__ T sD[_TNinB_REDUCE_]; // allocate share mem for each thread
            sD[threadIdx.x] = 0;

            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){
                sD[threadIdx.x] = in[blockIdx.x*blockDim.x + threadIdx.x];
            }
            __syncthreads();

            if(blockDim.x >= 1024){  if(threadIdx.x < 512){sD[threadIdx.x] += sD[threadIdx.x + 512];} __syncthreads();}
	        if(blockDim.x >= 512 ){  if(threadIdx.x < 256){sD[threadIdx.x] += sD[threadIdx.x + 256];} __syncthreads();}
	        if(blockDim.x >= 256 ){  if(threadIdx.x < 128){sD[threadIdx.x] += sD[threadIdx.x + 128];} __syncthreads();}
	        if(blockDim.x >= 128 ){  if(threadIdx.x < 64 ){sD[threadIdx.x] += sD[threadIdx.x + 64 ];} __syncthreads();}

	        if(threadIdx.x<32)
	            warp_unroll(sD,threadIdx.x);
	        __syncthreads();

            if(threadIdx.x==0)
                out[blockIdx.x] = sD[0]; // write to global for block

        }
        //=======================

        
        __device__ void warp_unroll(volatile cuDoubleComplex *smem,int thidx){

          smem[thidx].x += smem[thidx + 32].x; smem[thidx].y += smem[thidx + 32].y;
          smem[thidx].x += smem[thidx + 16].x; smem[thidx].y += smem[thidx + 16].y;
          smem[thidx].x += smem[thidx + 8].x; smem[thidx].y += smem[thidx + 8].y;
          smem[thidx].x += smem[thidx + 4].x; smem[thidx].y += smem[thidx + 4].y;
          smem[thidx].x += smem[thidx + 2].x; smem[thidx].y += smem[thidx + 2].y;
          smem[thidx].x += smem[thidx + 1].x; smem[thidx].y += smem[thidx + 1].y;

        }
        __global__ void cuReduce_kernel_cd(cuDoubleComplex* out, cuDoubleComplex* in, cytnx_uint64 Nelem){
            __shared__ cuDoubleComplex sD[_TNinB_REDUCE_]; // allocate share mem for each thread
            sD[threadIdx.x] = make_cuDoubleComplex(0,0);

            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){
                sD[threadIdx.x] = in[blockIdx.x*blockDim.x + threadIdx.x];
            }
            __syncthreads();

            if(blockDim.x >= 1024){  if(threadIdx.x < 512){
                                        sD[threadIdx.x].x += sD[threadIdx.x + 512].x;
                                        sD[threadIdx.x].y += sD[threadIdx.x + 512].y;
                                     } __syncthreads();}
            if(blockDim.x >= 512){  if(threadIdx.x < 256){
                                        sD[threadIdx.x].x += sD[threadIdx.x + 256].x;
                                        sD[threadIdx.x].y += sD[threadIdx.x + 256].y;
                                     } __syncthreads();}
            if(blockDim.x >= 256){  if(threadIdx.x < 128){
                                        sD[threadIdx.x].x += sD[threadIdx.x + 128].x;
                                        sD[threadIdx.x].y += sD[threadIdx.x + 128].y;
                                     } __syncthreads();}
            if(blockDim.x >= 128){  if(threadIdx.x < 64){
                                        sD[threadIdx.x].x += sD[threadIdx.x + 64].x;
                                        sD[threadIdx.x].y += sD[threadIdx.x + 64].y;
                                     } __syncthreads();}

	        if(threadIdx.x<32)
	            warp_unroll(sD,threadIdx.x);
	        __syncthreads();

            if(threadIdx.x==0)
                out[blockIdx.x] = sD[0]; // write to global for block

        }
        
        __device__ void warp_unroll(volatile cuFloatComplex *smem,int thidx){

          smem[thidx].x += smem[thidx + 32].x; smem[thidx].y += smem[thidx + 32].y;
          smem[thidx].x += smem[thidx + 16].x; smem[thidx].y += smem[thidx + 16].y;
          smem[thidx].x += smem[thidx + 8].x; smem[thidx].y += smem[thidx + 8].y;
          smem[thidx].x += smem[thidx + 4].x; smem[thidx].y += smem[thidx + 4].y;
          smem[thidx].x += smem[thidx + 2].x; smem[thidx].y += smem[thidx + 2].y;
          smem[thidx].x += smem[thidx + 1].x; smem[thidx].y += smem[thidx + 1].y;
        }
        __global__ void cuReduce_kernel_cf(cuFloatComplex* out, cuFloatComplex* in, cytnx_uint64 Nelem){
            __shared__ cuFloatComplex sD[_TNinB_REDUCE_]; // allocate share mem for each thread
            sD[threadIdx.x] = make_cuFloatComplex(0,0);

            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){
                sD[threadIdx.x] = in[blockIdx.x*blockDim.x + threadIdx.x];
            }
            __syncthreads();

            if(blockDim.x >= 1024){  if(threadIdx.x < 512){
                                        sD[threadIdx.x].x += sD[threadIdx.x + 512].x;
                                        sD[threadIdx.x].y += sD[threadIdx.x + 512].y;
                                     } __syncthreads();}
            if(blockDim.x >= 512){  if(threadIdx.x < 256){
                                        sD[threadIdx.x].x += sD[threadIdx.x + 256].x;
                                        sD[threadIdx.x].y += sD[threadIdx.x + 256].y;
                                     } __syncthreads();}
            if(blockDim.x >= 256){  if(threadIdx.x < 128){
                                        sD[threadIdx.x].x += sD[threadIdx.x + 128].x;
                                        sD[threadIdx.x].y += sD[threadIdx.x + 128].y;
                                     } __syncthreads();}
            if(blockDim.x >= 128){  if(threadIdx.x < 64){
                                        sD[threadIdx.x].x += sD[threadIdx.x + 64].x;
                                        sD[threadIdx.x].y += sD[threadIdx.x + 64].y;
                                     } __syncthreads();}

	        if(threadIdx.x<32)
	            warp_unroll(sD,threadIdx.x);
	        __syncthreads();

            if(threadIdx.x==0)
                out[blockIdx.x] = sD[0]; // write to global for block

        }


        template<class T>
        void swap(T* &a, T* &b){
            T *tmp = a;
            a = b;
            b = tmp; 
        }

        void cuReduce_gpu_d(double* out, double* in, const cytnx_uint64 &Nelem){
            //cytnx_double * outptr = (cytnx_double*)out->Mem;
            //cytnx_double * ptr = (cytnx_double*)in->Mem;
            cytnx_uint64 Nelems = Nelem;
            cytnx_uint64 NBlocks;

            NBlocks = Nelems/_TNinB_REDUCE_;
            if(Nelems%_TNinB_REDUCE_) NBlocks+=1;

            //alloc mem for each block:
            cytnx_double *dblk;
            //std::cout << NBlocks*sizeof(cytnx_double) << std::endl;
            cudaMalloc((void**)&dblk,NBlocks*sizeof(cytnx_double));
            
              
            if(NBlocks==1){
                cuReduce_kernel<<<NBlocks,_TNinB_REDUCE_>>>(out,in,Nelems);
            }else{
                cuReduce_kernel<<<NBlocks,_TNinB_REDUCE_>>>(dblk,in,Nelems);
            }
            Nelems = NBlocks;

            while(Nelems>1){
                NBlocks = Nelems/_TNinB_REDUCE_;
                if(Nelems%_TNinB_REDUCE_) NBlocks+=1;

                if(NBlocks==1){
                    cuReduce_kernel<<<NBlocks,_TNinB_REDUCE_>>>(out,dblk,Nelems);
                }else{
                    cytnx_double *dblk2;
                    cudaMalloc((void**)&dblk2,NBlocks*sizeof(cytnx_double));
                    // do something:
                    cuReduce_kernel<<<NBlocks,_TNinB_REDUCE_>>>(dblk2,dblk,Nelems);

                    swap(dblk2,dblk); //swap new data to old data, and free the old
                    cudaFree(dblk2);
                }
                Nelems = NBlocks;
            }
            cudaFree(dblk);

            
        }

        void cuReduce_gpu_f(float* out, float* in, const cytnx_uint64 &Nelem){
            cytnx_uint64 Nelems = Nelem;
            cytnx_uint64 NBlocks;

            NBlocks = Nelems/_TNinB_REDUCE_;
            if(Nelems%_TNinB_REDUCE_) NBlocks+=1;

            //alloc mem for each block:
            cytnx_float *dblk;
            //std::cout << NBlocks*sizeof(cytnx_double) << std::endl;
            cudaMalloc((void**)&dblk,NBlocks*sizeof(cytnx_float));
            
              
            if(NBlocks==1){
                cuReduce_kernel<<<NBlocks,_TNinB_REDUCE_>>>(out,in,Nelems);
            }else{
                cuReduce_kernel<<<NBlocks,_TNinB_REDUCE_>>>(dblk,in,Nelems);
            }
            Nelems = NBlocks;

            while(Nelems>1){
                NBlocks = Nelems/_TNinB_REDUCE_;
                if(Nelems%_TNinB_REDUCE_) NBlocks+=1;

                if(NBlocks==1){
                    cuReduce_kernel<<<NBlocks,_TNinB_REDUCE_>>>(out,dblk,Nelems);
                }else{
                    cytnx_float *dblk2;
                    cudaMalloc((void**)&dblk2,NBlocks*sizeof(cytnx_float));
                    // do something:
                    cuReduce_kernel<<<NBlocks,_TNinB_REDUCE_>>>(dblk2,dblk,Nelems);

                    swap(dblk2,dblk); //swap new data to old data, and free the old
                    cudaFree(dblk2);
                }
                Nelems = NBlocks;
            }
            cudaFree(dblk);

            
        }

        void cuReduce_gpu_cf(cytnx_complex64* out, cytnx_complex64* in, const cytnx_uint64 &Nelem){
            cytnx_uint64 Nelems = Nelem;
            cytnx_uint64 NBlocks;

            NBlocks = Nelems/_TNinB_REDUCE_;
            if(Nelems%_TNinB_REDUCE_) NBlocks+=1;

            //alloc mem for each block:
            cuFloatComplex *dblk;
            //std::cout << NBlocks*sizeof(cytnx_double) << std::endl;
            cudaMalloc((void**)&dblk,NBlocks*sizeof(cuFloatComplex));
            
              
            if(NBlocks==1){
                cuReduce_kernel_cf<<<NBlocks,_TNinB_REDUCE_>>>((cuFloatComplex*)out,(cuFloatComplex*)in,Nelems);
            }else{
                cuReduce_kernel_cf<<<NBlocks,_TNinB_REDUCE_>>>(dblk,(cuFloatComplex*)in,Nelems);
            }
            Nelems = NBlocks;

            while(Nelems>1){
                NBlocks = Nelems/_TNinB_REDUCE_;
                if(Nelems%_TNinB_REDUCE_) NBlocks+=1;

                if(NBlocks==1){
                    cuReduce_kernel_cf<<<NBlocks,_TNinB_REDUCE_>>>((cuFloatComplex*)out,dblk,Nelems);
                }else{
                    cuFloatComplex *dblk2;
                    cudaMalloc((void**)&dblk2,NBlocks*sizeof(cuFloatComplex));
                    // do something:
                    cuReduce_kernel_cf<<<NBlocks,_TNinB_REDUCE_>>>(dblk2,dblk,Nelems);

                    swap(dblk2,dblk); //swap new data to old data, and free the old
                    cudaFree(dblk2);
                }
                Nelems = NBlocks;
            }
            cudaFree(dblk);

            
        }

        void cuReduce_gpu_cd(cytnx_complex128* out, cytnx_complex128* in, const cytnx_uint64 &Nelem){
            cytnx_uint64 Nelems = Nelem;
            cytnx_uint64 NBlocks;

            NBlocks = Nelems/_TNinB_REDUCE_;
            if(Nelems%_TNinB_REDUCE_) NBlocks+=1;

            //alloc mem for each block:
            cuDoubleComplex *dblk;
            //std::cout << NBlocks*sizeof(cytnx_double) << std::endl;
            cudaMalloc((void**)&dblk,NBlocks*sizeof(cuDoubleComplex));
            
              
            if(NBlocks==1){
                cuReduce_kernel_cd<<<NBlocks,_TNinB_REDUCE_>>>((cuDoubleComplex*)out,(cuDoubleComplex*)in,Nelems);
            }else{
                cuReduce_kernel_cd<<<NBlocks,_TNinB_REDUCE_>>>(dblk,(cuDoubleComplex*)in,Nelems);
            }
            Nelems = NBlocks;

            while(Nelems>1){
                NBlocks = Nelems/_TNinB_REDUCE_;
                if(Nelems%_TNinB_REDUCE_) NBlocks+=1;

                if(NBlocks==1){
                    cuReduce_kernel_cd<<<NBlocks,_TNinB_REDUCE_>>>((cuDoubleComplex*)out,dblk,Nelems);
                }else{
                    cuDoubleComplex *dblk2;
                    cudaMalloc((void**)&dblk2,NBlocks*sizeof(cuDoubleComplex));
                    // do something:
                    cuReduce_kernel_cd<<<NBlocks,_TNinB_REDUCE_>>>(dblk2,dblk,Nelems);

                    swap(dblk2,dblk); //swap new data to old data, and free the old
                    cudaFree(dblk2);
                }
                Nelems = NBlocks;
            }
            cudaFree(dblk);

            
        }




    }
    
}
