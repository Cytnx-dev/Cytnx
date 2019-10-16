
#include "cuConj_inplace_internal.hpp"
#include "cytnx_error.hpp"
#include "Type.hpp"
#include "lapack_wrapper.hpp"



namespace cytnx{
 
    namespace linalg_internal{
        
        __global__ void cuConj_inplace_kernel(cuDoubleComplex *ten, const cytnx_uint64 Nelem){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){   
                ten[blockIdx.x*blockDim.x + threadIdx.x].y = -ten[blockIdx.x*blockDim.x + threadIdx.x].y;
               
            }
            __syncthreads();
        }

        __global__ void cuConj_inplace_kernel(cuFloatComplex *ten, const cytnx_uint64 Nelem){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){   
                ten[blockIdx.x*blockDim.x + threadIdx.x].y = -ten[blockIdx.x*blockDim.x + threadIdx.x].y;
            }
            __syncthreads();
        }

    }// namespace linalg_internal

}// cytnx




namespace cytnx{
    namespace linalg_internal{

        void cuConj_inplace_internal_cd(boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem){
            cytnx_uint32 NBlocks = Nelem/256;
            if(Nelem%256) NBlocks+=1;
            cuConj_inplace_kernel<<<NBlocks,256>>>((cuDoubleComplex*)ten->Mem,Nelem);
        }

        void cuConj_inplace_internal_cf(boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem){
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks+=1;
            cuConj_inplace_kernel<<<NBlocks,512>>>((cuFloatComplex*)ten->Mem,Nelem);
        }


    }// namespace linalg_internal

}// namespace cytnx


