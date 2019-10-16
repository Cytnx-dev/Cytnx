#include "cuDiag_internal.hpp"
#include "utils/utils_internal_interface.hpp"

//#ifdef UNI_OMP
//    #include <omp.h>
//#endif


namespace cytnx{
 
    namespace linalg_internal{
        template<class T>
        __global__ void cuDiag_internal_kernel(T *out, const T *ten, const cytnx_uint64 L){
            
            if(blockIdx.x*blockDim.x + threadIdx.x < L){                           
                out[(blockIdx.x*blockDim.x + threadIdx.x)*L+blockIdx.x*blockDim.x + threadIdx.x] = ten[blockIdx.x*blockDim.x + threadIdx.x];                                                                           
            }
            __syncthreads();
        }


    }// namespace linalg_internal

}// cytnx




namespace cytnx{
    namespace linalg_internal{

        void cuDiag_internal_b(boost::intrusive_ptr<Storage_base> &out, const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L){
            cytnx_uint32 NBlocks = L/512;
            if(L%512) NBlocks+=1;
            cuDiag_internal_kernel<<<NBlocks,512>>>((cytnx_bool*)out->Mem,(cytnx_bool*)ten->Mem,L);
        }

        void cuDiag_internal_i16(boost::intrusive_ptr<Storage_base> &out, const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L){
            cytnx_uint32 NBlocks = L/512;
            if(L%512) NBlocks+=1;
            cuDiag_internal_kernel<<<NBlocks,512>>>((cytnx_int16*)out->Mem,(cytnx_int16*)ten->Mem,L);
        }

        void cuDiag_internal_u16(boost::intrusive_ptr<Storage_base> &out, const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L){
            cytnx_uint32 NBlocks = L/512;
            if(L%512) NBlocks+=1;
            cuDiag_internal_kernel<<<NBlocks,512>>>((cytnx_uint16*)out->Mem,(cytnx_uint16*)ten->Mem,L);
        }

        void cuDiag_internal_i32(boost::intrusive_ptr<Storage_base> &out, const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L){
            cytnx_uint32 NBlocks = L/512;
            if(L%512) NBlocks+=1;
            cuDiag_internal_kernel<<<NBlocks,512>>>((cytnx_int32*)out->Mem,(cytnx_int32*)ten->Mem,L);
        }

        void cuDiag_internal_u32(boost::intrusive_ptr<Storage_base> &out, const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L){
            cytnx_uint32 NBlocks = L/512;
            if(L%512) NBlocks+=1;
            cuDiag_internal_kernel<<<NBlocks,512>>>((cytnx_uint32*)out->Mem,(cytnx_uint32*)ten->Mem,L);
        }

        void cuDiag_internal_i64(boost::intrusive_ptr<Storage_base> &out, const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L){
            cytnx_uint32 NBlocks = L/512;
            if(L%512) NBlocks+=1;
            cuDiag_internal_kernel<<<NBlocks,512>>>((cytnx_int64*)out->Mem,(cytnx_int64*)ten->Mem,L);
        }

        void cuDiag_internal_u64(boost::intrusive_ptr<Storage_base> &out, const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L){
            cytnx_uint32 NBlocks = L/512;
            if(L%512) NBlocks+=1;
            cuDiag_internal_kernel<<<NBlocks,512>>>((cytnx_uint64*)out->Mem,(cytnx_uint64*)ten->Mem,L);
        }


        void cuDiag_internal_d(boost::intrusive_ptr<Storage_base> &out, const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L){
            cytnx_uint32 NBlocks = L/512;
            if(L%512) NBlocks+=1;
            cuDiag_internal_kernel<<<NBlocks,512>>>((cytnx_double*)out->Mem,(cytnx_double*)ten->Mem,L);
        }

        void cuDiag_internal_f(boost::intrusive_ptr<Storage_base> &out, const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L){
            cytnx_uint32 NBlocks = L/512;
            if(L%512) NBlocks+=1;
            cuDiag_internal_kernel<<<NBlocks,512>>>((cytnx_float*)out->Mem,(cytnx_float*)ten->Mem,L);
        }

        void cuDiag_internal_cd(boost::intrusive_ptr<Storage_base> &out, const boost::intrusive_ptr<Storage_base> &ten,  const cytnx_uint64 &L)
        {
            cytnx_uint32 NBlocks = L/256;
            if(L%256) NBlocks+=1;
            cuDiag_internal_kernel<<<NBlocks,256>>>((cuDoubleComplex*)out->Mem,(cuDoubleComplex*)ten->Mem,L);
        }

        void cuDiag_internal_cf(boost::intrusive_ptr<Storage_base> &out, const boost::intrusive_ptr<Storage_base> &ten,  const cytnx_uint64 &L)
        {
            cytnx_uint32 NBlocks = L/256;
            if(L%256) NBlocks+=1;
            cuDiag_internal_kernel<<<NBlocks,256>>>((cuFloatComplex*)out->Mem,(cuFloatComplex*)ten->Mem,L);
        }


    }// namespace linalg_internal

}// namespace cytnx


