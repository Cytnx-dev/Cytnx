#include "utils/utils_internal_gpu/cuCast_gpu.hpp"
#include "Storage.hpp"
#ifdef UNI_OMP
#include <omp.h>
#endif

using namespace std;
namespace cytnx{
    namespace utils_internal{

        template<class T3>
        __global__ void cuFill_kernel(const T3 *src, T3 val, cytnx_uint64 Nelem){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){
                des[blockIdx.x*blockDim.x + threadIdx.x] = val;
            }
        }

        //========================================================================
        void cuCast_gpu_cd(void* in, void* val, const cytnx_uint64 &Nelem){
            cuDoubleComplex* ptr = (cuDoubleComplex*)in;
            cuDoubleComplex _val = *((cuDoubleComplex*)elem);
            
            cytnx_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel<<<NBlocks,512>>>(ptr,_val,Nelem);
        }

        void cuCast_gpu_cf(void* in, void* val, const cytnx_uint64 &Nelem){
            cuFloatComplex* ptr = (cuFloatComplex*)in;
            cuFloatComplex _val = *((cuFloatComplex*)elem);
            
            cytnx_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel<<<NBlocks,512>>>(ptr,_val,Nelem);
        }

        void cuCast_gpu_d(void* in, void* val, const cytnx_uint64 &Nelem){
            cytnx_double* ptr = (cytnx_double*)in;
            cytnx_double _val = *((cytnx_double*)elem);
            
            cytnx_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel<<<NBlocks,512>>>(ptr,_val,Nelem);
        }
        
        void cuCast_gpu_f(void* in, void* val, const cytnx_uint64 &Nelem){
            cytnx_float* ptr = (cytnx_float*)in;
            cytnx_float _val = *((cytnx_float*)elem);
            
            cytnx_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel<<<NBlocks,512>>>(ptr,_val,Nelem);
        }
        
        void cuCast_gpu_i64(void* in, void* val, const cytnx_uint64 &Nelem){
            cytnx_int64* ptr = (cytnx_int64*)in;
            cytnx_int64 _val = *((cytnx_int64*)elem);
            
            cytnx_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel<<<NBlocks,512>>>(ptr,_val,Nelem);
        }
        
        void cuCast_gpu_u64(void* in, void* val, const cytnx_uint64 &Nelem){
            cytnx_uint64* ptr = (cytnx_uint64*)in;
            cytnx_uint64 _val = *((cytnx_uint64*)elem);
            
            cytnx_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel<<<NBlocks,512>>>(ptr,_val,Nelem);
        }
        
        void cuCast_gpu_i32(void* in, void* val, const cytnx_uint64 &Nelem){
            cytnx_int32* ptr = (cytnx_int32*)in;
            cytnx_int32 _val = *((cytnx_int32*)elem);
            
            cytnx_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel<<<NBlocks,512>>>(ptr,_val,Nelem);
        }
        
        void cuCast_gpu_u32(void* in, void* val, const cytnx_uint64 &Nelem){
            cytnx_uint32* ptr = (cytnx_uint32*)in;
            cytnx_uint32 _val = *((cytnx_uint32*)elem);
            
            cytnx_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel<<<NBlocks,512>>>(ptr,_val,Nelem);
        }
    }//namespace utils_internal
}//namespace cytnx
