#include "cuSetArange_gpu.hpp"


namespace cytnx{
    namespace utils_internal{

        template<class T>
        __global__ void cuSetArange_kernel(T* in, cytnx_double start, cytnx_double step, cytnx_uint64 Nelem){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){
                in[blockIdx.x*blockDim.x + threadIdx.x] = start + step*blockIdx.x*blockDim.x + threadIdx.x;
            }
        }
        __global__ void cuSetArange_kernel(cuDoubleComplex* in , cytnx_double start, cytnx_double step, cytnx_uint64 Nelem){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){
                in[blockIdx.x*blockDim.x + threadIdx.x] = make_cuDoubleComplex(start + step*(blockIdx.x*blockDim.x + threadIdx.x),0);
            }
        }
        __global__ void cuSetArange_kernel(cuFloatComplex* in , cytnx_double start, cytnx_double step, cytnx_uint64 Nelem){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){
                in[blockIdx.x*blockDim.x + threadIdx.x] = make_cuFloatComplex(start + step*(blockIdx.x*blockDim.x + threadIdx.x),0);
            }
        }


        // type = 0, start < end , incremental
        // type = 1, start > end , decremental 
        void cuSetArange_gpu_cd(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start, const cytnx_double &end, const cytnx_double &step, const cytnx_uint64 &Nelem){
            cuDoubleComplex * ptr = (cuDoubleComplex*)in->Mem;
            cytnx_uint64 NBlocks = Nelem/512;

            if(Nelem%512) NBlocks+=1;
            cuSetArange_kernel<<<NBlocks,512>>>(ptr,start,step,Nelem);
        }
        void cuSetArange_gpu_cf(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start, const cytnx_double &end, const cytnx_double &step, const cytnx_uint64 &Nelem){
            cuFloatComplex * ptr = (cuFloatComplex*)in->Mem;
            cytnx_uint64 NBlocks = Nelem/512;

            if(Nelem%512) NBlocks+=1;
            cuSetArange_kernel<<<NBlocks,512>>>(ptr,start,step,Nelem);
        }
        void cuSetArange_gpu_d(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start, const cytnx_double &end, const cytnx_double &step, const cytnx_uint64 &Nelem){
            cytnx_double * ptr = (cytnx_double*)in->Mem;
            cytnx_uint64 NBlocks = Nelem/512;

            if(Nelem%512) NBlocks+=1;
            cuSetArange_kernel<<<NBlocks,512>>>(ptr,start,step,Nelem);
        }
        void cuSetArange_gpu_f(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start, const cytnx_double &end, const cytnx_double &step, const cytnx_uint64 &Nelem){
            cytnx_float * ptr = (cytnx_float*)in->Mem;
            cytnx_uint64 NBlocks = Nelem/512;

            if(Nelem%512) NBlocks+=1;
            cuSetArange_kernel<<<NBlocks,512>>>(ptr,start,step,Nelem);
        }
        void cuSetArange_gpu_i64(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start, const cytnx_double &end, const cytnx_double &step, const cytnx_uint64 &Nelem){
            cytnx_int64 * ptr = (cytnx_int64*)in->Mem;
            cytnx_uint64 NBlocks = Nelem/512;

            if(Nelem%512) NBlocks+=1;
            cuSetArange_kernel<<<NBlocks,512>>>(ptr,start,step,Nelem);
        }
        void cuSetArange_gpu_u64(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start, const cytnx_double &end, const cytnx_double &step, const cytnx_uint64 &Nelem){
            cytnx_uint64 * ptr = (cytnx_uint64*)in->Mem;
            cytnx_uint64 NBlocks = Nelem/512;

            if(Nelem%512) NBlocks+=1;
            cuSetArange_kernel<<<NBlocks,512>>>(ptr,start,step,Nelem);

        }
        void cuSetArange_gpu_i32(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start, const cytnx_double &end, const cytnx_double &step, const cytnx_uint64 &Nelem){
            cytnx_int32 * ptr = (cytnx_int32*)in->Mem;
            cytnx_uint64 NBlocks = Nelem/512;

            if(Nelem%512) NBlocks+=1;
            cuSetArange_kernel<<<NBlocks,512>>>(ptr,start,step,Nelem);
        }
        void cuSetArange_gpu_u32(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start, const cytnx_double &end, const cytnx_double &step, const cytnx_uint64 &Nelem){
            cytnx_uint32 * ptr = (cytnx_uint32*)in->Mem;
            cytnx_uint64 NBlocks = Nelem/512;

            if(Nelem%512) NBlocks+=1;
            cuSetArange_kernel<<<NBlocks,512>>>(ptr,start,step,Nelem);
        }
        void cuSetArange_gpu_i16(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start, const cytnx_double &end, const cytnx_double &step, const cytnx_uint64 &Nelem){
            cytnx_int16 * ptr = (cytnx_int16*)in->Mem;
            cytnx_uint64 NBlocks = Nelem/512;

            if(Nelem%512) NBlocks+=1;
            cuSetArange_kernel<<<NBlocks,512>>>(ptr,start,step,Nelem);
        }
        void cuSetArange_gpu_u16(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start, const cytnx_double &end, const cytnx_double &step, const cytnx_uint64 &Nelem){
            cytnx_uint16 * ptr = (cytnx_uint16*)in->Mem;
            cytnx_uint64 NBlocks = Nelem/512;

            if(Nelem%512) NBlocks+=1;
            cuSetArange_kernel<<<NBlocks,512>>>(ptr,start,step,Nelem);
        }
        void cuSetArange_gpu_b(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start, const cytnx_double &end, const cytnx_double &step, const cytnx_uint64 &Nelem){
            cytnx_bool * ptr = (cytnx_bool*)in->Mem;
            cytnx_uint64 NBlocks = Nelem/512;

            if(Nelem%512) NBlocks+=1;
            cuSetArange_kernel<<<NBlocks,512>>>(ptr,start,step,Nelem);
        }

    }
}
