#include <cuda.h>
#include <device_launch_parameters.h>
#include "cuOuter_internal.hpp"
#include "utils/utils_internal_interface.hpp"

#ifdef UNI_OMP
    #include <omp.h>
#endif

namespace cytnx{

    namespace linalg_internal{

        //====================================================================
        
        template<class T1,class T2,class T3>
        __global__ void cuOuter_kernel(T1 *out, const T2 *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL, const T3 *ptr){
              if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){
                  out[blockIdx.x*blockDim.x + threadIdx.x] = val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)] * ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL];
              }
              __syncthreads();
        }

        //=====================================================================

        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cuDoubleComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuDoubleComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_cdtcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cuDoubleComplex *_Lin = (cuDoubleComplex*)Lin->Mem;
            cuDoubleComplex *_Rin = (cuDoubleComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
            
        }



        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cuDoubleComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuFloatComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],cuComplexFloatToDouble(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]));
            }
            __syncthreads();
        }
        void cuOuter_internal_cdtcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cuDoubleComplex *_Lin = (cuDoubleComplex*)Lin->Mem;
            cuFloatComplex *_Rin = (cuFloatComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }

        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cuDoubleComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_double *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuDoubleComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cdtd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cuDoubleComplex *_Lin = (cuDoubleComplex*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;


            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }

        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cuDoubleComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_float *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuDoubleComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cdtf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cuDoubleComplex *_Lin = (cuDoubleComplex*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;


            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);



        }


        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cuDoubleComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_uint64 *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuDoubleComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cdtu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cuDoubleComplex *_Lin = (cuDoubleComplex*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }


        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cuDoubleComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_uint32 *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuDoubleComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cdtu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cuDoubleComplex *_Lin = (cuDoubleComplex*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }



        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cuDoubleComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_int64 *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuDoubleComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cdti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cuDoubleComplex *_Lin = (cuDoubleComplex*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);


        }


        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cuDoubleComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_int32 *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuDoubleComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cdti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cuDoubleComplex *_Lin = (cuDoubleComplex*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }

        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cuDoubleComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_int16 *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuDoubleComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cdti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cuDoubleComplex *_Lin = (cuDoubleComplex*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }

        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cuDoubleComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_uint16 *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuDoubleComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cdtu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cuDoubleComplex *_Lin = (cuDoubleComplex*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }

        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cuDoubleComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_bool *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuDoubleComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cdtb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cuDoubleComplex *_Lin = (cuDoubleComplex*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
//-------------------------------------------
        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cuFloatComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuDoubleComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(cuComplexFloatToDouble(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)]),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_cftcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cuFloatComplex *_Lin = (cuFloatComplex*)Lin->Mem;
            cuDoubleComplex *_Rin = (cuDoubleComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }

        __global__ void cuOuter_kernel(cuFloatComplex *out, const cuFloatComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuFloatComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_cftcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
            cuFloatComplex *_Lin = (cuFloatComplex*)Lin->Mem;
            cuFloatComplex *_Rin = (cuFloatComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }


        __global__ void cuOuter_kernel(cuFloatComplex *out, const cuFloatComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_double *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuFloatComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cftd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
            cuFloatComplex *_Lin = (cuFloatComplex*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }

        __global__ void cuOuter_kernel(cuFloatComplex *out, const cuFloatComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_float *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuFloatComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cftf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
            cuFloatComplex *_Lin = (cuFloatComplex*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);


        }


        __global__ void cuOuter_kernel(cuFloatComplex *out, const cuFloatComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_uint64 *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuFloatComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cftu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
            cuFloatComplex *_Lin = (cuFloatComplex*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;
            
            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }


        __global__ void cuOuter_kernel(cuFloatComplex *out, const cuFloatComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_uint32 *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuFloatComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cftu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
            cuFloatComplex *_Lin = (cuFloatComplex*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }


        __global__ void cuOuter_kernel(cuFloatComplex *out, const cuFloatComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_int64 *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuFloatComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cfti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
            cuFloatComplex *_Lin = (cuFloatComplex*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }


        __global__ void cuOuter_kernel(cuFloatComplex *out, const cuFloatComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_int32 *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuFloatComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cfti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
            cuFloatComplex *_Lin = (cuFloatComplex*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }

        __global__ void cuOuter_kernel(cuFloatComplex *out, const cuFloatComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_int16 *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuFloatComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cfti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
            cuFloatComplex *_Lin = (cuFloatComplex*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }      

        __global__ void cuOuter_kernel(cuFloatComplex *out, const cuFloatComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_uint16 *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuFloatComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cftu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
            cuFloatComplex *_Lin = (cuFloatComplex*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }  

        __global__ void cuOuter_kernel(cuFloatComplex *out, const cuFloatComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_bool *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuFloatComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cftb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
            cuFloatComplex *_Lin = (cuFloatComplex*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }        

//-----------------------------------------
        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cytnx_double *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuDoubleComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(make_cuDoubleComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_dtcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cuDoubleComplex *_Rin = (cuDoubleComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }


        __global__ void cuOuter_kernel(cuFloatComplex *out, const cytnx_double *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuFloatComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(make_cuFloatComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_dtcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cuFloatComplex *_Rin = (cuFloatComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }


        void cuOuter_internal_dtd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);


        }
        void cuOuter_internal_dtf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }
        void cuOuter_internal_dtu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_dtu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }
        void cuOuter_internal_dti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_dti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }
        void cuOuter_internal_dti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }

        void cuOuter_internal_dtu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }

        __global__ void cuOuter_kernel(cytnx_double *out, const cytnx_double *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_bool *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)] * cytnx_double(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_dtb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }

//--------------------------------------
        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cytnx_float *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuDoubleComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(make_cuDoubleComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_ftcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cuDoubleComplex *_Rin = (cuDoubleComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }


        __global__ void cuOuter_kernel(cuFloatComplex *out, const cytnx_float *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuFloatComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(make_cuFloatComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_ftcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cuFloatComplex *_Rin = (cuFloatComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }

        void cuOuter_internal_ftd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_ftf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_ftu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_ftu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }
        void cuOuter_internal_fti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_fti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }
        void cuOuter_internal_fti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }
        void cuOuter_internal_ftu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }

        __global__ void cuOuter_kernel(cytnx_float *out, const cytnx_float *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_bool *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)] * cytnx_float(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_ftb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }


//-------------------------------
        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cytnx_int64 *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuDoubleComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(make_cuDoubleComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_i64tcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cuDoubleComplex *_Rin = (cuDoubleComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }

        __global__ void cuOuter_kernel(cuFloatComplex *out, const cytnx_int64 *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuFloatComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(make_cuFloatComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_i64tcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cuFloatComplex *_Rin = (cuFloatComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }


        void cuOuter_internal_i64td(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }
        void cuOuter_internal_i64tf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }
        void cuOuter_internal_i64ti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_i64tu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }
        void cuOuter_internal_i64ti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }
        void cuOuter_internal_i64tu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }
        void cuOuter_internal_i64ti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }
        void cuOuter_internal_i64tu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }
        __global__ void cuOuter_kernel(cytnx_int64 *out, const cytnx_int64 *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_bool *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)] * cytnx_int64(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_i64tb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }


//----------------------------------------
        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cytnx_uint64 *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuDoubleComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(make_cuDoubleComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_u64tcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
             cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
             cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
             cuDoubleComplex  *_Rin = (cuDoubleComplex*)Rin->Mem;
  
            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }

        __global__ void cuOuter_kernel(cuFloatComplex *out, const cytnx_uint64 *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuFloatComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(make_cuFloatComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_u64tcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
               cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
               cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
               cuFloatComplex  *_Rin = (cuFloatComplex*)Rin->Mem;
  
               cytnx_uint64 Nelem = Lin->len*Rin->len;
               cytnx_uint32 NBlocks = Nelem/512;
               if(Nelem%512) NBlocks += 1;
                
               cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_u64td(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
               cytnx_double *_out = (cytnx_double*)out->Mem;
               cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
               cytnx_double  *_Rin = (cytnx_double*)Rin->Mem;
  
               cytnx_uint64 Nelem = Lin->len*Rin->len;
               cytnx_uint32 NBlocks = Nelem/512;
               if(Nelem%512) NBlocks += 1;
                
               cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }
        void cuOuter_internal_u64tf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
               cytnx_float *_out = (cytnx_float*)out->Mem;
               cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
               cytnx_float  *_Rin = (cytnx_float*)Rin->Mem;
  
               cytnx_uint64 Nelem = Lin->len*Rin->len;
               cytnx_uint32 NBlocks = Nelem/512;
               if(Nelem%512) NBlocks += 1;
                
               cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }
        void cuOuter_internal_u64ti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
               cytnx_int64 *_out = (cytnx_int64*)out->Mem;
               cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
               cytnx_int64  *_Rin = (cytnx_int64*)Rin->Mem;
  
               cytnx_uint64 Nelem = Lin->len*Rin->len;
               cytnx_uint32 NBlocks = Nelem/512;
               if(Nelem%512) NBlocks += 1;
                
               cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }
        void cuOuter_internal_u64tu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }
        void cuOuter_internal_u64ti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }
        void cuOuter_internal_u64tu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_u64ti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }
        void cuOuter_internal_u64tu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        __global__ void cuOuter_kernel(cytnx_uint64 *out, const cytnx_uint64 *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_bool *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)] * cytnx_uint64(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_u64tb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }

//----------------------------------------------


        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cytnx_int32 *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuDoubleComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(make_cuDoubleComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_i32tcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
              cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
              cuDoubleComplex *_Rin = (cuDoubleComplex*)Rin->Mem;
  
              cytnx_uint64 Nelem = Lin->len*Rin->len;
              cytnx_uint32 NBlocks = Nelem/512;
              if(Nelem%512) NBlocks += 1;
                
              cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);


        }

        __global__ void cuOuter_kernel(cuFloatComplex *out, const cytnx_int32 *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuFloatComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(make_cuFloatComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_i32tcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cuFloatComplex *_Rin = (cuFloatComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);


        }
        void cuOuter_internal_i32td(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_i32tf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);


        }
        void cuOuter_internal_i32ti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);


        }
        void cuOuter_internal_i32tu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);


        }
        void cuOuter_internal_i32ti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_i32tu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_i32ti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_int16*_Rin = (cytnx_int16*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_i32tu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        __global__ void cuOuter_kernel(cytnx_int32 *out, const cytnx_int32  *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_bool *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)] * cytnx_int32(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_i32tb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int32  *_out = (cytnx_int32 *)out->Mem;
            cytnx_int32  *_Lin = (cytnx_int32 *)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
//------------------------------

        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cytnx_uint32 *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuDoubleComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(make_cuDoubleComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_u32tcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cuDoubleComplex *_Rin = (cuDoubleComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);


        }

        __global__ void cuOuter_kernel(cuFloatComplex *out, const cytnx_uint32 *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuFloatComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(make_cuFloatComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_u32tcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
             cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
              cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
              cuFloatComplex *_Rin = (cuFloatComplex*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_u32td(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_double *_out = (cytnx_double*)out->Mem;
              cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
              cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_u32tf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_float *_out = (cytnx_float*)out->Mem;
              cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
              cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_u32ti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_int64 *_out = (cytnx_int64*)out->Mem;
              cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
              cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);


        }
        void cuOuter_internal_u32tu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
              cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
              cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_u32ti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_int32 *_out = (cytnx_int32*)out->Mem;
              cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
              cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_u32tu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint32 *_out = (cytnx_uint32*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }
        void cuOuter_internal_u32ti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_uint32 *_out = (cytnx_uint32*)out->Mem;
              cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
              cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_u32tu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint32 *_out = (cytnx_uint32*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }
        __global__ void cuOuter_kernel(cytnx_uint32 *out, const cytnx_uint32  *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_bool *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)] * cytnx_uint32(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_u32tb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint32  *_out = (cytnx_uint32 *)out->Mem;
            cytnx_uint32  *_Lin = (cytnx_uint32 *)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }

//------------------------------

        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cytnx_int16 *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuDoubleComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(make_cuDoubleComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_i16tcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
            cuDoubleComplex *_Rin = (cuDoubleComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);


        }

        __global__ void cuOuter_kernel(cuFloatComplex *out, const cytnx_int16 *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuFloatComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(make_cuFloatComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_i16tcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
             cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
              cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
              cuFloatComplex *_Rin = (cuFloatComplex*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_i16td(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_double *_out = (cytnx_double*)out->Mem;
              cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
              cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_i16tf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_float *_out = (cytnx_float*)out->Mem;
              cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
              cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_i16ti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_int64 *_out = (cytnx_int64*)out->Mem;
              cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
              cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);


        }
        void cuOuter_internal_i16tu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
              cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
              cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_i16ti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_int32 *_out = (cytnx_int32*)out->Mem;
              cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
              cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_i16tu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint32 *_out = (cytnx_uint32*)out->Mem;
            cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }
        void cuOuter_internal_i16ti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_int16 *_out = (cytnx_int16*)out->Mem;
              cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
              cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_i16tu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int16 *_out = (cytnx_int16*)out->Mem;
            cytnx_int16 *_Lin = (cytnx_int16*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }
        __global__ void cuOuter_kernel(cytnx_int16 *out, const cytnx_int16  *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_bool *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)] * cytnx_int16(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_i16tb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_int16  *_out = (cytnx_int16 *)out->Mem;
            cytnx_int16  *_Lin = (cytnx_int16 *)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }

//------------------------------

        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cytnx_uint16 *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuDoubleComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(make_cuDoubleComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_u16tcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
            cuDoubleComplex *_Rin = (cuDoubleComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);


        }

        __global__ void cuOuter_kernel(cuFloatComplex *out, const cytnx_uint16 *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuFloatComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(make_cuFloatComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_u16tcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
             cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
              cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
              cuFloatComplex *_Rin = (cuFloatComplex*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_u16td(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_double *_out = (cytnx_double*)out->Mem;
              cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
              cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_u16tf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_float *_out = (cytnx_float*)out->Mem;
              cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
              cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_u16ti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_int64 *_out = (cytnx_int64*)out->Mem;
              cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
              cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);


        }
        void cuOuter_internal_u16tu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
              cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
              cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_u16ti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_int32 *_out = (cytnx_int32*)out->Mem;
              cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
              cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_u16tu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint32 *_out = (cytnx_uint32*)out->Mem;
            cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }
        void cuOuter_internal_u16ti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_int16 *_out = (cytnx_int16*)out->Mem;
              cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
              cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        void cuOuter_internal_u16tu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint16 *_out = (cytnx_uint16*)out->Mem;
            cytnx_uint16 *_Lin = (cytnx_uint16*)Lin->Mem;
            cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);
        }
        __global__ void cuOuter_kernel(cytnx_uint16 *out, const cytnx_uint16  *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_bool *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)] * cytnx_uint16(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_u16tb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cytnx_uint16  *_out = (cytnx_uint16 *)out->Mem;
            cytnx_uint16  *_Lin = (cytnx_uint16 *)Lin->Mem;
            cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }

//------------------------------

        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cytnx_bool *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuDoubleComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(make_cuDoubleComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_btcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
            cuDoubleComplex *_Rin = (cuDoubleComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);


        }

        __global__ void cuOuter_kernel(cuFloatComplex *out, const cytnx_bool *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuFloatComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(make_cuFloatComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_btcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
             cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
              cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
              cuFloatComplex *_Rin = (cuFloatComplex*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        __global__ void cuOuter_kernel(cytnx_double *out, const cytnx_bool *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_double *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cytnx_double(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)])*ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL];
            }
            __syncthreads();
        }
        void cuOuter_internal_btd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_double *_out = (cytnx_double*)out->Mem;
              cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
              cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        __global__ void cuOuter_kernel(cytnx_float *out, const cytnx_bool *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_float *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cytnx_float(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)])*ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL];
            }
            __syncthreads();
        }
        void cuOuter_internal_btf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_float *_out = (cytnx_float*)out->Mem;
              cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
              cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        __global__ void cuOuter_kernel(cytnx_int64 *out, const cytnx_bool *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_int64 *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cytnx_int64(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)])*ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL];
            }
            __syncthreads();
        }
        void cuOuter_internal_bti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_int64 *_out = (cytnx_int64*)out->Mem;
              cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
              cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        __global__ void cuOuter_kernel(cytnx_uint64 *out, const cytnx_bool *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_uint64 *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cytnx_uint64(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)])*ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL];
            }
            __syncthreads();
        }
        void cuOuter_internal_btu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
              cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
              cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        __global__ void cuOuter_kernel(cytnx_int32 *out, const cytnx_bool *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_int32 *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cytnx_int32(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)])*ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL];
            }
            __syncthreads();
        }
        void cuOuter_internal_bti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_int32*_out = (cytnx_int32*)out->Mem;
              cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
              cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        __global__ void cuOuter_kernel(cytnx_uint32 *out, const cytnx_bool *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_uint32 *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cytnx_uint32(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)])*ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL];
            }
            __syncthreads();
        }
        void cuOuter_internal_btu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_uint32*_out = (cytnx_uint32*)out->Mem;
              cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
              cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        __global__ void cuOuter_kernel(cytnx_int16 *out, const cytnx_bool *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_int16 *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cytnx_int16(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)])*ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL];
            }
            __syncthreads();
        }
        void cuOuter_internal_bti16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_int16*_out = (cytnx_int16*)out->Mem;
              cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
              cytnx_int16 *_Rin = (cytnx_int16*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }
        __global__ void cuOuter_kernel(cytnx_uint16 *out, const cytnx_bool *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_uint16 *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cytnx_uint16(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)])*ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL];
            }
            __syncthreads();
        }
        void cuOuter_internal_btu16(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_uint16*_out = (cytnx_uint16*)out->Mem;
              cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
              cytnx_uint16 *_Rin = (cytnx_uint16*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }

        void cuOuter_internal_btb(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2){
              cytnx_bool*_out = (cytnx_bool*)out->Mem;
              cytnx_bool *_Lin = (cytnx_bool*)Lin->Mem;
              cytnx_bool *_Rin = (cytnx_bool*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin, const cytnx_uint64 &i1, const cytnx_uint64 &j1, const cytnx_uint64 &i2, const cytnx_uint64 &j2);

        }


    }//namespace linalg_internal
}//namespace cytnx


