#include "linalg/linalg_internal_gpu/cuOuter_internal.hpp"
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
        void cuOuter_internal_cdtcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cuDoubleComplex *_Lin = (cuDoubleComplex*)Lin->Mem;
            cuDoubleComplex *_Rin = (cuDoubleComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);
            
        }



        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cuDoubleComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuFloatComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],cuComplexFloatToDouble(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]));
            }
            __syncthreads();
        }
        void cuOuter_internal_cdtcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cuDoubleComplex *_Lin = (cuDoubleComplex*)Lin->Mem;
            cuFloatComplex *_Rin = (cuFloatComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }

        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cuDoubleComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_double *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuDoubleComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cdtd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cuDoubleComplex *_Lin = (cuDoubleComplex*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;


            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }

        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cuDoubleComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_float *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuDoubleComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cdtf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cuDoubleComplex *_Lin = (cuDoubleComplex*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;


            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);



        }


        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cuDoubleComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_uint64 *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuDoubleComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cdtu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cuDoubleComplex *_Lin = (cuDoubleComplex*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }


        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cuDoubleComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_uint32 *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuDoubleComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cdtu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cuDoubleComplex *_Lin = (cuDoubleComplex*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }



        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cuDoubleComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_int64 *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuDoubleComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cdti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cuDoubleComplex *_Lin = (cuDoubleComplex*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);


        }


        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cuDoubleComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_int32 *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuDoubleComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cdti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cuDoubleComplex *_Lin = (cuDoubleComplex*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }


        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cuFloatComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuDoubleComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(cuComplexFloatToDouble(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)]),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_cftcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cuFloatComplex *_Lin = (cuFloatComplex*)Lin->Mem;
            cuDoubleComplex *_Rin = (cuDoubleComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }

        __global__ void cuOuter_kernel(cuFloatComplex *out, const cuFloatComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuFloatComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_cftcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
            cuFloatComplex *_Lin = (cuFloatComplex*)Lin->Mem;
            cuFloatComplex *_Rin = (cuFloatComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);
        }


        __global__ void cuOuter_kernel(cuFloatComplex *out, const cuFloatComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_double *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuFloatComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cftd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
            cuFloatComplex *_Lin = (cuFloatComplex*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }

        __global__ void cuOuter_kernel(cuFloatComplex *out, const cuFloatComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_float *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuFloatComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cftf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
            cuFloatComplex *_Lin = (cuFloatComplex*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);


        }


        __global__ void cuOuter_kernel(cuFloatComplex *out, const cuFloatComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_uint64 *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuFloatComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cftu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
            cuFloatComplex *_Lin = (cuFloatComplex*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;
            
            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }


        __global__ void cuOuter_kernel(cuFloatComplex *out, const cuFloatComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_uint32 *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuFloatComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cftu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
            cuFloatComplex *_Lin = (cuFloatComplex*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }


        __global__ void cuOuter_kernel(cuFloatComplex *out, const cuFloatComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_int64 *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuFloatComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cfti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
            cuFloatComplex *_Lin = (cuFloatComplex*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }


        __global__ void cuOuter_kernel(cuFloatComplex *out, const cuFloatComplex *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cytnx_int32 *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],make_cuFloatComplex(ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL],0));
            }
            __syncthreads();
        }
        void cuOuter_internal_cfti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
            cuFloatComplex *_Lin = (cuFloatComplex*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }
        

        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cytnx_double *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuDoubleComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(make_cuDoubleComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_dtcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cuDoubleComplex *_Rin = (cuDoubleComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);
        }


        __global__ void cuOuter_kernel(cuFloatComplex *out, const cytnx_double *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuFloatComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(make_cuFloatComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_dtcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cuFloatComplex *_Rin = (cuFloatComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }


        void cuOuter_internal_dtd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);


        }
        void cuOuter_internal_dtf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);
        }
        void cuOuter_internal_dtu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }
        void cuOuter_internal_dtu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);
        }
        void cuOuter_internal_dti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }
        void cuOuter_internal_dti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){

            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_double *_Lin = (cytnx_double*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);
        }



        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cytnx_float *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuDoubleComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(make_cuDoubleComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_ftcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cuDoubleComplex *_Rin = (cuDoubleComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }


        __global__ void cuOuter_kernel(cuFloatComplex *out, const cytnx_float *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuFloatComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(make_cuFloatComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_ftcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cuFloatComplex *_Rin = (cuFloatComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }

        void cuOuter_internal_ftd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }
        void cuOuter_internal_ftf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }
        void cuOuter_internal_ftu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }
        void cuOuter_internal_ftu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);
        }
        void cuOuter_internal_fti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }
        void cuOuter_internal_fti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_float *_Lin = (cytnx_float*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);
        }



        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cytnx_int64 *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuDoubleComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(make_cuDoubleComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_i64tcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cuDoubleComplex *_Rin = (cuDoubleComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }

        __global__ void cuOuter_kernel(cuFloatComplex *out, const cytnx_int64 *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuFloatComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(make_cuFloatComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_i64tcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cuFloatComplex *_Rin = (cuFloatComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }


        void cuOuter_internal_i64td(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);
        }
        void cuOuter_internal_i64tf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);
        }
        void cuOuter_internal_i64ti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }
        void cuOuter_internal_i64tu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);
        }
        void cuOuter_internal_i64ti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);
        }
        void cuOuter_internal_i64tu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int64 *_Lin = (cytnx_int64*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);
        }

        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cytnx_uint64 *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuDoubleComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(make_cuDoubleComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_u64tcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
             cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
             cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
             cuDoubleComplex  *_Rin = (cuDoubleComplex*)Rin->Mem;
  
            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }

        __global__ void cuOuter_kernel(cuFloatComplex *out, const cytnx_uint64 *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuFloatComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(make_cuFloatComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_u64tcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
               cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
               cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
               cuFloatComplex  *_Rin = (cuFloatComplex*)Rin->Mem;
  
               cytnx_uint64 Nelem = Lin->len*Rin->len;
               cytnx_uint32 NBlocks = Nelem/512;
               if(Nelem%512) NBlocks += 1;
                
               cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }
        void cuOuter_internal_u64td(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
               cytnx_double *_out = (cytnx_double*)out->Mem;
               cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
               cytnx_double  *_Rin = (cytnx_double*)Rin->Mem;
  
               cytnx_uint64 Nelem = Lin->len*Rin->len;
               cytnx_uint32 NBlocks = Nelem/512;
               if(Nelem%512) NBlocks += 1;
                
               cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);
        }
        void cuOuter_internal_u64tf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
               cytnx_float *_out = (cytnx_float*)out->Mem;
               cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
               cytnx_float  *_Rin = (cytnx_float*)Rin->Mem;
  
               cytnx_uint64 Nelem = Lin->len*Rin->len;
               cytnx_uint32 NBlocks = Nelem/512;
               if(Nelem%512) NBlocks += 1;
                
               cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);
        }
        void cuOuter_internal_u64ti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
               cytnx_int64 *_out = (cytnx_int64*)out->Mem;
               cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
               cytnx_int64  *_Rin = (cytnx_int64*)Rin->Mem;
  
               cytnx_uint64 Nelem = Lin->len*Rin->len;
               cytnx_uint32 NBlocks = Nelem/512;
               if(Nelem%512) NBlocks += 1;
                
               cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);
        }
        void cuOuter_internal_u64tu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);
        }
        void cuOuter_internal_u64ti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);
        }
        void cuOuter_internal_u64tu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_uint64 *_Lin = (cytnx_uint64*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }




        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cytnx_int32 *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuDoubleComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(make_cuDoubleComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_i32tcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
              cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
              cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
              cuDoubleComplex *_Rin = (cuDoubleComplex*)Rin->Mem;
  
              cytnx_uint64 Nelem = Lin->len*Rin->len;
              cytnx_uint32 NBlocks = Nelem/512;
              if(Nelem%512) NBlocks += 1;
                
              cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);


        }

        __global__ void cuOuter_kernel(cuFloatComplex *out, const cytnx_int32 *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuFloatComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(make_cuFloatComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_i32tcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cuFloatComplex *_Rin = (cuFloatComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);


        }
        void cuOuter_internal_i32td(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cytnx_double *_out = (cytnx_double*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }
        void cuOuter_internal_i32tf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cytnx_float *_out = (cytnx_float*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);


        }
        void cuOuter_internal_i32ti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cytnx_int64 *_out = (cytnx_int64*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);


        }
        void cuOuter_internal_i32tu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);


        }
        void cuOuter_internal_i32ti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }
        void cuOuter_internal_i32tu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cytnx_int32 *_out = (cytnx_int32*)out->Mem;
            cytnx_int32 *_Lin = (cytnx_int32*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }

        __global__ void cuOuter_kernel(cuDoubleComplex *out, const cytnx_uint32 *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuDoubleComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmul(make_cuDoubleComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_u32tcd(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cuDoubleComplex *_out = (cuDoubleComplex*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cuDoubleComplex *_Rin = (cuDoubleComplex*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);


        }

        __global__ void cuOuter_kernel(cuFloatComplex *out, const cytnx_uint32 *val, const cytnx_uint64 Nelem, const cytnx_uint64 OffL,const cuFloatComplex *ptr){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){                                                
                  out[blockIdx.x*blockDim.x + threadIdx.x] = cuCmulf(make_cuFloatComplex(val[cytnx_uint64((blockIdx.x*blockDim.x + threadIdx.x)/OffL)],0),ptr[(blockIdx.x*blockDim.x + threadIdx.x)%OffL]);
            }
            __syncthreads();
        }
        void cuOuter_internal_u32tcf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
             cuFloatComplex *_out = (cuFloatComplex*)out->Mem;
              cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
              cuFloatComplex *_Rin = (cuFloatComplex*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }
        void cuOuter_internal_u32td(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
              cytnx_double *_out = (cytnx_double*)out->Mem;
              cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
              cytnx_double *_Rin = (cytnx_double*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }
        void cuOuter_internal_u32tf(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
              cytnx_float *_out = (cytnx_float*)out->Mem;
              cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
              cytnx_float *_Rin = (cytnx_float*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }
        void cuOuter_internal_u32ti64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
              cytnx_int64 *_out = (cytnx_int64*)out->Mem;
              cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
              cytnx_int64 *_Rin = (cytnx_int64*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);


        }
        void cuOuter_internal_u32tu64(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
              cytnx_uint64 *_out = (cytnx_uint64*)out->Mem;
              cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
              cytnx_uint64 *_Rin = (cytnx_uint64*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }
        void cuOuter_internal_u32ti32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
              cytnx_int32 *_out = (cytnx_int32*)out->Mem;
              cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
              cytnx_int32 *_Rin = (cytnx_int32*)Rin->Mem;

              cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);

        }
        void cuOuter_internal_u32tu32(boost::intrusive_ptr<Storage_base> & out, const boost::intrusive_ptr<Storage_base> & Lin, const boost::intrusive_ptr<Storage_base> & Rin){
            cytnx_uint32 *_out = (cytnx_uint32*)out->Mem;
            cytnx_uint32 *_Lin = (cytnx_uint32*)Lin->Mem;
            cytnx_uint32 *_Rin = (cytnx_uint32*)Rin->Mem;

            cytnx_uint64 Nelem = Lin->len*Rin->len;
            cytnx_uint32 NBlocks = Nelem/512;
            if(Nelem%512) NBlocks += 1;
                
            cuOuter_kernel<<<NBlocks,512>>>(_out,_Lin,Nelem,Rin->len,_Rin);
        }





    }//namespace linalg_internal
}//namespace cytnx


