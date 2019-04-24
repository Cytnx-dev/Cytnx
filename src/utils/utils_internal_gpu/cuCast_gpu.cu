#include "utils/utils_internal_gpu/cuCast_gpu.hpp"
#include "Storage.hpp"
#ifdef UNI_OMP
#include <omp.h>
#endif

using namespace std;
namespace tor10{
    namespace utils_internal{

        cuCast_gpu_interface::cuCast_gpu_interface(){
            UElemCast_gpu = vector<vector<ElemCast_io_gpu> >(N_Type,vector<ElemCast_io_gpu>(N_Type,NULL));

            UElemCast_gpu[tor10type.ComplexDouble][tor10type.ComplexDouble] = cuCast_gpu_cdtcd;
            UElemCast_gpu[tor10type.ComplexDouble][tor10type.ComplexFloat ] = cuCast_gpu_cdtcf;
            //UElemCast_gpu[tor10type.ComplexDouble][tor10type.Double       ] = cuCast_gpu_invalid;
            //UElemCast_gpu[tor10type.ComplexDouble][tor10type.Float        ] = cuCast_gpu_invalid;
            //UElemCast_gpu[tor10type.ComplexDouble][tor10type.Int64        ] = cuCast_gpu_invalid;
            //UElemCast_gpu[tor10type.ComplexDouble][tor10type.Uint64       ] = cuCast_gpu_invalid;
            //UElemCast_gpu[tor10type.ComplexDouble][tor10type.Int32        ] = cuCast_gpu_invalid;
            //UElemCast_gpu[tor10type.ComplexDouble][tor10type.Uint32       ] = cuCast_gpu_invalid;

            UElemCast_gpu[tor10type.ComplexFloat][tor10type.ComplexDouble] = cuCast_gpu_cftcd;
            UElemCast_gpu[tor10type.ComplexFloat][tor10type.ComplexFloat ] = cuCast_gpu_cftcf;
            //UElemCast_gpu[tor10type.ComplexFloat][tor10type.Double       ] = cuCast_gpu_invalid;
            //UElemCast_gpu[tor10type.ComplexFloat][tor10type.Float        ] = cuCast_gpu_invalid;
            //UElemCast_gpu[tor10type.ComplexFloat][tor10type.Int64        ] = cuCast_gpu_invalid;
            //UElemCast_gpu[tor10type.ComplexFloat][tor10type.Uint64       ] = cuCast_gpu_invalid;
            //UElemCast_gpu[tor10type.ComplexFloat][tor10type.Int32        ] = cuCast_gpu_invalid;
            //UElemCast_gpu[tor10type.ComplexFloat][tor10type.Uint32       ] = cuCast_gpu_invalid;

            UElemCast_gpu[tor10type.Double][tor10type.ComplexDouble] = cuCast_gpu_dtcd;
            UElemCast_gpu[tor10type.Double][tor10type.ComplexFloat ] = cuCast_gpu_dtcf;
            UElemCast_gpu[tor10type.Double][tor10type.Double       ] = cuCast_gpu_dtd;
            UElemCast_gpu[tor10type.Double][tor10type.Float        ] = cuCast_gpu_dtf;
            UElemCast_gpu[tor10type.Double][tor10type.Int64        ] = cuCast_gpu_dti64;
            UElemCast_gpu[tor10type.Double][tor10type.Uint64       ] = cuCast_gpu_dtu64;
            UElemCast_gpu[tor10type.Double][tor10type.Int32        ] = cuCast_gpu_dti32;
            UElemCast_gpu[tor10type.Double][tor10type.Uint32       ] = cuCast_gpu_dtu32;

            UElemCast_gpu[tor10type.Float][tor10type.ComplexDouble] = cuCast_gpu_ftcd;
            UElemCast_gpu[tor10type.Float][tor10type.ComplexFloat ] = cuCast_gpu_ftcf;
            UElemCast_gpu[tor10type.Float][tor10type.Double       ] = cuCast_gpu_ftd;
            UElemCast_gpu[tor10type.Float][tor10type.Float        ] = cuCast_gpu_ftf;
            UElemCast_gpu[tor10type.Float][tor10type.Int64        ] = cuCast_gpu_fti64;
            UElemCast_gpu[tor10type.Float][tor10type.Uint64       ] = cuCast_gpu_ftu64;
            UElemCast_gpu[tor10type.Float][tor10type.Int32        ] = cuCast_gpu_fti32;
            UElemCast_gpu[tor10type.Float][tor10type.Uint32       ] = cuCast_gpu_ftu32;

            UElemCast_gpu[tor10type.Int64][tor10type.ComplexDouble] = cuCast_gpu_i64tcd;
            UElemCast_gpu[tor10type.Int64][tor10type.ComplexFloat ] = cuCast_gpu_i64tcf;
            UElemCast_gpu[tor10type.Int64][tor10type.Double       ] = cuCast_gpu_i64td;
            UElemCast_gpu[tor10type.Int64][tor10type.Float        ] = cuCast_gpu_i64tf;
            UElemCast_gpu[tor10type.Int64][tor10type.Int64        ] = cuCast_gpu_i64ti64;
            UElemCast_gpu[tor10type.Int64][tor10type.Uint64       ] = cuCast_gpu_i64tu64;
            UElemCast_gpu[tor10type.Int64][tor10type.Int32        ] = cuCast_gpu_i64ti32;
            UElemCast_gpu[tor10type.Int64][tor10type.Uint32       ] = cuCast_gpu_i64tu32;

            UElemCast_gpu[tor10type.Uint64][tor10type.ComplexDouble] = cuCast_gpu_u64tcd;
            UElemCast_gpu[tor10type.Uint64][tor10type.ComplexFloat ] = cuCast_gpu_u64tcf;
            UElemCast_gpu[tor10type.Uint64][tor10type.Double       ] = cuCast_gpu_u64td;
        }
        utils_internal::cuCast_gpu_interface cuCast_gpu; // interface object. 

        //=======================================================================

        __global__ void cuCastElem_kernel_cd2cf(const cuDoubleComplex *src, cuFloatComplex *des, tor10_uint64 Nelem){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){
                des[blockIdx.x*blockDim.x + threadIdx.x] = cuComplexDoubleToFloat(src[blockIdx.x*blockDim.x + threadIdx.x]);
            }
        }

        __global__ void cuCastElem_kernel_cf2cd(const cuFloatComplex *src, cuDoubleComplex *des, tor10_uint64 Nelem){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){
                des[blockIdx.x*blockDim.x + threadIdx.x] = cuComplexFloatToDouble(src[blockIdx.x*blockDim.x + threadIdx.x]);
            }
        }
        
        template<class T>
        __global__ void cuCastElem_kernel_r2cf(const T *src, cuFloatComplex *des, tor10_uint64 Nelem){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){
                des[blockIdx.x*blockDim.x + threadIdx.x].x = src[blockIdx.x*blockDim.x + threadIdx.x];
            }
        }
        template<class T2>
        __global__ void cuCastElem_kernel_r2cd(const T2 *src, cuDoubleComplex *des, tor10_uint64 Nelem){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){
                des[blockIdx.x*blockDim.x + threadIdx.x].x = src[blockIdx.x*blockDim.x + threadIdx.x];
            }
        }

        template<class T3,class T4>
        __global__ void cuCastElem_kernel_r2r(const T3 *src, T4 *des, tor10_uint64 Nelem){
            if(blockIdx.x*blockDim.x + threadIdx.x < Nelem){
                des[blockIdx.x*blockDim.x + threadIdx.x] = src[blockIdx.x*blockDim.x + threadIdx.x];
            }
        }

        //========================================================================
        void cuCast_gpu_cdtcd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
                out->Init(len_in,alloc_device);
            }
            checkCudaErrors(cudaMemcpy(out->Mem,in->Mem,sizeof(tor10_complex128)*len_in,cudaMemcpyDeviceToDevice)); 
        }

        void cuCast_gpu_cdtcf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());
                out->Init(len_in,alloc_device);
            }

            cuDoubleComplex* _in = static_cast<cuDoubleComplex*>(in->Mem);
            cuFloatComplex*  _out= static_cast<cuFloatComplex*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_cd2cf<<<NBlocks,512>>>(_in,_out,len_in);

        }

        void cuCast_gpu_cftcd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
                out->Init(len_in,alloc_device);
            }
            cuFloatComplex* _in = static_cast<cuFloatComplex*>(in->Mem);
            cuDoubleComplex*  _out= static_cast<cuDoubleComplex*>(out->Mem);
            
            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_cf2cd<<<NBlocks,512>>>(_in,_out,len_in);
        }

        void cuCast_gpu_cftcf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());
                out->Init(len_in,alloc_device);
            }
            checkCudaErrors(cudaMemcpy(out->Mem,in->Mem,sizeof(tor10_complex64)*len_in,cudaMemcpyDeviceToDevice)); 
        }


        void cuCast_gpu_dtcd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){

            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
                out->Init(len_in,alloc_device);
            }
            tor10_double* _in = static_cast<tor10_double*>(in->Mem);
            cuDoubleComplex*  _out= static_cast<cuDoubleComplex*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2cd<<<NBlocks,512>>>(_in,_out,len_in);

        }

        void cuCast_gpu_dtcf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());    
                out->Init(len_in,alloc_device);
            }
            tor10_double* _in = static_cast<tor10_double*>(in->Mem);
            cuFloatComplex*  _out= static_cast<cuFloatComplex*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2cf<<<NBlocks,512>>>(_in,_out,len_in);

        }

        void cuCast_gpu_dtd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){       
                out = boost::intrusive_ptr<Storage_base>(new DoubleStorage());
                out->Init(len_in,alloc_device);
            }
            checkCudaErrors(cudaMemcpy(out->Mem,in->Mem,sizeof(tor10_double)*len_in,cudaMemcpyDeviceToDevice)); 

        }
        void cuCast_gpu_dtf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new FloatStorage());
                out -> Init(len_in);
            }
            tor10_double* _in = static_cast<tor10_double*>(in->Mem);
            tor10_float*  _out= static_cast<tor10_float*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);
        }
        void cuCast_gpu_dti64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new Int64Storage());
                out-> Init(len_in);
            }
            tor10_double* _in = static_cast<tor10_double*>(in->Mem);
            tor10_int64*  _out= static_cast<tor10_int64*>(out->Mem);
            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);
        }
        void cuCast_gpu_dtu64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new Uint64Storage());
                out->Init(len_in,alloc_device);
            }
            tor10_double* _in = static_cast<tor10_double*>(in->Mem);
            tor10_uint64*  _out= static_cast<tor10_uint64*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);

        }
        void cuCast_gpu_dti32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new Int32Storage());
                out->Init(len_in,alloc_device);

            }
            tor10_double* _in = static_cast<tor10_double*>(in->Mem);
            tor10_int32*  _out= static_cast<tor10_int32*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);

        }
        void cuCast_gpu_dtu32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new Uint32Storage());
                out->Init(len_in,alloc_device);
            }
            tor10_double* _in = static_cast<tor10_double*>(in->Mem);
            tor10_uint32*  _out= static_cast<tor10_uint32*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);

        }

        void cuCast_gpu_ftcd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
                out->Init(len_in,alloc_device);
            }
            tor10_float* _in = static_cast<tor10_float*>(in->Mem);
            cuDoubleComplex*  _out= static_cast<cuDoubleComplex*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2cd<<<NBlocks,512>>>(_in,_out,len_in);
        }

        void cuCast_gpu_ftcf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());
                out->Init(len_in,alloc_device);
            }
            tor10_float* _in = static_cast<tor10_float*>(in->Mem);
            cuFloatComplex*  _out= static_cast<cuFloatComplex*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2cf<<<NBlocks,512>>>(_in,_out,len_in);
        }
        void cuCast_gpu_ftd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new DoubleStorage());
                out->Init(len_in,alloc_device);
            }
            tor10_float* _in = static_cast<tor10_float*>(in->Mem);
            tor10_double*  _out= static_cast<tor10_double*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);
        }
        void cuCast_gpu_ftf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new FloatStorage());
                out->Init(len_in,alloc_device);
            }
            checkCudaErrors(cudaMemcpy(out->Mem,in->Mem,sizeof(tor10_float)*len_in,cudaMemcpyDeviceToDevice)); 
        }
        void cuCast_gpu_fti64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new Int64Storage());
                out->Init(len_in,alloc_device);
            }
            tor10_float* _in = static_cast<tor10_float*>(in->Mem);
            tor10_int64*  _out= static_cast<tor10_int64*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);
        }
        void cuCast_gpu_ftu64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new Uint64Storage());
                out->Init(len_in,alloc_device);
            }
            tor10_float* _in = static_cast<tor10_float*>(in->Mem);
            tor10_uint64*  _out= static_cast<tor10_uint64*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);
        }
        void cuCast_gpu_fti32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new Int32Storage());
                out->Init(len_in,alloc_device);
            }
            tor10_float* _in = static_cast<tor10_float*>(in->Mem);
            tor10_int32*  _out= static_cast<tor10_int32*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);
        }
        void cuCast_gpu_ftu32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new Uint32Storage());
                out->Init(len_in,alloc_device);
            }
            tor10_float* _in = static_cast<tor10_float*>(in->Mem);
            tor10_uint32*  _out= static_cast<tor10_uint32*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);
        }

        void cuCast_gpu_i64tcd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
                out->Init(len_in,alloc_device);
            }
            tor10_int64* _in = static_cast<tor10_int64*>(in->Mem);
            cuDoubleComplex*  _out= static_cast<cuDoubleComplex*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2cd<<<NBlocks,512>>>(_in,_out,len_in);
        }
        void cuCast_gpu_i64tcf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());
                out->Init(len_in,alloc_device);
            }
            tor10_int64* _in = static_cast<tor10_int64*>(in->Mem);
            cuFloatComplex*  _out= static_cast<cuFloatComplex*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2cf<<<NBlocks,512>>>(_in,_out,len_in);
        }

        void cuCast_gpu_i64td(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new DoubleStorage());
                out->Init(len_in,alloc_device);
            }
            tor10_int64* _in = static_cast<tor10_int64*>(in->Mem);
            tor10_double*  _out= static_cast<tor10_double*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);

        }
        void cuCast_gpu_i64tf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new FloatStorage());
                out->Init(len_in,alloc_device);
            }
            tor10_int64* _in = static_cast<tor10_int64*>(in->Mem);
            tor10_float*  _out= static_cast<tor10_float*>(out->Mem);


            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);

        }
        void cuCast_gpu_i64ti64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new Int64Storage());
                out->Init(len_in,alloc_device);
            }
            checkCudaErrors(cudaMemcpy(out->Mem,in->Mem,sizeof(tor10_int64)*len_in,cudaMemcpyDeviceToDevice)); 

        }
        void cuCast_gpu_i64tu64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new Uint64Storage());
                out->Init(len_in,alloc_device);
            }
            tor10_int64* _in = static_cast<tor10_int64*>(in->Mem);
            tor10_uint64*  _out= static_cast<tor10_uint64*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);
        }
        void cuCast_gpu_i64ti32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new Int32Storage());
                out->Init(len_in,alloc_device);
            }
            tor10_int64* _in = static_cast<tor10_int64*>(in->Mem);
            tor10_int32*  _out= static_cast<tor10_int32*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);
        }
        void cuCast_gpu_i64tu32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new Uint32Storage());
                out->Init(len_in,alloc_device);
            }
            tor10_int64* _in = static_cast<tor10_int64*>(in->Mem);
            tor10_uint32*  _out= static_cast<tor10_uint32*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);
        }

        void cuCast_gpu_u64tcd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
                out->Init(len_in,alloc_device);
            }
            tor10_uint64* _in = static_cast<tor10_uint64*>(in->Mem);
            cuDoubleComplex*  _out= static_cast<cuDoubleComplex*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2cd<<<NBlocks,512>>>(_in,_out,len_in);

        }
        void cuCast_gpu_u64tcf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());
                out->Init(len_in,alloc_device);
            }
            tor10_uint64* _in = static_cast<tor10_uint64*>(in->Mem);
            cuFloatComplex*  _out= static_cast<cuFloatComplex*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2cf<<<NBlocks,512>>>(_in,_out,len_in);

        }
        void cuCast_gpu_u64td(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new DoubleStorage());
                out->Init(len_in,alloc_device);
            }
            tor10_uint64* _in = static_cast<tor10_uint64*>(in->Mem);
            tor10_double*  _out= static_cast<tor10_double*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);
        }
        void cuCast_gpu_u64tf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new FloatStorage());
                out->Init(len_in,alloc_device);
            }
            tor10_uint64* _in = static_cast<tor10_uint64*>(in->Mem);
            tor10_float*  _out= static_cast<tor10_float*>(out->Mem);


            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);
        }
        void cuCast_gpu_u64ti64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new Int64Storage());
                out->Init(len_in,alloc_device);
            }
            tor10_uint64* _in = static_cast<tor10_uint64*>(in->Mem);
            tor10_int64*  _out= static_cast<tor10_int64*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);
        }
        void cuCast_gpu_u64tu64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new Uint64Storage());
                out->Init(len_in,alloc_device);
            }
            checkCudaErrors(cudaMemcpy(out->Mem,in->Mem,sizeof(tor10_uint64)*len_in,cudaMemcpyDeviceToDevice)); 
           
        }
        void cuCast_gpu_u64ti32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new Int32Storage());
                out->Init(len_in,alloc_device);
            }
            tor10_uint64* _in = static_cast<tor10_uint64*>(in->Mem);
            tor10_int32*  _out= static_cast<tor10_int32*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);
        }
        void cuCast_gpu_u64tu32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new Uint32Storage());
                out->Init(len_in,alloc_device);
            }
            tor10_uint64* _in = static_cast<tor10_uint64*>(in->Mem);
            tor10_uint32*  _out= static_cast<tor10_uint32*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);
        }

        void cuCast_gpu_i32tcd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
                out->Init(len_in,alloc_device);
            }
            tor10_int32* _in = static_cast<tor10_int32*>(in->Mem);
            cuDoubleComplex*  _out= static_cast<cuDoubleComplex*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2cd<<<NBlocks,512>>>(_in,_out,len_in);
        }
        void cuCast_gpu_i32tcf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());
                out->Init(len_in,alloc_device);
            }
            tor10_int32* _in = static_cast<tor10_int32*>(in->Mem);
            cuFloatComplex*  _out= static_cast<cuFloatComplex*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2cf<<<NBlocks,512>>>(_in,_out,len_in);
        }
        void cuCast_gpu_i32td(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new DoubleStorage());
                out->Init(len_in,alloc_device);
            }
            tor10_int32* _in = static_cast<tor10_int32*>(in->Mem);
            tor10_double*  _out= static_cast<tor10_double*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);
        }
        void cuCast_gpu_i32tf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new FloatStorage());
                out->Init(len_in,alloc_device);
            }
            tor10_int32* _in = static_cast<tor10_int32*>(in->Mem);
            tor10_float*  _out= static_cast<tor10_float*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);
        }
        void cuCast_gpu_i32ti64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new Int64Storage());
                out->Init(len_in,alloc_device);
            }
            tor10_int32* _in = static_cast<tor10_int32*>(in->Mem);
            tor10_int64*  _out= static_cast<tor10_int64*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);
        }
        void cuCast_gpu_i32tu64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new Uint64Storage());
                out->Init(len_in,alloc_device);
            }
            tor10_int32* _in = static_cast<tor10_int32*>(in->Mem);
            tor10_uint64*  _out= static_cast<tor10_uint64*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);
        }
        void cuCast_gpu_i32ti32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new Int32Storage());
                out->Init(len_in,alloc_device);
            }
            checkCudaErrors(cudaMemcpy(out->Mem,in->Mem,sizeof(tor10_int32)*len_in,cudaMemcpyDeviceToDevice)); 
        }
        void cuCast_gpu_i32tu32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new Uint32Storage());
                out->Init(len_in,alloc_device);
            }
            tor10_int32* _in = static_cast<tor10_int32*>(in->Mem);
            tor10_uint32*  _out= static_cast<tor10_uint32*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);
        }

        void cuCast_gpu_u32tcd(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new ComplexDoubleStorage());
                out->Init(len_in,alloc_device);
            }
            tor10_uint32* _in = static_cast<tor10_uint32*>(in->Mem);
            cuDoubleComplex*  _out= static_cast<cuDoubleComplex*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2cd<<<NBlocks,512>>>(_in,_out,len_in);

        }
        void cuCast_gpu_u32tcf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new ComplexFloatStorage());
                out->Init(len_in,alloc_device);
            }
            tor10_uint32* _in = static_cast<tor10_uint32*>(in->Mem);
            cuFloatComplex*  _out= static_cast<cuFloatComplex*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2cf<<<NBlocks,512>>>(_in,_out,len_in);
        }
        void cuCast_gpu_u32td(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new DoubleStorage());
                out->Init(len_in,alloc_device);
            }
            tor10_uint32* _in = static_cast<tor10_uint32*>(in->Mem);
            tor10_double*  _out= static_cast<tor10_double*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);
        }
        void cuCast_gpu_u32tf(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new FloatStorage());
                out->Init(len_in,alloc_device);
            }
            tor10_uint32* _in = static_cast<tor10_uint32*>(in->Mem);
            tor10_float*  _out= static_cast<tor10_float*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);
        }
        void cuCast_gpu_u32ti64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new Int64Storage());
                out->Init(len_in,alloc_device);
            }
            tor10_uint32* _in = static_cast<tor10_uint32*>(in->Mem);
            tor10_int64*  _out= static_cast<tor10_int64*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);
        }
        void cuCast_gpu_u32tu64(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new Uint64Storage());
                out->Init(len_in,alloc_device);
            }
            tor10_uint32* _in = static_cast<tor10_uint32*>(in->Mem);
            tor10_uint64*  _out= static_cast<tor10_uint64*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);
        }
        void cuCast_gpu_u32ti32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new Int32Storage());
                out->Init(len_in,alloc_device);
            }
            tor10_uint32* _in = static_cast<tor10_uint32*>(in->Mem);
            tor10_int32*  _out= static_cast<tor10_int32*>(out->Mem);

            tor10_uint64 NBlocks = len_in/512;
            if(len_in%512) NBlocks+=1;
            cuCastElem_kernel_r2r<<<NBlocks,512>>>(_in,_out,len_in);
        }
        void cuCast_gpu_u32tu32(const boost::intrusive_ptr<Storage_base>& in, boost::intrusive_ptr<Storage_base>& out, const unsigned long long &len_in, const int &alloc_device){
            if(alloc_device>=0){
                out = boost::intrusive_ptr<Storage_base>(new Uint32Storage());
                out->Init(len_in,alloc_device);
            }
            checkCudaErrors(cudaMemcpy(out->Mem,in->Mem,sizeof(tor10_uint32)*len_in,cudaMemcpyDeviceToDevice)); 

        }
    }//namespace utils_internal
}//namespace tor10
