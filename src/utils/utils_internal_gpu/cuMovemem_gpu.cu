#include "cuMovemem_gpu.hpp"
#include "cuAlloc_gpu.hpp"
#include "Storage.hpp"
#ifdef UNI_OMP
#include <omp.h>
#endif

using namespace std;

namespace cytnx{
    namespace utils_internal{
    #ifdef UNI_GPU
        template<class T>
        __global__ void cuMovemem_kernel(T* ddes, T*dsrc, cytnx_uint64* accu_old, cytnx_uint64* permuted_accu_new, cytnx_uint32 rank, cytnx_uint64 Nelem){
                extern __shared__ cytnx_uint64 SHaccu[];

                cytnx_uint64 ids;
                ///copy to share mem:
                if(rank<=blockDim.x){
                    if(threadIdx.x<rank){
                        SHaccu[threadIdx.x] = accu_old[threadIdx.x];
                        SHaccu[threadIdx.x+rank] = permuted_accu_new[threadIdx.x];
                    }
                }else{
                    cytnx_uint32 Np=rank/blockDim.x;
                    if(rank%blockDim.x) Np+=1;
                    for(cytnx_uint32 i=0;i<Np;i++){
                        ids = i*blockDim.x + threadIdx.x;
                        if(ids < rank){
                            SHaccu[ids] = accu_old[ids];
                            SHaccu[ids+rank] = permuted_accu_new[ids];
                        }
                    }
                }
                __syncthreads();

                cytnx_uint64 tid = blockIdx.x*blockDim.x + threadIdx.x;
                ids = 0;
                for(cytnx_uint32 i=0;i<rank;i++){
                    ids += (tid/SHaccu[i])*SHaccu[rank+i];
                    tid = tid%SHaccu[i];
                }
                if(blockIdx.x*blockDim.x+threadIdx.x<Nelem) ddes[ids] = dsrc[blockIdx.x*blockDim.x+threadIdx.x];

        }
        
        boost::intrusive_ptr<Storage_base> cuMovemem_gpu_cd(boost::intrusive_ptr<Storage_base> &in, const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64>&mapper, const std::vector<cytnx_uint64> &invmapper, const bool is_inplace){
            #ifdef UNI_DEBUG
            cytnx_error_msg(in->dtype != Type.ComplexDouble,"[DEBUG][internal error] in.dtype_str is [%s] but call cuMovemem_gpu with type ComplexDouble",in->dtype_str().c_str());
            cytnx_error_msg(in->device == Device.cpu,"%s", "[DEBUG][internal error] in.device is on cpu but all cuda function.");
            #endif

            

            std::vector<cytnx_uint64> newshape(old_shape.size());
            for(cytnx_uint64 i=0;i<old_shape.size();i++)
                newshape[i] = old_shape[mapper[i]];

            std::vector<cytnx_uint64> shifter_old(old_shape.size());
            std::vector<cytnx_uint64> shifter_new(old_shape.size());

            cytnx_uint64 accu_old=1,accu_new=1;
            for(cytnx_int64 i=old_shape.size()-1;i>=0;i--){
                shifter_old[i] = accu_old;
                shifter_new[i] = accu_new;
                accu_old*=old_shape[i];
                accu_new*=newshape[i];
            }
            std::vector<cytnx_uint64> old_inds(old_shape.size());

            std::vector<cytnx_uint64> permuted_shifter_new(old_shape.size());
            for(unsigned int i=0;i<old_shape.size();i++)
                permuted_shifter_new[i] = shifter_new[invmapper[i]];

            ///allocate a GPU for psn-vec/so-vec/tmp des-vec
            cytnx_uint64 *dshifter_old, *dperm_shifter_new;
            cuDoubleComplex *dtmp;
            cytnx_uint64 Nelem = accu_old;        

            cudaSetDevice(in->device); // ensure the following allocation on the same device as src.
            checkCudaErrors(cudaMalloc((void**)&dshifter_old, sizeof(cytnx_uint64)*shifter_old.size()));
            checkCudaErrors(cudaMalloc((void**)&dperm_shifter_new, sizeof(cytnx_uint64)*permuted_shifter_new.size()));
            dtmp = (cuDoubleComplex*)cuMalloc_gpu(sizeof(cuDoubleComplex)*Nelem); 

            /// copy psn-vec/so-vec to device
            checkCudaErrors(cudaMemcpy(dperm_shifter_new, &permuted_shifter_new[0], sizeof(cytnx_uint64)*permuted_shifter_new.size(),cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(dshifter_old, &shifter_old[0], sizeof(cytnx_uint64)*shifter_old.size(),cudaMemcpyHostToDevice));


            ///calculate how many blocks, and shared mem size, thpb fixed at 256 (need fine tune)
            cytnx_uint64 NBlocks = Nelem/256;
            if(Nelem%256){
                NBlocks+=1;
            }
            cuMovemem_kernel<<< NBlocks,256,shifter_old.size()*2*sizeof(cytnx_uint64) >>>(dtmp,(cuDoubleComplex*)in->Mem,dshifter_old,dperm_shifter_new,old_shape.size(),Nelem);


            ///house keeping:
            checkCudaErrors(cudaFree(dshifter_old));
            checkCudaErrors(cudaFree(dperm_shifter_new));

            boost::intrusive_ptr<Storage_base> out(new ComplexDoubleStorage());
            if(is_inplace){

                ///cpy back:
                checkCudaErrors(cudaMemcpy(in->Mem,dtmp, sizeof(cytnx_complex128)*Nelem,cudaMemcpyDeviceToDevice));
                cudaFree(dtmp);
                return out;

            }else{

                out->_Init_byptr(dtmp,Nelem);
                return out;
            }

        }

        boost::intrusive_ptr<Storage_base> cuMovemem_gpu_cf(boost::intrusive_ptr<Storage_base> &in, const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64>&mapper, const std::vector<cytnx_uint64> &invmapper, const bool is_inplace){
            #ifdef UNI_DEBUG
            cytnx_error_msg(in->dtype != Type.ComplexFloat,"[DEBUG][internal error] in.dtype_str is [%s] but call cuMovemem_gpu with type ComplexFloat",in->dtype_str().c_str());
            cytnx_error_msg(in->device == Device.cpu,"%s", "[DEBUG][internal error] in.device is on cpu but all cuda function.");
            #endif

            

            std::vector<cytnx_uint64> newshape(old_shape.size());
            for(cytnx_uint64 i=0;i<old_shape.size();i++)
                newshape[i] = old_shape[mapper[i]];

            std::vector<cytnx_uint64> shifter_old(old_shape.size());
            std::vector<cytnx_uint64> shifter_new(old_shape.size());

            cytnx_uint64 accu_old=1,accu_new=1;
            for(cytnx_int64 i=old_shape.size()-1;i>=0;i--){
                shifter_old[i] = accu_old;
                shifter_new[i] = accu_new;
                accu_old*=old_shape[i];
                accu_new*=newshape[i];
            }
            std::vector<cytnx_uint64> old_inds(old_shape.size());

            std::vector<cytnx_uint64> permuted_shifter_new(old_shape.size());
            for(unsigned int i=0;i<old_shape.size();i++)
                permuted_shifter_new[i] = shifter_new[invmapper[i]];

            ///allocate a GPU for psn-vec/so-vec/tmp des-vec
            cytnx_uint64 *dshifter_old, *dperm_shifter_new;
            cuFloatComplex *dtmp;
            cytnx_uint64 Nelem = accu_old;        

            cudaSetDevice(in->device); // ensure the following allocation on the same device as src.
            checkCudaErrors(cudaMalloc((void**)&dshifter_old, sizeof(cytnx_uint64)*shifter_old.size()));
            checkCudaErrors(cudaMalloc((void**)&dperm_shifter_new, sizeof(cytnx_uint64)*permuted_shifter_new.size()));
            dtmp = (cuFloatComplex*)cuMalloc_gpu(sizeof(cuFloatComplex)*Nelem); 

            /// copy psn-vec/so-vec to device
            checkCudaErrors(cudaMemcpy(dperm_shifter_new, &permuted_shifter_new[0], sizeof(cytnx_uint64)*permuted_shifter_new.size(),cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(dshifter_old, &shifter_old[0], sizeof(cytnx_uint64)*shifter_old.size(),cudaMemcpyHostToDevice));


            ///calculate how many blocks, and shared mem size, thpb fixed at 256 (need fine tune)
            cytnx_uint64 NBlocks = Nelem/256;
            if(Nelem%256){
                NBlocks+=1;
            }
            cuMovemem_kernel<<< NBlocks,256,shifter_old.size()*2*sizeof(cytnx_uint64) >>>(dtmp,(cuFloatComplex*)in->Mem,dshifter_old,dperm_shifter_new,old_shape.size(),Nelem);


            ///house keeping:
            checkCudaErrors(cudaFree(dshifter_old));
            checkCudaErrors(cudaFree(dperm_shifter_new));

            boost::intrusive_ptr<Storage_base> out(new ComplexFloatStorage());
            if(is_inplace){

                ///cpy back:
                checkCudaErrors(cudaMemcpy(in->Mem,dtmp, sizeof(cytnx_complex64)*Nelem,cudaMemcpyDeviceToDevice));
                cudaFree(dtmp);
                return out;

            }else{

                out->_Init_byptr(dtmp,Nelem);
                return out;
            }

        }
        
        boost::intrusive_ptr<Storage_base> cuMovemem_gpu_d(boost::intrusive_ptr<Storage_base> &in, const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64>&mapper, const std::vector<cytnx_uint64> &invmapper, const bool is_inplace){
            #ifdef UNI_DEBUG
            cytnx_error_msg(in->dtype != Type.Double,"[DEBUG][internal error] in.dtype_str is [%s] but call cuMovemem_gpu with type Double",in->dtype_str().c_str());
            cytnx_error_msg(in->device == Device.cpu,"%s", "[DEBUG][internal error] in.device is on cpu but all cuda function.");
            #endif

            

            std::vector<cytnx_uint64> newshape(old_shape.size());
            for(cytnx_uint64 i=0;i<old_shape.size();i++)
                newshape[i] = old_shape[mapper[i]];

            std::vector<cytnx_uint64> shifter_old(old_shape.size());
            std::vector<cytnx_uint64> shifter_new(old_shape.size());

            cytnx_uint64 accu_old=1,accu_new=1;
            for(cytnx_int64 i=old_shape.size()-1;i>=0;i--){
                shifter_old[i] = accu_old;
                shifter_new[i] = accu_new;
                accu_old*=old_shape[i];
                accu_new*=newshape[i];
            }
            std::vector<cytnx_uint64> old_inds(old_shape.size());

            std::vector<cytnx_uint64> permuted_shifter_new(old_shape.size());
            for(unsigned int i=0;i<old_shape.size();i++)
                permuted_shifter_new[i] = shifter_new[invmapper[i]];

            ///allocate a GPU for psn-vec/so-vec/tmp des-vec
            cytnx_uint64 *dshifter_old, *dperm_shifter_new;
            double *dtmp;
            cytnx_uint64 Nelem = accu_old;        

            cudaSetDevice(in->device); // ensure the following allocation on the same device as src.
            checkCudaErrors(cudaMalloc((void**)&dshifter_old, sizeof(cytnx_uint64)*shifter_old.size()));
            checkCudaErrors(cudaMalloc((void**)&dperm_shifter_new, sizeof(cytnx_uint64)*permuted_shifter_new.size()));
            dtmp = (double*)cuMalloc_gpu(sizeof(double)*Nelem); 

            /// copy psn-vec/so-vec to device
            checkCudaErrors(cudaMemcpy(dperm_shifter_new, &permuted_shifter_new[0], sizeof(cytnx_uint64)*permuted_shifter_new.size(),cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(dshifter_old, &shifter_old[0], sizeof(cytnx_uint64)*shifter_old.size(),cudaMemcpyHostToDevice));


            ///calculate how many blocks, and shared mem size, thpb fixed at 256 (need fine tune)
            cytnx_uint64 NBlocks = Nelem/256;
            if(Nelem%256){
                NBlocks+=1;
            }
            cuMovemem_kernel<<< NBlocks,256,shifter_old.size()*2*sizeof(cytnx_uint64) >>>(dtmp,(double*)in->Mem,dshifter_old,dperm_shifter_new,old_shape.size(),Nelem);


            ///house keeping:
            checkCudaErrors(cudaFree(dshifter_old));
            checkCudaErrors(cudaFree(dperm_shifter_new));

            boost::intrusive_ptr<Storage_base> out(new DoubleStorage());
            if(is_inplace){

                ///cpy back:
                checkCudaErrors(cudaMemcpy(in->Mem,dtmp, sizeof(double)*Nelem,cudaMemcpyDeviceToDevice));
                cudaFree(dtmp);
                return out;

            }else{

                out->_Init_byptr(dtmp,Nelem);
                return out;
            }
        }
        
        boost::intrusive_ptr<Storage_base> cuMovemem_gpu_f(boost::intrusive_ptr<Storage_base> &in, const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64>&mapper, const std::vector<cytnx_uint64> &invmapper, const bool is_inplace){
            #ifdef UNI_DEBUG
            cytnx_error_msg(in->dtype != Type.Float,"[DEBUG][internal error] in.dtype_str is [%s] but call cuMovemem_gpu with type Float",in->dtype_str().c_str());
            cytnx_error_msg(in->device == Device.cpu,"%s", "[DEBUG][internal error] in.device is on cpu but all cuda function.");
            #endif

            

            std::vector<cytnx_uint64> newshape(old_shape.size());
            for(cytnx_uint64 i=0;i<old_shape.size();i++)
                newshape[i] = old_shape[mapper[i]];

            std::vector<cytnx_uint64> shifter_old(old_shape.size());
            std::vector<cytnx_uint64> shifter_new(old_shape.size());

            cytnx_uint64 accu_old=1,accu_new=1;
            for(cytnx_int64 i=old_shape.size()-1;i>=0;i--){
                shifter_old[i] = accu_old;
                shifter_new[i] = accu_new;
                accu_old*=old_shape[i];
                accu_new*=newshape[i];
            }
            std::vector<cytnx_uint64> old_inds(old_shape.size());

            std::vector<cytnx_uint64> permuted_shifter_new(old_shape.size());
            for(unsigned int i=0;i<old_shape.size();i++)
                permuted_shifter_new[i] = shifter_new[invmapper[i]];

            ///allocate a GPU for psn-vec/so-vec/tmp des-vec
            cytnx_uint64 *dshifter_old, *dperm_shifter_new;
            float *dtmp;
            cytnx_uint64 Nelem = accu_old;        

            cudaSetDevice(in->device); // ensure the following allocation on the same device as src.
            checkCudaErrors(cudaMalloc((void**)&dshifter_old, sizeof(cytnx_uint64)*shifter_old.size()));
            checkCudaErrors(cudaMalloc((void**)&dperm_shifter_new, sizeof(cytnx_uint64)*permuted_shifter_new.size()));
            dtmp = (float*)cuMalloc_gpu(sizeof(float)*Nelem); 

            /// copy psn-vec/so-vec to device
            checkCudaErrors(cudaMemcpy(dperm_shifter_new, &permuted_shifter_new[0], sizeof(cytnx_uint64)*permuted_shifter_new.size(),cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(dshifter_old, &shifter_old[0], sizeof(cytnx_uint64)*shifter_old.size(),cudaMemcpyHostToDevice));


            ///calculate how many blocks, and shared mem size, thpb fixed at 256 (need fine tune)
            cytnx_uint64 NBlocks = Nelem/256;
            if(Nelem%256){
                NBlocks+=1;
            }
            cuMovemem_kernel<<< NBlocks,256,shifter_old.size()*2*sizeof(cytnx_uint64) >>>(dtmp,(float*)in->Mem,dshifter_old,dperm_shifter_new,old_shape.size(),Nelem);


            ///house keeping:
            checkCudaErrors(cudaFree(dshifter_old));
            checkCudaErrors(cudaFree(dperm_shifter_new));

            boost::intrusive_ptr<Storage_base> out(new FloatStorage());
            if(is_inplace){

                ///cpy back:
                checkCudaErrors(cudaMemcpy(in->Mem,dtmp, sizeof(float)*Nelem,cudaMemcpyDeviceToDevice));
                cudaFree(dtmp);
                return out;

            }else{

                out->_Init_byptr(dtmp,Nelem);
                return out;
            }

        }

        boost::intrusive_ptr<Storage_base> cuMovemem_gpu_i64(boost::intrusive_ptr<Storage_base> &in, const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64>&mapper, const std::vector<cytnx_uint64> &invmapper, const bool is_inplace){
            #ifdef UNI_DEBUG
            cytnx_error_msg(in->dtype != Type.Int64,"[DEBUG][internal error] in.dtype_str is [%s] but call cuMovemem_gpu with type Int64",in->dtype_str().c_str());
            cytnx_error_msg(in->device == Device.cpu,"%s", "[DEBUG][internal error] in.device is on cpu but all cuda function.");
            #endif

            

            std::vector<cytnx_uint64> newshape(old_shape.size());
            for(cytnx_uint64 i=0;i<old_shape.size();i++)
                newshape[i] = old_shape[mapper[i]];

            std::vector<cytnx_uint64> shifter_old(old_shape.size());
            std::vector<cytnx_uint64> shifter_new(old_shape.size());

            cytnx_uint64 accu_old=1,accu_new=1;
            for(cytnx_int64 i=old_shape.size()-1;i>=0;i--){
                shifter_old[i] = accu_old;
                shifter_new[i] = accu_new;
                accu_old*=old_shape[i];
                accu_new*=newshape[i];
            }
            std::vector<cytnx_uint64> old_inds(old_shape.size());

            std::vector<cytnx_uint64> permuted_shifter_new(old_shape.size());
            for(unsigned int i=0;i<old_shape.size();i++)
                permuted_shifter_new[i] = shifter_new[invmapper[i]];

            ///allocate a GPU for psn-vec/so-vec/tmp des-vec
            cytnx_uint64 *dshifter_old, *dperm_shifter_new;
            cytnx_int64 *dtmp;
            cytnx_uint64 Nelem = accu_old;        

            cudaSetDevice(in->device); // ensure the following allocation on the same device as src.
            checkCudaErrors(cudaMalloc((void**)&dshifter_old, sizeof(cytnx_uint64)*shifter_old.size()));
            checkCudaErrors(cudaMalloc((void**)&dperm_shifter_new, sizeof(cytnx_uint64)*permuted_shifter_new.size()));
            dtmp = (cytnx_int64*)cuMalloc_gpu(sizeof(cytnx_int64)*Nelem); 

            /// copy psn-vec/so-vec to device
            checkCudaErrors(cudaMemcpy(dperm_shifter_new, &permuted_shifter_new[0], sizeof(cytnx_uint64)*permuted_shifter_new.size(),cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(dshifter_old, &shifter_old[0], sizeof(cytnx_uint64)*shifter_old.size(),cudaMemcpyHostToDevice));


            ///calculate how many blocks, and shared mem size, thpb fixed at 256 (need fine tune)
            cytnx_uint64 NBlocks = Nelem/256;
            if(Nelem%256){
                NBlocks+=1;
            }
            cuMovemem_kernel<<< NBlocks,256,shifter_old.size()*2*sizeof(cytnx_uint64) >>>(dtmp,(cytnx_int64*)in->Mem,dshifter_old,dperm_shifter_new,old_shape.size(),Nelem);


            ///house keeping:
            checkCudaErrors(cudaFree(dshifter_old));
            checkCudaErrors(cudaFree(dperm_shifter_new));

            boost::intrusive_ptr<Storage_base> out(new Int64Storage());
            if(is_inplace){

                ///cpy back:
                checkCudaErrors(cudaMemcpy(in->Mem,dtmp, sizeof(cytnx_int64)*Nelem,cudaMemcpyDeviceToDevice));
                cudaFree(dtmp);
                return out;

            }else{

                out->_Init_byptr(dtmp,Nelem);
                return out;
            }


       }

        boost::intrusive_ptr<Storage_base> cuMovemem_gpu_u64(boost::intrusive_ptr<Storage_base> &in, const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64>&mapper, const std::vector<cytnx_uint64> &invmapper, const bool is_inplace){
            #ifdef UNI_DEBUG
            cytnx_error_msg(in->dtype != Type.Uint64,"[DEBUG][internal error] in.dtype_str is [%s] but call cuMovemem_gpu with type Uint64",in->dtype_str().c_str());
            cytnx_error_msg(in->device == Device.cpu,"%s", "[DEBUG][internal error] in.device is on cpu but all cuda function.");
            #endif

            

            std::vector<cytnx_uint64> newshape(old_shape.size());
            for(cytnx_uint64 i=0;i<old_shape.size();i++)
                newshape[i] = old_shape[mapper[i]];

            std::vector<cytnx_uint64> shifter_old(old_shape.size());
            std::vector<cytnx_uint64> shifter_new(old_shape.size());

            cytnx_uint64 accu_old=1,accu_new=1;
            for(cytnx_int64 i=old_shape.size()-1;i>=0;i--){
                shifter_old[i] = accu_old;
                shifter_new[i] = accu_new;
                accu_old*=old_shape[i];
                accu_new*=newshape[i];
            }
            std::vector<cytnx_uint64> old_inds(old_shape.size());

            std::vector<cytnx_uint64> permuted_shifter_new(old_shape.size());
            for(unsigned int i=0;i<old_shape.size();i++)
                permuted_shifter_new[i] = shifter_new[invmapper[i]];

            ///allocate a GPU for psn-vec/so-vec/tmp des-vec
            cytnx_uint64 *dshifter_old, *dperm_shifter_new;
            cytnx_uint64 *dtmp;
            cytnx_uint64 Nelem = accu_old;        

            cudaSetDevice(in->device); // ensure the following allocation on the same device as src.
            checkCudaErrors(cudaMalloc((void**)&dshifter_old, sizeof(cytnx_uint64)*shifter_old.size()));
            checkCudaErrors(cudaMalloc((void**)&dperm_shifter_new, sizeof(cytnx_uint64)*permuted_shifter_new.size()));
            dtmp = (cytnx_uint64*)cuMalloc_gpu(sizeof(cytnx_uint64)*Nelem); 

            /// copy psn-vec/so-vec to device
            checkCudaErrors(cudaMemcpy(dperm_shifter_new, &permuted_shifter_new[0], sizeof(cytnx_uint64)*permuted_shifter_new.size(),cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(dshifter_old, &shifter_old[0], sizeof(cytnx_uint64)*shifter_old.size(),cudaMemcpyHostToDevice));


            ///calculate how many blocks, and shared mem size, thpb fixed at 256 (need fine tune)
            cytnx_uint64 NBlocks = Nelem/256;
            if(Nelem%256){
                NBlocks+=1;
            }
            cuMovemem_kernel<<< NBlocks,256,shifter_old.size()*2*sizeof(cytnx_uint64) >>>(dtmp,(cytnx_uint64*)in->Mem,dshifter_old,dperm_shifter_new,old_shape.size(),Nelem);


            ///house keeping:
            checkCudaErrors(cudaFree(dshifter_old));
            checkCudaErrors(cudaFree(dperm_shifter_new));

            boost::intrusive_ptr<Storage_base> out(new Uint64Storage());
            if(is_inplace){

                ///cpy back:
                checkCudaErrors(cudaMemcpy(in->Mem,dtmp, sizeof(cytnx_uint64)*Nelem,cudaMemcpyDeviceToDevice));
                cudaFree(dtmp);
                return out;

            }else{

                out->_Init_byptr(dtmp,Nelem);
                return out;
            }
        }

        boost::intrusive_ptr<Storage_base> cuMovemem_gpu_i32(boost::intrusive_ptr<Storage_base> &in, const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64>&mapper, const std::vector<cytnx_uint64> &invmapper, const bool is_inplace){
            #ifdef UNI_DEBUG
            cytnx_error_msg(in->dtype != Type.Int32,"[DEBUG][internal error] in.dtype_str is [%s] but call cuMovemem_gpu with type Int32",in->dtype_str().c_str());
            cytnx_error_msg(in->device == Device.cpu,"%s", "[DEBUG][internal error] in.device is on cpu but all cuda function.");
            #endif

            

            std::vector<cytnx_uint64> newshape(old_shape.size());
            for(cytnx_uint64 i=0;i<old_shape.size();i++)
                newshape[i] = old_shape[mapper[i]];

            std::vector<cytnx_uint64> shifter_old(old_shape.size());
            std::vector<cytnx_uint64> shifter_new(old_shape.size());

            cytnx_uint64 accu_old=1,accu_new=1;
            for(cytnx_int64 i=old_shape.size()-1;i>=0;i--){
                shifter_old[i] = accu_old;
                shifter_new[i] = accu_new;
                accu_old*=old_shape[i];
                accu_new*=newshape[i];
            }
            std::vector<cytnx_uint64> old_inds(old_shape.size());

            std::vector<cytnx_uint64> permuted_shifter_new(old_shape.size());
            for(unsigned int i=0;i<old_shape.size();i++)
                permuted_shifter_new[i] = shifter_new[invmapper[i]];

            ///allocate a GPU for psn-vec/so-vec/tmp des-vec
            cytnx_uint64 *dshifter_old, *dperm_shifter_new;
            cytnx_int32 *dtmp;
            cytnx_uint64 Nelem = accu_old;        

            cudaSetDevice(in->device); // ensure the following allocation on the same device as src.
            checkCudaErrors(cudaMalloc((void**)&dshifter_old, sizeof(cytnx_uint64)*shifter_old.size()));
            checkCudaErrors(cudaMalloc((void**)&dperm_shifter_new, sizeof(cytnx_uint64)*permuted_shifter_new.size()));
            dtmp = (cytnx_int32*)cuMalloc_gpu(sizeof(cytnx_int32)*Nelem); 

            /// copy psn-vec/so-vec to device
            checkCudaErrors(cudaMemcpy(dperm_shifter_new, &permuted_shifter_new[0], sizeof(cytnx_uint64)*permuted_shifter_new.size(),cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(dshifter_old, &shifter_old[0], sizeof(cytnx_uint64)*shifter_old.size(),cudaMemcpyHostToDevice));


            ///calculate how many blocks, and shared mem size, thpb fixed at 256 (need fine tune)
            cytnx_uint64 NBlocks = Nelem/256;
            if(Nelem%256){
                NBlocks+=1;
            }
            cuMovemem_kernel<<< NBlocks,256,shifter_old.size()*2*sizeof(cytnx_uint64) >>>(dtmp,(cytnx_int32*)in->Mem,dshifter_old,dperm_shifter_new,old_shape.size(),Nelem);


            ///house keeping:
            checkCudaErrors(cudaFree(dshifter_old));
            checkCudaErrors(cudaFree(dperm_shifter_new));

            boost::intrusive_ptr<Storage_base> out(new Int32Storage());
            if(is_inplace){

                ///cpy back:
                checkCudaErrors(cudaMemcpy(in->Mem,dtmp, sizeof(cytnx_int32)*Nelem,cudaMemcpyDeviceToDevice));
                cudaFree(dtmp);
                return out;

            }else{

                out->_Init_byptr(dtmp,Nelem);
                return out;
            }
        }

        boost::intrusive_ptr<Storage_base> cuMovemem_gpu_u32(boost::intrusive_ptr<Storage_base> &in, const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64>&mapper, const std::vector<cytnx_uint64> &invmapper, const bool is_inplace){
            #ifdef UNI_DEBUG
            cytnx_error_msg(in->dtype != Type.Uint32,"[DEBUG][internal error] in.dtype_str is [%s] but call cuMovemem_gpu with type Uint32",in->dtype_str().c_str());
            cytnx_error_msg(in->device == Device.cpu,"%s", "[DEBUG][internal error] in.device is on cpu but all cuda function.");
            #endif

            

            std::vector<cytnx_uint64> newshape(old_shape.size());
            for(cytnx_uint64 i=0;i<old_shape.size();i++)
                newshape[i] = old_shape[mapper[i]];

            std::vector<cytnx_uint64> shifter_old(old_shape.size());
            std::vector<cytnx_uint64> shifter_new(old_shape.size());

            cytnx_uint64 accu_old=1,accu_new=1;
            for(cytnx_int64 i=old_shape.size()-1;i>=0;i--){
                shifter_old[i] = accu_old;
                shifter_new[i] = accu_new;
                accu_old*=old_shape[i];
                accu_new*=newshape[i];
            }
            std::vector<cytnx_uint64> old_inds(old_shape.size());

            std::vector<cytnx_uint64> permuted_shifter_new(old_shape.size());
            for(unsigned int i=0;i<old_shape.size();i++)
                permuted_shifter_new[i] = shifter_new[invmapper[i]];

            ///allocate a GPU for psn-vec/so-vec/tmp des-vec
            cytnx_uint64 *dshifter_old, *dperm_shifter_new;
            cytnx_uint32 *dtmp;
            cytnx_uint64 Nelem = accu_old;        

            cudaSetDevice(in->device); // ensure the following allocation on the same device as src.
            checkCudaErrors(cudaMalloc((void**)&dshifter_old, sizeof(cytnx_uint64)*shifter_old.size()));
            checkCudaErrors(cudaMalloc((void**)&dperm_shifter_new, sizeof(cytnx_uint64)*permuted_shifter_new.size()));
            dtmp = (cytnx_uint32*)cuMalloc_gpu(sizeof(cytnx_uint32)*Nelem); 

            /// copy psn-vec/so-vec to device
            checkCudaErrors(cudaMemcpy(dperm_shifter_new, &permuted_shifter_new[0], sizeof(cytnx_uint64)*permuted_shifter_new.size(),cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(dshifter_old, &shifter_old[0], sizeof(cytnx_uint64)*shifter_old.size(),cudaMemcpyHostToDevice));


            ///calculate how many blocks, and shared mem size, thpb fixed at 256 (need fine tune)
            cytnx_uint64 NBlocks = Nelem/256;
            if(Nelem%256){
                NBlocks+=1;
            }
            cuMovemem_kernel<<< NBlocks,256,shifter_old.size()*2*sizeof(cytnx_uint64) >>>(dtmp,(cytnx_uint32*)in->Mem,dshifter_old,dperm_shifter_new,old_shape.size(),Nelem);


            ///house keeping:
            checkCudaErrors(cudaFree(dshifter_old));
            checkCudaErrors(cudaFree(dperm_shifter_new));

            boost::intrusive_ptr<Storage_base> out(new Uint32Storage());
            if(is_inplace){

                ///cpy back:
                checkCudaErrors(cudaMemcpy(in->Mem,dtmp, sizeof(cytnx_uint32)*Nelem,cudaMemcpyDeviceToDevice));
                cudaFree(dtmp);
                return out;

            }else{

                out->_Init_byptr(dtmp,Nelem);
                return out;
            }
        }
        boost::intrusive_ptr<Storage_base> cuMovemem_gpu_u16(boost::intrusive_ptr<Storage_base> &in, const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64>&mapper, const std::vector<cytnx_uint64> &invmapper, const bool is_inplace){
            #ifdef UNI_DEBUG
            cytnx_error_msg(in->dtype != Type.Uint16,"[DEBUG][internal error] in.dtype_str is [%s] but call cuMovemem_gpu with type Uint16",in->dtype_str().c_str());
            cytnx_error_msg(in->device == Device.cpu,"%s", "[DEBUG][internal error] in.device is on cpu but all cuda function.");
            #endif

            

            std::vector<cytnx_uint64> newshape(old_shape.size());
            for(cytnx_uint64 i=0;i<old_shape.size();i++)
                newshape[i] = old_shape[mapper[i]];

            std::vector<cytnx_uint64> shifter_old(old_shape.size());
            std::vector<cytnx_uint64> shifter_new(old_shape.size());

            cytnx_uint64 accu_old=1,accu_new=1;
            for(cytnx_int64 i=old_shape.size()-1;i>=0;i--){
                shifter_old[i] = accu_old;
                shifter_new[i] = accu_new;
                accu_old*=old_shape[i];
                accu_new*=newshape[i];
            }
            std::vector<cytnx_uint64> old_inds(old_shape.size());

            std::vector<cytnx_uint64> permuted_shifter_new(old_shape.size());
            for(unsigned int i=0;i<old_shape.size();i++)
                permuted_shifter_new[i] = shifter_new[invmapper[i]];

            ///allocate a GPU for psn-vec/so-vec/tmp des-vec
            cytnx_uint64 *dshifter_old, *dperm_shifter_new;
            cytnx_uint16 *dtmp;
            cytnx_uint64 Nelem = accu_old;        

            cudaSetDevice(in->device); // ensure the following allocation on the same device as src.
            checkCudaErrors(cudaMalloc((void**)&dshifter_old, sizeof(cytnx_uint64)*shifter_old.size()));
            checkCudaErrors(cudaMalloc((void**)&dperm_shifter_new, sizeof(cytnx_uint64)*permuted_shifter_new.size()));
            dtmp = (cytnx_uint16*)cuMalloc_gpu(sizeof(cytnx_uint16)*Nelem); 

            /// copy psn-vec/so-vec to device
            checkCudaErrors(cudaMemcpy(dperm_shifter_new, &permuted_shifter_new[0], sizeof(cytnx_uint64)*permuted_shifter_new.size(),cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(dshifter_old, &shifter_old[0], sizeof(cytnx_uint64)*shifter_old.size(),cudaMemcpyHostToDevice));


            ///calculate how many blocks, and shared mem size, thpb fixed at 256 (need fine tune)
            cytnx_uint64 NBlocks = Nelem/256;
            if(Nelem%256){
                NBlocks+=1;
            }
            cuMovemem_kernel<<< NBlocks,256,shifter_old.size()*2*sizeof(cytnx_uint64) >>>(dtmp,(cytnx_uint16*)in->Mem,dshifter_old,dperm_shifter_new,old_shape.size(),Nelem);


            ///house keeping:
            checkCudaErrors(cudaFree(dshifter_old));
            checkCudaErrors(cudaFree(dperm_shifter_new));

            boost::intrusive_ptr<Storage_base> out(new Uint16Storage());
            if(is_inplace){

                ///cpy back:
                checkCudaErrors(cudaMemcpy(in->Mem,dtmp, sizeof(cytnx_uint16)*Nelem,cudaMemcpyDeviceToDevice));
                cudaFree(dtmp);
                return out;

            }else{

                out->_Init_byptr(dtmp,Nelem);
                return out;
            }
        }
        boost::intrusive_ptr<Storage_base> cuMovemem_gpu_i16(boost::intrusive_ptr<Storage_base> &in, const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64>&mapper, const std::vector<cytnx_uint64> &invmapper, const bool is_inplace){
            #ifdef UNI_DEBUG
            cytnx_error_msg(in->dtype != Type.Int16,"[DEBUG][internal error] in.dtype_str is [%s] but call cuMovemem_gpu with type Int16",in->dtype_str().c_str());
            cytnx_error_msg(in->device == Device.cpu,"%s", "[DEBUG][internal error] in.device is on cpu but all cuda function.");
            #endif

            

            std::vector<cytnx_uint64> newshape(old_shape.size());
            for(cytnx_uint64 i=0;i<old_shape.size();i++)
                newshape[i] = old_shape[mapper[i]];

            std::vector<cytnx_uint64> shifter_old(old_shape.size());
            std::vector<cytnx_uint64> shifter_new(old_shape.size());

            cytnx_uint64 accu_old=1,accu_new=1;
            for(cytnx_int64 i=old_shape.size()-1;i>=0;i--){
                shifter_old[i] = accu_old;
                shifter_new[i] = accu_new;
                accu_old*=old_shape[i];
                accu_new*=newshape[i];
            }
            std::vector<cytnx_uint64> old_inds(old_shape.size());

            std::vector<cytnx_uint64> permuted_shifter_new(old_shape.size());
            for(unsigned int i=0;i<old_shape.size();i++)
                permuted_shifter_new[i] = shifter_new[invmapper[i]];

            ///allocate a GPU for psn-vec/so-vec/tmp des-vec
            cytnx_uint64 *dshifter_old, *dperm_shifter_new;
            cytnx_int16 *dtmp;
            cytnx_uint64 Nelem = accu_old;        

            cudaSetDevice(in->device); // ensure the following allocation on the same device as src.
            checkCudaErrors(cudaMalloc((void**)&dshifter_old, sizeof(cytnx_uint64)*shifter_old.size()));
            checkCudaErrors(cudaMalloc((void**)&dperm_shifter_new, sizeof(cytnx_uint64)*permuted_shifter_new.size()));
            dtmp = (cytnx_int16*)cuMalloc_gpu(sizeof(cytnx_int16)*Nelem); 

            /// copy psn-vec/so-vec to device
            checkCudaErrors(cudaMemcpy(dperm_shifter_new, &permuted_shifter_new[0], sizeof(cytnx_uint64)*permuted_shifter_new.size(),cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(dshifter_old, &shifter_old[0], sizeof(cytnx_uint64)*shifter_old.size(),cudaMemcpyHostToDevice));


            ///calculate how many blocks, and shared mem size, thpb fixed at 256 (need fine tune)
            cytnx_uint64 NBlocks = Nelem/256;
            if(Nelem%256){
                NBlocks+=1;
            }
            cuMovemem_kernel<<< NBlocks,256,shifter_old.size()*2*sizeof(cytnx_uint64) >>>(dtmp,(cytnx_int16*)in->Mem,dshifter_old,dperm_shifter_new,old_shape.size(),Nelem);


            ///house keeping:
            checkCudaErrors(cudaFree(dshifter_old));
            checkCudaErrors(cudaFree(dperm_shifter_new));

            boost::intrusive_ptr<Storage_base> out(new Int16Storage());
            if(is_inplace){

                ///cpy back:
                checkCudaErrors(cudaMemcpy(in->Mem,dtmp, sizeof(cytnx_int16)*Nelem,cudaMemcpyDeviceToDevice));
                cudaFree(dtmp);
                return out;

            }else{

                out->_Init_byptr(dtmp,Nelem);
                return out;
            }
        }
        boost::intrusive_ptr<Storage_base> cuMovemem_gpu_b(boost::intrusive_ptr<Storage_base> &in, const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64>&mapper, const std::vector<cytnx_uint64> &invmapper, const bool is_inplace){
            #ifdef UNI_DEBUG
            cytnx_error_msg(in->dtype != Type.Bool,"[DEBUG][internal error] in.dtype_str is [%s] but call cuMovemem_gpu with type Bool",in->dtype_str().c_str());
            cytnx_error_msg(in->device == Device.cpu,"%s", "[DEBUG][internal error] in.device is on cpu but all cuda function.");
            #endif

            

            std::vector<cytnx_uint64> newshape(old_shape.size());
            for(cytnx_uint64 i=0;i<old_shape.size();i++)
                newshape[i] = old_shape[mapper[i]];

            std::vector<cytnx_uint64> shifter_old(old_shape.size());
            std::vector<cytnx_uint64> shifter_new(old_shape.size());

            cytnx_uint64 accu_old=1,accu_new=1;
            for(cytnx_int64 i=old_shape.size()-1;i>=0;i--){
                shifter_old[i] = accu_old;
                shifter_new[i] = accu_new;
                accu_old*=old_shape[i];
                accu_new*=newshape[i];
            }
            std::vector<cytnx_uint64> old_inds(old_shape.size());

            std::vector<cytnx_uint64> permuted_shifter_new(old_shape.size());
            for(unsigned int i=0;i<old_shape.size();i++)
                permuted_shifter_new[i] = shifter_new[invmapper[i]];

            ///allocate a GPU for psn-vec/so-vec/tmp des-vec
            cytnx_uint64 *dshifter_old, *dperm_shifter_new;
            cytnx_bool *dtmp;
            cytnx_uint64 Nelem = accu_old;        

            cudaSetDevice(in->device); // ensure the following allocation on the same device as src.
            checkCudaErrors(cudaMalloc((void**)&dshifter_old, sizeof(cytnx_uint64)*shifter_old.size()));
            checkCudaErrors(cudaMalloc((void**)&dperm_shifter_new, sizeof(cytnx_uint64)*permuted_shifter_new.size()));
            dtmp = (cytnx_bool*)cuMalloc_gpu(sizeof(cytnx_bool)*Nelem); 

            /// copy psn-vec/so-vec to device
            checkCudaErrors(cudaMemcpy(dperm_shifter_new, &permuted_shifter_new[0], sizeof(cytnx_uint64)*permuted_shifter_new.size(),cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(dshifter_old, &shifter_old[0], sizeof(cytnx_uint64)*shifter_old.size(),cudaMemcpyHostToDevice));


            ///calculate how many blocks, and shared mem size, thpb fixed at 256 (need fine tune)
            cytnx_uint64 NBlocks = Nelem/256;
            if(Nelem%256){
                NBlocks+=1;
            }
            cuMovemem_kernel<<< NBlocks,256,shifter_old.size()*2*sizeof(cytnx_uint64) >>>(dtmp,(cytnx_bool*)in->Mem,dshifter_old,dperm_shifter_new,old_shape.size(),Nelem);


            ///house keeping:
            checkCudaErrors(cudaFree(dshifter_old));
            checkCudaErrors(cudaFree(dperm_shifter_new));

            boost::intrusive_ptr<Storage_base> out(new BoolStorage());
            if(is_inplace){

                ///cpy back:
                checkCudaErrors(cudaMemcpy(in->Mem,dtmp, sizeof(cytnx_bool)*Nelem,cudaMemcpyDeviceToDevice));
                cudaFree(dtmp);
                return out;

            }else{

                out->_Init_byptr(dtmp,Nelem);
                return out;
            }
        }


    #endif // UNI_GPU
    }//namespace utils_internal
}//namespace cytnx
