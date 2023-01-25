#include "cuMovemem_gpu.hpp"
#include "cuAlloc_gpu.hpp"
#include "Storage.hpp"
#include <algorithm>
#ifdef UNI_OMP
#include <omp.h>
#endif

#ifdef UNI_GPU
    #ifdef UNI_CUTT
    #include "cutt.h"
    #endif
#endif

using namespace std;

namespace cytnx{
    namespace utils_internal{



    #ifdef UNI_GPU
        template <class BidirectionalIterator>
        void reverse_perm(BidirectionalIterator first, BidirectionalIterator last,int N)
        {
          while ((first!=last)&&(first!=--last)) {
            *first = (N-1) - *first;
            *last = (N-1) - *last;
            std::iter_swap (first,last);
            ++first;
          }
          if(N%2) *first = (N-1) - *first;
        }

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

        // T is the cytnx type, cuT is the cuda type. For all types they should be the same except for cuDoubleComplex and cuFloatComplex.
        template<class T, class cuT>
        boost::intrusive_ptr<Storage_base> cuMovemem_gpu_general(boost::intrusive_ptr<Storage_base> &in, const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64>&mapper, const std::vector<cytnx_uint64> &invmapper, const bool is_inplace){
            T proxy;
            unsigned int dtype_T = Type_class::cy_typeid(proxy);
            #ifdef UNI_DEBUG
            cytnx_error_msg(in->dtype != dtype_T, "[DEBUG][internal error] in.dtype_str is [%s] but call cuMovemem_gpu with type %s",in->dtype_str().c_str(),Type.getname(dtype_T));
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
            cuT *dtmp;
            cytnx_uint64 Nelem = accu_old;

            cudaSetDevice(in->device); // ensure the following allocation on the same device as src.
            checkCudaErrors(cudaMalloc((void**)&dshifter_old, sizeof(cytnx_uint64)*shifter_old.size()));
            checkCudaErrors(cudaMalloc((void**)&dperm_shifter_new, sizeof(cytnx_uint64)*permuted_shifter_new.size()));
            dtmp = (cuT*)cuMalloc_gpu(sizeof(cuT)*in->cap);

            /// copy psn-vec/so-vec to device
            checkCudaErrors(cudaMemcpy(dperm_shifter_new, &permuted_shifter_new[0], sizeof(cytnx_uint64)*permuted_shifter_new.size(),cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(dshifter_old, &shifter_old[0], sizeof(cytnx_uint64)*shifter_old.size(),cudaMemcpyHostToDevice));


            ///calculate how many blocks, and shared mem size, thpb fixed at 256 (need fine tune)
            cytnx_uint64 NBlocks = Nelem/256;
            if(Nelem%256){
                NBlocks+=1;
            }
            cuMovemem_kernel<<< NBlocks,256,shifter_old.size()*2*sizeof(cytnx_uint64) >>>(dtmp,(cuT*)in->Mem,dshifter_old,dperm_shifter_new,old_shape.size(),Nelem);


            ///house keeping:
            checkCudaErrors(cudaFree(dshifter_old));
            checkCudaErrors(cudaFree(dperm_shifter_new));
            

            boost::intrusive_ptr<Storage_base> out = __SII.USIInit[dtype_T]();
            if(is_inplace){

                ///cpy back:
                checkCudaErrors(cudaMemcpy(in->Mem,dtmp, sizeof(T)*Nelem,cudaMemcpyDeviceToDevice));
                cudaFree(dtmp);
                return out;

            }else{

                out->_Init_byptr(dtmp,Nelem,in->device,true,in->cap);
                return out;
            }

        }


#ifdef UNI_CUTT
         template<class T, class cuT>
         boost::intrusive_ptr<Storage_base> cuMovemem_cutt_gpu(boost::intrusive_ptr<Storage_base> &in, const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64>&mapper, const std::vector<cytnx_uint64> &invmapper, const bool is_inplace){
            T proxy;
            unsigned int dtype_T = Type_class::cy_typeid(proxy);
            #ifdef UNI_DEBUG
            cytnx_error_msg(in->dtype != dtype_T, "[DEBUG][internal error] in.dtype_str is [%s] but call cuMovemem_cutt with type %s",in->dtype_str().c_str(),Type.getname(dtype_T));
            cytnx_error_msg(in->device == Device.cpu,"%s", "[DEBUG][internal error] in.device is on cpu but all cuda function.");
            #endif 

            cuT *dtmp;
            dtmp = (cuT*)cuMalloc_gpu(sizeof(cuT)*in->cap);
            cytnx_uint64 Nelem = in->len;

            std::vector<int> perm(mapper.begin(),mapper.end());
            std::vector<int> size(old_shape.begin(),old_shape.end());
            std::reverse(size.begin(),size.end()); //matching API CUTT
            reverse_perm(perm.begin(),perm.end(),perm.size()); //matching API CUTT

            cuttHandle plan;
            cuttPlan(&plan,perm.size(),size.data(),perm.data(),sizeof(cuT),0);
            cuttExecute(plan,in->Mem,dtmp);

            cuttDestroy(plan);
           
            boost::intrusive_ptr<Storage_base> out = __SII.USIInit[dtype_T]();
            if(is_inplace){

                ///cpy back:
                checkCudaErrors(cudaMemcpy(in->Mem,dtmp, sizeof(T)*Nelem,cudaMemcpyDeviceToDevice));
                cudaFree(dtmp);
                return out;

            }else{

                out->_Init_byptr(dtmp,Nelem,in->device,true,in->cap);
                return out;
            }
                 
         }
#endif 


        
        boost::intrusive_ptr<Storage_base> cuMovemem_gpu_cd(boost::intrusive_ptr<Storage_base> &in, const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64>&mapper, const std::vector<cytnx_uint64> &invmapper, const bool is_inplace){
            return cuMovemem_gpu_general<cytnx_complex128, cuDoubleComplex>(in,old_shape,mapper,invmapper,is_inplace);
        }

        boost::intrusive_ptr<Storage_base> cuMovemem_gpu_cf(boost::intrusive_ptr<Storage_base> &in, const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64>&mapper, const std::vector<cytnx_uint64> &invmapper, const bool is_inplace){
            #ifdef UNI_CUTT
                return cuMovemem_cutt_gpu<cytnx_complex64,cuFloatComplex>(in,old_shape,mapper,invmapper,is_inplace);
            #else
                return cuMovemem_gpu_general<cytnx_complex64, cuFloatComplex>(in, old_shape,mapper,invmapper,is_inplace);
            #endif
        }
        
        boost::intrusive_ptr<Storage_base> cuMovemem_gpu_d(boost::intrusive_ptr<Storage_base> &in, const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64>&mapper, const std::vector<cytnx_uint64> &invmapper, const bool is_inplace){
            #ifdef UNI_CUTT
                return cuMovemem_cutt_gpu<cytnx_double,cytnx_double>(in,old_shape,mapper,invmapper,is_inplace);
            #else
                return cuMovemem_gpu_general<cytnx_double, cytnx_double>(in, old_shape,mapper,invmapper,is_inplace);
            #endif
        }
        
        boost::intrusive_ptr<Storage_base> cuMovemem_gpu_f(boost::intrusive_ptr<Storage_base> &in, const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64>&mapper, const std::vector<cytnx_uint64> &invmapper, const bool is_inplace){
            #ifdef UNI_CUTT
                return cuMovemem_cutt_gpu<cytnx_float,cytnx_float>(in,old_shape,mapper,invmapper,is_inplace);
            #else
                return cuMovemem_gpu_general<cytnx_float, cytnx_float>(in, old_shape,mapper,invmapper,is_inplace);
            #endif

        }

        boost::intrusive_ptr<Storage_base> cuMovemem_gpu_i64(boost::intrusive_ptr<Storage_base> &in, const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64>&mapper, const std::vector<cytnx_uint64> &invmapper, const bool is_inplace){
            return cuMovemem_gpu_general<cytnx_int64, cytnx_int64>(in,old_shape,mapper,invmapper,is_inplace);
       }

        boost::intrusive_ptr<Storage_base> cuMovemem_gpu_u64(boost::intrusive_ptr<Storage_base> &in, const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64>&mapper, const std::vector<cytnx_uint64> &invmapper, const bool is_inplace){
            return cuMovemem_gpu_general<cytnx_uint64, cytnx_uint64>(in,old_shape,mapper,invmapper,is_inplace);
        }

        boost::intrusive_ptr<Storage_base> cuMovemem_gpu_i32(boost::intrusive_ptr<Storage_base> &in, const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64>&mapper, const std::vector<cytnx_uint64> &invmapper, const bool is_inplace){
            return cuMovemem_gpu_general<cytnx_int32, cytnx_int32>(in,old_shape,mapper,invmapper,is_inplace);
        }

        boost::intrusive_ptr<Storage_base> cuMovemem_gpu_u32(boost::intrusive_ptr<Storage_base> &in, const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64>&mapper, const std::vector<cytnx_uint64> &invmapper, const bool is_inplace){
            return cuMovemem_gpu_general<cytnx_uint32, cytnx_uint32>(in,old_shape,mapper,invmapper,is_inplace);
        }
        boost::intrusive_ptr<Storage_base> cuMovemem_gpu_u16(boost::intrusive_ptr<Storage_base> &in, const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64>&mapper, const std::vector<cytnx_uint64> &invmapper, const bool is_inplace){
            return cuMovemem_gpu_general<cytnx_uint16, cytnx_uint16>(in,old_shape,mapper,invmapper,is_inplace);
        }
        boost::intrusive_ptr<Storage_base> cuMovemem_gpu_i16(boost::intrusive_ptr<Storage_base> &in, const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64>&mapper, const std::vector<cytnx_uint64> &invmapper, const bool is_inplace){
            return cuMovemem_gpu_general<cytnx_int16, cytnx_int16>(in,old_shape,mapper,invmapper,is_inplace);
        }
        boost::intrusive_ptr<Storage_base> cuMovemem_gpu_b(boost::intrusive_ptr<Storage_base> &in, const std::vector<cytnx_uint64> &old_shape, const std::vector<cytnx_uint64>&mapper, const std::vector<cytnx_uint64> &invmapper, const bool is_inplace){
            return cuMovemem_gpu_general<cytnx_bool, cytnx_bool>(in,old_shape,mapper,invmapper,is_inplace);
        }


    #endif // UNI_GPU
    }//namespace utils_internal
}//namespace cytnx
