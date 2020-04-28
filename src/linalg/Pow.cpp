#include "linalg.hpp"
#include "linalg_internal_interface.hpp"

namespace cytnx{
    namespace linalg{
        Tensor Pow(const Tensor &Tin, const double &p){
            
            Tensor out;
            if(Tin.dtype() > 4) out = Tin.astype(Type.Double);
            else out = Tin.clone();            


            if(Tin.device() == Device.cpu){
                cytnx::linalg_internal::lii.Pow_ii[out.dtype()](out._impl->storage()._impl,out._impl->storage()._impl,out._impl->storage()._impl->size(),p);
            }else{
                #ifdef UNI_GPU
                //checkCudaErrors(cudaSetDevice(out.device()));
                //cytnx::linalg_internal::lii.cuPow_ii[out.dtype()](out._impl->storage()._impl,Tin._impl->storage()._impl,Tin._impl->storage()._impl->size(),p);
                cytnx_error_msg(true,"[Pow][GPU] developing%s","\n");
                #else
                    cytnx_error_msg(true,"[Pow] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
                #endif
            }

            return out;

        }
    }
}// cytnx



