#include "linalg/linalg.hpp"


namespace cytnx{
    namespace linalg{
        void Pow_(Tensor &Tin, const double &p){
            
            if(Tin.dtype() > 4) Tin = Tin.astype(Type.Float);


            if(Tin.device() == Device.cpu){
                cytnx::linalg_internal::lii.Pow_ii[Tin.dtype()](Tin._impl->storage()._impl,Tin._impl->storage()._impl,Tin._impl->storage()._impl->size(),p);
            }else{
                #ifdef UNI_GPU
                checkCudaErrors(cudaSetDevice(Tin.device()));
                cytnx::linalg_internal::lii.cuPow_ii[Tin.dtype()](Tin._impl->storage()._impl,Tin._impl->storage()._impl,Tin._impl->storage()._impl->size(),p);
                #else
                    cytnx_error_msg(true,"[Pow_] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
                #endif
            }


        }
    }
}// cytnx



