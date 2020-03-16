#include "random.hpp"
#include "random_internal_interface.hpp"

namespace cytnx{
    namespace random{
        void Make_normal(Storage &Sin, const double &mean, const double &std, const unsigned int &seed){
            cytnx_error_msg((Sin.dtype()<1) || (Sin.dtype()>4),"[ERROR][Random.Make_normal] Normal distribution only accept real/imag floating type.%s","\n");
            if(Sin.device()==Device.cpu){
                random_internal::rii.Normal[Sin.dtype()](Sin._impl,mean,std,seed);
            }else{
                #ifdef UNI_GPU
                    cytnx_error_msg(true,"[Developing][Make_normal]%s","\n");
                #else
                    cytnx_error_msg(true,"[ERROR][Make_normal] Tensor is on GPU without CUDA support.%s","\n");
                #endif
            }

        }
        void Make_normal(Tensor  &Tin, const double &mean, const double &std, const unsigned int &seed){
            cytnx_error_msg((Tin.dtype()<1) || (Tin.dtype()>4),"[ERROR][Random.Make_normal] Normal distribution only accept real/imag floating type.%s","\n");
            if(Tin.device()==Device.cpu){
                random_internal::rii.Normal[Tin.dtype()](Tin._impl->storage()._impl,mean,std,seed);
            }else{
                #ifdef UNI_GPU
                    cytnx_error_msg(true,"[Developing][Make_normal]%s","\n");
                #else
                    cytnx_error_msg(true,"[ERROR][Make_normal] Tensor is on GPU without CUDA support.%s","\n");
                #endif
            }
        }

    }
}
