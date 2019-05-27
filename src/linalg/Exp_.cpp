#include "linalg/linalg.hpp"


namespace cytnx{
    namespace linalg{
        void Exp_(Tensor &Tin){
            
            if(Tin.dtype() > 4) Tin = Tin.astype(cytnxtype.Float);


            if(Tin.device() == cytnxdevice.cpu){
                cytnx::linalg_internal::lii.Exp_ii[Tin.dtype()](Tin._impl->_get_storage()._impl,Tin._impl->_get_storage()._impl,Tin._impl->_get_storage()._impl->size());
            }else{
                #ifdef UNI_GPU
                cytnx::linalg_internal::lii.cuExp_ii[Tin.dtype()](Tin._impl->_get_storage()._impl,Tin._impl->_get_storage()._impl,Tin._impl->_get_storage()._impl->size());
                #else
                    cytnx_error_msg(true,"[Exp_] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
                #endif
            }


        }
    }
}// cytnx



