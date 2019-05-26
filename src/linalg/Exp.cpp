#include "linalg/linalg.hpp"


namespace cytnx{
    namespace linalg{
        Tensor Exp(const Tensor &Tin){
            
            Tensor out;
            if(Tin.dtype() > 4) out = Tin.astype(cytnxtype.Float);
            else out = Tin.clone();            


            if(Tin.device() == cytnxdevice.cpu){
                cytnx::linalg_internal::lii.Exp_ii[out.dtype()](out._impl->_get_storage(),Tin._impl->_get_storage(),Tin._impl->_get_storage()->size());
            }else{
                #ifdef UNI_GPU
                cytnx::linalg_internal::lii.cuExp_ii[out.dtype()](out._impl->_get_storage(),Tin._impl->_get_storage(),Tin._impl->_get_storage()->size());
                #else
                    cytnx_error_msg(true,"[Exp] fatal error, the tensor is on GPU without CUDA support.%s","\n"); 
                #endif
            }

            return out;

        }
    }
}// cytnx



