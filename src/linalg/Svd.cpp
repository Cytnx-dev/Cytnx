#include "linalg/linalg.hpp"
#include <iostream>

namespace tor10{

    std::vector<Tensor> Svd(const Tensor &Tin, const bool &is_U, const bool &is_vT){
        
        tor10_error_msg(Tin.shape().size() != 2,"[Add] error, Svd can only operate on rank-2 Tensor.%s","\n");
        tor10_error_msg(!Tin.is_contiguous(), "[Add] error tensor must be contiguous. Call Contiguous_() or Contiguous() first%s","\n");
        
        tor10_uint64 n_singlu = std::max(tor10_uint64(1),std::min(Tin.shape()[0],Tin.shape()[1])); 

        Tensor in;
        if(Tin.dtype() > tor10type.Float) in = Tin.astype(tor10type.Float);
        else in = Tin;

        std::cout << n_singlu << std::endl;

        Tensor U,S,vT;

        if(Tin.device()==tor10device.cpu){

            S.Init({n_singlu},in.dtype()<=2?in.dtype()+2:in.dtype(),in.device()); // if type is complex, S should be real
            if(is_U){ U.Init({in.shape()[0],n_singlu},in.dtype(),in.device());  }
            if(is_vT){ vT.Init({n_singlu,in.shape()[1]},in.dtype(),in.device());}

            linalg_internal::lii.Svd_ii[in.dtype()](in._impl->_get_storage(), 
                                                    U._impl->_get_storage(),
                                                    vT._impl->_get_storage(),  
                                                    S._impl->_get_storage(),in.shape()[0],in.shape()[1]);

            std::vector<Tensor> out;
            out.push_back(S);
            if(is_U) out.push_back(U);
            if(is_vT) out.push_back(vT);
            
            return out;

        }else{


        }    

       


    }


}


