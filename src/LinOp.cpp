#include "LinOp.hpp"

#ifdef UNI_OMP
#include <omp.h>
#endif

namespace cytnx{
    /*
    Tensor LinOp::_mv_elemfunc(const Tensor &Tin){
        cytnx_error_msg(this->_type!="mv_elem","[ERROR][LinOp][Internal] Fatal call _mv_elemfunc when type==mv %s","\n");
        
        #ifdef UNI_OMP
        #else
            
            //traversal all the 
            

        #endif

        

    }
    */
    Tensor LinOp::matvec(const Tensor &Tin){
        if(this->_type== "mv_elem"){
            //return this->_mv_elemfunc(Tin);
            cytnx_error_msg(true,"Developing%s","\n");
            return Tensor();
        }else{
            if(this->_mvfunc == nullptr){
                cytnx_error_msg(true,"[ERROR][LinOp] LinOp required assign the Linear mapping function before using it.%s","\n");
            }
            return _mvfunc(Tin);
        }
    }


    
    
}




