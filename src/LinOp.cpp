#include "LinOp.hpp"

namespace cytnx{
    Tensor LinOp::matvec(const Tensor &Tin){
        if(this->_mvfunc == nullptr){
            cytnx_error_msg(true,"[ERROR][LinOp] LinOp required assign the Linear mapping function before using it.%s","\n");
        }
        return _mvfunc(Tin);
    }
    
}


