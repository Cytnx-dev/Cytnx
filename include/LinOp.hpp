#ifndef _H_LinOp_
#define _H_LinOp_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include "Tensor.hpp"
#include <vector>
#include <fstream>
#include <functional>
#include "intrusive_ptr_base.hpp"
#include "utils/vec_clone.hpp"
namespace cytnx{
    /// @cond

    class LinOp{
        private:
            // function pointer:
            std::function<Tensor(const Tensor&)> _mvfunc;
        
            // type:
            std::string _type;

        public:

        // we need driver of void f(nx,vin,vout)
        LinOp(): _mvfunc(nullptr){};

        void Init(const std::string &type, std::function<Tensor(const Tensor&)> custom_f){
            if(type != "mv"){
                cytnx_error_msg(true,"[ERROR][Currently only mv (matvec)]%s","\n");
            }
            this->_mvfunc = custom_f;
            this->_type = type;
        }
        
        // this expose to interitance:
        // need user to check the output to be Tensor
        virtual Tensor matvec(const Tensor &Tin); 
    
        
       
    };
    /// @endcond
}


#endif
