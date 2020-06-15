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
    
    class LinOp{
        private:
            // function pointer:
            std::function<Tensor(const Tensor&)> _mvfunc;
        
            // type:
            std::string _type;

        public:
        /// @cond
        // we need driver of void f(nx,vin,vout)
        /// @endcond

        /**
        @brief Linear Operator class for iterative solvers.
        @param type the type of operator, currently it can only be "mv" (matvec)
        @param custom_f the custom function that defines the operation on the input vector. 
        
        ## Note:
            the custom_f should be a function with signature Tensor f(const Tensor &)
            
        ## Details:
            The LinOp class is a class that defines a custom Linear operation acting on a Tensor. 
            To use, either provide a function with proper signature (using set_func(), or via initialize) or inherit this class. 
            See the following examples for how to use them. 
            
        ## Example:
        ### c++ API:
        \include example/LinOp/init.cpp
        #### output>
        \verbinclude example/LinOp/init.cpp.out
        ### python API:
        \include example/LinOp/init.py               
        #### output>
        \verbinclude example/LinOp/init.py.out

        */
        LinOp(const std::string &type, std::function<Tensor(const Tensor&)> custom_f = nullptr ){
            cytnx_error_msg(type!="mv","[ERROR][LinOp] currently only type=\"mv\" (matvec) can be used.%s","\n");
            this->_type = type; 
            this->_mvfunc=custom_f;

        };

        void set_func(std::function<Tensor(const Tensor&)> custom_f){
            this->_mvfunc = custom_f;
        }
        
        /// @cond
        // this expose to interitance:
        // need user to check the output to be Tensor
        /// @endcond
        virtual Tensor matvec(const Tensor &Tin); 
    
        
       
    };

}


#endif
