#ifndef _H_LinOp_
#define _H_LinOp_

#include "Type.hpp"
#include "cytnx_error.hpp"
#include "Tensor.hpp"
#include <vector>
#include <fstream>
#include <functional>
#include <map>
#include <utility>
#include <algorithm>
#include "intrusive_ptr_base.hpp"
#include "utils/vec_clone.hpp"
namespace cytnx{
    
    class LinOp{
        private:
            // function pointer:
            std::function<Tensor(const Tensor&)> _mvfunc;
        
            // type:
            std::string _type;

            // nx
            cytnx_uint64 _nx;
            
            // device
            int _device;
            int _dtype;

            // pre-storage data:
            std::map<cytnx_uint64,std::pair<std::vector<cytnx_uint64>,Storage> > _elems; //map[i] -> pair[<js>,<Storage>]
            std::map<cytnx_uint64,std::pair<std::vector<cytnx_uint64>,Storage> >::iterator _elems_it; 

            //Tensor _mv_elemfunc(const Tensor &);


        public:
        /// @cond
        // we need driver of void f(nx,vin,vout)
        /// @endcond

        /**
        @brief Linear Operator class for iterative solvers.
        @param type the type of operator, currently it can only be "mv" (matvec) or "mv_elem" (matvec with pre-store element)
        @param nx the last dimension of operator, this should be the dimension of the input vector.
        @param dtype the input/output Tensor's dtype. Note that this should match the input/output Tensor's dtype of custom function. 
        @param device the input/output Tensor's device. Note that this should match the input/output Tensor's device of custom function. 
        @param custom_f the custom function that defines the operation on the input vector. 

        ## Note:
            1. the custom_f should be a function with signature Tensor f(const Tensor &)
            2. the device and dtype should be set as that required by custom_f. This should be the same as the input and output vector of custom_f. 
               by default, we assume custom_f take input and output vector to be on CPU and Double type. 
            
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
        LinOp(const std::string &type, const cytnx_uint64 &nx, const int &dtype=Type.Double, const int &device=Device.cpu, std::function<Tensor(const Tensor&)> custom_f = nullptr){
            if(type=="mv"){
                this->_mvfunc=custom_f;

            }else if(type=="mv_elem"){
                cytnx_error_msg(custom_f != nullptr,"[ERROR][LinOp] with type=mv_elem cannot accept any function. use set_elem/set_elemts instead.%s","\n");
                    
            
            }else
                cytnx_error_msg(type!="mv","[ERROR][LinOp] currently only type=\"mv\" (matvec) can be used.%s","\n");

            this->_type = type; 
            this->_nx = nx;
            cytnx_error_msg(device<-1 || device >=Device.Ngpus,"[ERROR] invalid device.%s","\n");
            this->_device = device;
            cytnx_error_msg(dtype<1 || dtype >= N_Type,"[ERROR] invalid dtype.%s","\n");
            this->_dtype = dtype;
        };
        void set_func(std::function<Tensor(const Tensor&)> custom_f, const int &dtype, const int &device){
            if(this->_type=="mv"){
                this->_mvfunc = custom_f;
                cytnx_error_msg(device<-1 || device >=Device.Ngpus,"[ERROR] invalid device.%s","\n");
                this->_device = device;
                cytnx_error_msg(dtype<1 || dtype >= N_Type,"[ERROR] invalid dtype.%s","\n");
                this->_dtype = dtype;
            }else{
                cytnx_error_msg(true,"[ERROR] cannot specify func with type=mv_elem%s. use set_elem instead.","\n");
            }
        };
        template<class T>
        void set_elem(const cytnx_uint64 &i, const cytnx_uint64 &j, const T &elem, const bool check_exists=true){
            this->_elems_it = this->_elems.find(i);
            if(this->_elems_it == this->_elems.end()){
                //not exists:
                Storage x(0,this->_dtype);
                x.append(elem);                
                this->_elems[i] = std::pair<std::vector<cytnx_uint64>,Storage>({j},x);

            }else{
                std::vector<cytnx_uint64> &vi = this->_elems_it->second.first; // pair:
                Storage &ie = this->_elems_it->second.second;
                if(check_exists){
                    cytnx_error_msg(std::find(vi.begin(), vi.end(), j)!=vi.end(),"[ERROR] the element is set%s","\n");
                }
                vi.push_back(j);
                ie.append(elem);
            }

        };
        void set_device(const int &device){
            cytnx_error_msg(device<-1 || device >=Device.Ngpus,"[ERROR] invalid device.%s","\n");
            this->_device = device;
        };
        void set_dtype(const int &dtype){
            cytnx_error_msg(dtype<1 || dtype >= N_Type,"[ERROR] invalid dtype.%s","\n");
            this->_dtype = dtype;
        };
        int device() const{
            return this->_device;
        };
        int dtype() const{
            return this->_dtype;
        };
        cytnx_uint64 nx() const{
            return this->_nx;
        };

        /// @cond
        // this expose to interitance:
        // need user to check the output to be Tensor
        /// @endcond
        virtual Tensor matvec(const Tensor &Tin); 
    
        
       
    };

}


#endif
