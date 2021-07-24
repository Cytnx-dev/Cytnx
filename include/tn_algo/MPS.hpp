#ifndef _H_MPS_
#define _H_MPS_

#include "cytnx_error.hpp"
#include "Device.hpp"
#include "intrusive_ptr_base.hpp"
#include "UniTensor.hpp"
#include <iostream>
#include <fstream>

#include "utils/vec_clone.hpp"
//#include "utils/dynamic_arg_resolver.hpp"
//#include "linalg.hpp"
#include "Accessor.hpp"
#include <vector>
#include <initializer_list>
#include <string>
#include "Scalar.hpp"

namespace cytnx{
    namespace tn_algo{

        class MPS_impl: public intrusive_ptr_base<MPS_impl>{
            private:
                
            public:
                friend class MPS;
                
                // place holder for the tensors:
                std::vector<UniTensor> _TNs;
                boost::intrusive_ptr<MPS_impl> clone() const{
                    boost::intrusive_ptr<MPS_impl> out(new MPS_impl());
                    out->_TNs = vec_clone(this->_TNs); 
                    return out;                   
                }


                virtual std::ostream& Print(std::ostream &os);
                virtual cytnx_uint64 size(){return 0;}; 
                virtual void Init(const cytnx_uint64 &N, const cytnx_uint64 &phys_dim, const cytnx_uint64 &virt_dim);
                
        };

        class RegularMPS: public MPS_impl{
            public:
                

                // specialization:
                std::ostream& Print(std::ostream &os);
                cytnx_uint64 size(){return this->_TNs.size();};    
                void Init(const cytnx_uint64 &N, const cytnx_uint64 &phys_dim, const cytnx_uint64 &virt_dim);



        };


        // API
        class MPS{
            private:

            public:

                ///@cond
                boost::intrusive_ptr<MPS_impl> _impl;
                MPS(): _impl(new RegularMPS()){
                    // currently default init is RegularMPS;:
                };

                MPS(const cytnx_uint64 &N, const cytnx_uint64 &phys_dim, const cytnx_uint64 &virt_dim): _impl(new RegularMPS()){
                    this->Init(N,phys_dim, virt_dim);
                };

                MPS(const MPS &rhs){
                    _impl = rhs._impl;
                }

                MPS& operator=(const MPS &rhs){
                    _impl = rhs._impl;
                    return *this;
                }
                ///@endcond

               
                // Initialization API:
                //-----------------------
                MPS& Init(const cytnx_uint64 &N, const cytnx_uint64 &phys_dim, const cytnx_uint64 &virt_dim){
                    this->_impl->Init(N, phys_dim, virt_dim);
                    return *this;
                }
                //-----------------------

                cytnx_uint64 size(){
                    return this->_impl->size();
                }                


        };

        std::ostream& operator<<(std::ostream& os, const MPS &in);

        



    }
}


#endif

