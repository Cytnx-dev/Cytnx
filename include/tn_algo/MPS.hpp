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
                
        };

        class RegularMPS: public MPS_impl{
            public:
                std::ostream& Print(std::ostream &os);
                cytnx_uint64 size(){return this->_TNs.size();};    
    
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

                MPS(const MPS &rhs){
                    _impl = rhs._impl;
                }
                ///@endcond

                MPS& operator=(const MPS &rhs){
                    _impl = rhs._impl;
                    return *this;
                }

                cytnx_uint64 size(){
                    return this->_impl->size();
                }                


        };

        std::ostream& operator<<(std::ostream& os, const MPS &in);

        



    }
}




