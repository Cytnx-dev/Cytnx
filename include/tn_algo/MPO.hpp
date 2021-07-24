#ifndef _H_MPO_
#define _H_MPO_

#include "cytnx_error.hpp"
#include "Device.hpp"
#include "intrusive_ptr_base.hpp"
#include "UniTensor.hpp"
#include <iostream>
#include <fstream>
#include "utils/vec_clone.hpp"
#include "Accessor.hpp"
#include <vector>
#include <initializer_list>
#include <string>
#include "Scalar.hpp"
#include "tn_algo/MPS.hpp"


namespace cytnx{
    namespace tn_algo{

        class MPO_impl: public intrusive_ptr_base<MPO_impl>{
            private:
                
            public:

                std::vector<UniTensor> _TNs;

                friend class MPS;
                friend class MPO;

                boost::intrusive_ptr<MPO_impl> clone() const{
                    boost::intrusive_ptr<MPO_impl> out(new MPO_impl());
                    out->_TNs = vec_clone(this->_TNs);
                    return out;                   
                }


                virtual std::ostream& Print(std::ostream &os);
                virtual cytnx_uint64 size(){return 0;}; 
                
        };


        class RegularMPO: public MPO_impl{
            public:
                std::ostream& Print(std::ostream &os);
                cytnx_uint64 size(){return this->_TNs.size();};    
    
        };


        // API
        class MPO{
            private:

            public:

                ///@cond
                boost::intrusive_ptr<MPO_impl> _impl;
                MPO(): _impl(new RegularMPO()){
                    // currently default init is RegularMPO; 
                };

                MPO(const MPO &rhs){
                    _impl = rhs._impl;
                }
                ///@endcond

                MPO& operator=(const MPO &rhs){
                    _impl = rhs._impl;
                    return *this;
                }

                cytnx_uint64 size(){
                    return this->_impl->size();
                }                


        };

        std::ostream& operator<<(std::ostream& os, const MPO &in);

        



    }
}

#endif

