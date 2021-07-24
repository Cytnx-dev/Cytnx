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

                
                std::vector<UniTensor>& get_data(){
                    return this->_TNs;
                }
                
                virtual std::ostream& Print(std::ostream &os);
                virtual cytnx_uint64 size(){return 0;}; 
                virtual void Init(const cytnx_uint64 &N, const cytnx_uint64 &phys_dim, const cytnx_uint64 &virt_dim);
                
        };

        // finite size:
        class RegularMPS: public MPS_impl{
            public:
                
                cytnx_int64 S_loc;
                cytnx_int64 phys_dim;
                cytnx_int64 virt_dim;


                // only for this:
                RegularMPS(): S_loc(0), phys_dim(-1), virt_dim(-1){};


                // specialization:
                std::ostream& Print(std::ostream &os);
                cytnx_uint64 size(){return this->_TNs.size();};    
                void Init(const cytnx_uint64 &N, const cytnx_uint64 &phys_dim, const cytnx_uint64 &virt_dim);



        };

        // infinite size:
        class iMPS: public MPS_impl{
            public:
                
                cytnx_int64 phys_dim;
                cytnx_int64 virt_dim;


                // only for this:
                iMPS(): phys_dim(-1), virt_dim(-1){};


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
                MPS(): _impl(new MPS_impl()){
                    // currently default init is RegularMPS;:
                };

                MPS(const cytnx_uint64 &N, const cytnx_uint64 &phys_dim, const cytnx_uint64 &virt_dim, const cytnx_int64 &mps_type=0): _impl(new MPS_impl()){
                    this->Init(N,phys_dim, virt_dim,mps_type);
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
                MPS& Init(const cytnx_uint64 &N, const cytnx_uint64 &phys_dim, const cytnx_uint64 &virt_dim, const cytnx_int64 &mps_type=0){
                    if(mps_type==0){
                        this->_impl =boost::intrusive_ptr<MPS_impl>(new RegularMPS());
                    }else if(mps_type==1){
                        this->_impl =boost::intrusive_ptr<MPS_impl>(new iMPS());
                    }else{
                        cytnx_error_msg(true,"[ERROR] invalid MPS type.%s","\n");
                    }
                    this->_impl->Init(N, phys_dim, virt_dim);
                    return *this;
                }
                //-----------------------

                cytnx_uint64 size(){
                    return this->_impl->size();
                }                

                std::vector<UniTensor> &data(){return this->_impl->get_data();};

                


        };

        std::ostream& operator<<(std::ostream& os, const MPS &in);

        



    }
}


#endif

