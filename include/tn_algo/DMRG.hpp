#ifndef _H_DMRG_
#define _H_DMRG_

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
#include "tn_algo/MPO.hpp"

namespace cytnx{
    namespace tn_algo{

        class DMRG_impl: public intrusive_ptr_base<DMRG_impl>{
            private:
                
            public:

                MPS mps;
                MPO mpo;

                //for getting excited states:
                std::vector<MPS> ortho_mps;
                double weight;

                //iterative solver param:
                cytnx_int64 maxit;
                cytnx_int64 krydim;

                //environ:
                std::vector<UniTensor> LR;
                std::vector<UniTensor> hLR; //excited states 

                friend class MPS;
                friend class MPO;

                void initialize();

        };



        // API
        class DMRG{
            private:

            public:

                ///@cond
                boost::intrusive_ptr<DMRG_impl> _impl;
                DMRG(): _impl(new DMRG_impl()){
                    // currently default init is DMRG_impl; 
                };

                DMRG(const DMRG &rhs){
                    _impl = rhs._impl;
                }
                ///@endcond

                DMRG& operator=(const DMRG &rhs){
                    _impl = rhs._impl;
                    return *this;
                }


                


        };


         



    }
}

#endif

