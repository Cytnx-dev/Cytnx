#include "tn_algo/DMRG.hpp"
#include "Generator.hpp"
using namespace std;

namespace cytnx{
    namespace tn_algo{

        void DMRG_impl::initialize(){
            // initialize everything
            // 1. setting env:

            // Initialiaze enviroment: 
            auto L0 = UniTensor(cytnx::zeros({this->mpo.get_op(0).shape()[0],1,1}),0); //Left boundary
            auto R0 = UniTensor(cytnx::zeros({this->mpo.get_op(this->mps.size()-1).shape()[1],1,1}),0); //Right boundary
            L0.get_block_()(0,0,0) = 1.; R0.get_block_()(-1,0,0)= 1.;

            


        }







    }

}



