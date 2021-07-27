#include "tn_algo/DMRG.hpp"
#include "Generator.hpp"
#include "Network.hpp"
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

            // Put in the left normalization form and calculate transfer matrices LR
            /*
             LR[0]:        LR[1]:            LR[2]:
             
                -----      -----A[0]---     -----A[1]---
                |          |     |          |     |
               ML----     LR[0]--M-----    LR[1]--M-----      ......
                |          |     |          |     |
                -----      -----A*[0]--     -----A*[1]--
             
             
              L_AMAH.net file is used to contract the LR[i]
              >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            */
            this->LR.resize(this->mps.size()+1);
            this->LR[0]  = L0;
            this->LR[-1] = R0;
            
            for(int p=0;p<this->mps.size() - 1;p++){
                this->mps.S_mvright();
                //anet = cytnx.Network("L_AMAH.net")
                //anet.PutUniTensors(["L","A","A_Conj","M"],[self.LR[p],self.mps.A[p],self.mps.A[p].Conj(),self.mpo.get_op(p)],is_clone=False);
                
                // hard coded the network:
                auto Lenv = this->LR[p].relabel({-2,-1,-3});
                auto tA = this->mps.data()[p].relabel({-1,-4,1});
                auto tAc = this->mps.data()[p].Conj(); tAc.set_labels({-3,-5,2});
                auto M = this->mpo.get_op(p).relabel({-2,0,-4,-5});
                this->LR[p+1] = Network::Contract({Lenv,tA,tAc,M},";0,1,2").Launch(true);

            }
            this->mps.S_mvright();
            
            


        }//DMRG_impl::initialize







    }//tn_algo

}//cytnx



