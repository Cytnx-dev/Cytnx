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
            this->LR.back() = R0;
            
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


            //prepare if calculate excited states:
            this->hLRs.resize(this->ortho_mps.size());
            for(cytnx_int64 ip=0;ip<this->ortho_mps.size();ip++){
                auto omps = this->ortho_mps[ip];
                
                //init environ:
                auto hL0 = UniTensor(zeros({1,1}),0); //Left boundary
                auto hR0 = UniTensor(zeros({1,1}),0); //Right boundary
                hL0.get_block_()(0,0) = 1.; hR0.get_block_()(0,0) = 1.;

                this->hLRs[ip].resize(this->mps.size()+1);

                // hLR is the alias/ref:
                auto &hLR = hLRs[ip];
                
                hLR[0] = hL0; hLR.back() = hR0;

                for(cytnx_int64 p=0;p<this->mps.size()-1;p++){
                    //anet = cytnx.Network("hL_AMAH.net")
                    //anet.PutUniTensors(["hL","Av","Ap"],[hLR[p],self.mps.A[p],omps.A[p].Conj()],is_clone=False);

                    // hard coded network:
                    auto Lenv = hLR[p].relabel({-1,-2});
                    auto Av = this->mps.data()[p].relabel({-1,-4,1});
                    auto Ap = this->mps.data()[p].Conj(); Ap.set_labels({-2,-4,2});
                    
                    hLR[p+1] = Network::Contract({Lenv,Av,Ap},";1,2").Launch(true);
                }
                

            }
            


        }//DMRG_impl::initialize



        void DMRG_impl::sweep(){
            
            // a. Optimize from right-to-left:
            /*
              psi:                   Projector:
              
                --A[p]--A[p+1]--s--              --         --
                   |       |                     |    | |    |
                                                LR[p]-M-M-LR[p+1]
                                                 |    | |    |
                                                 --         --
              b. Transfer matrix from right to left :
               LR[-1]:       LR[-2]:            
             
                   ---          ---A[-1]---         
                     |               |    | 
                   --MR         -----M--LR[-1]   ......
                     |               |    |
                   ---          ---A*[-1]--
             
              c. For Right to Left, we want A's to be in shape
                         -------------      
                        /             \     
               virt ____| chi     chi |____ virt
                        |             |     
               phys ____| 2           |        
                        \             /     
                         -------------      
            */
            /*
            for(cytnx_int64 p=this->mps.size()-2;p>-1;p--){ //in range(self.mps.Nsites()-2,-1,-1): 

                auto dim_l  = this->mps.data()[p].shape()[0];
                auto dim_r = this->mps.data()[p+1].shape()[2];


                auto psi = Contract(this->mps.data()[p],self.mps.data()[p+1]);// contract

                auto lbl = psi.labels() // memorize label
                auto psi_T = psi.get_block_(); psi_T.flatten_() // flatten to 1d

                cytnx_uint64 new_dim = min(dim_l*this->mps.phys_dim,dim_r*this->mps.phys_dim,this->mps.virt_dim);
                

                // calculate local ortho_mps:
                //omps = []
                std::vector<Tensor> omps;
                anet = cytnx.Network("hL_AA_hR.net")
                for ip in range(len(self.ortho_mps)):
                    opsi = cytnx.Contract(self.ortho_mps[ip].A[p],self.ortho_mps[ip].A[p+1])
                    opsi.set_rowrank(0)
                    anet.PutUniTensors(["hL","psi","hR"],[self.hLRs[ip][p],opsi,self.hLRs[ip][p+2]],is_clone=False)
                    out = anet.Launch(optimal=True).get_block_()
                    omps.append(out) 
                    omps[-1].flatten_()
                    
                                
                
                psi_T, Entemp = optimize_psi(psi_T, (self.LR[p],self.mpo.get_op(p),self.mpo.get_op(p+1),self.LR[p+2]),self.maxit, self.krydim,omps,self.weight)
                psi_T.reshape_(dim_l, self.mps.phys_dim, self.mps.phys_dim, dim_r) ## convert psi back to 4-leg form 
                psi = cytnx.UniTensor(psi_T,2);    
                psi.set_labels(lbl);
                #self.Ekeep.append(Entemp);

                s,self.mps.A[p],self.mps.A[p+1] = cytnx.linalg.Svd_truncate(psi,new_dim)

                # s = s.Div(s.get_block_().Norm().item()) 
                # s.Div_(s.get_block_().Norm().item()) // a bug : cannot use
                slabel = s.labels()
                s = s/s.get_block_().Norm().item() 
                s.set_labels(slabel)


                self.mps.A[p] = cytnx.Contract(self.mps.A[p],s) ## absorb s into next neighbor
                self.mps.S_loc = p


                # A[p].print_diagram()
                # A[p+1].print_diagram()

                # update LR from right to left:
                anet = cytnx.Network("R_AMAH.net")
                anet.PutUniTensors(["R","B","M","B_Conj"],[self.LR[p+2],self.mps.A[p+1],self.mpo.get_op(p+1),self.mps.A[p+1].Conj()],is_clone=False)
                self.LR[p+1] = anet.Launch(optimal=True)


                # update hLR from right to left for excited states:
                for ip in range(len(self.ortho_mps)):
                    omps = self.ortho_mps[ip]
                    anet = cytnx.Network("hR_AMAH.net")
                    anet.PutUniTensors(["hR","Bv","Bp"],[self.hLRs[ip][p+2],self.mps.A[p+1],omps.A[p+1].Conj()],is_clone=False)
                    self.hLRs[ip][p+1] = anet.Launch(optimal=True)
                
                #print('Sweep[r->l]: %d/%d, Loc:%d,Energy: %f'%(k,numsweeps,p,Ekeep[-1]))

            self.mps.A[0].set_rowrank(1)
            _,self.mps.A[0] = cytnx.linalg.Svd(self.mps.A[0],is_U=False, is_vT=True)
            self.mps.S_loc = -1

            */


        }//DMRG_impl::sweep




    }//tn_algo

}//cytnx



