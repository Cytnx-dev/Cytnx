#include "linalg.hpp"
#include "linalg_internal_interface.hpp"

#include <iostream>
#include <vector>

using namespace std;
namespace cytnx{
    namespace linalg{
         
        std::vector<Tensor> Svd(const Tensor &Tin, const bool &is_U, const bool &is_vT){
            
            cytnx_error_msg(Tin.shape().size() != 2,"[Svd] error, Svd can only operate on rank-2 Tensor.%s","\n");
            cytnx_error_msg(!Tin.is_contiguous(), "[Svd] error tensor must be contiguous. Call Contiguous_() or Contiguous() first%s","\n");
            
            cytnx_uint64 n_singlu = std::max(cytnx_uint64(1),std::min(Tin.shape()[0],Tin.shape()[1])); 

            Tensor in;
            if(Tin.dtype() > Type.Float) in = Tin.astype(Type.Double);
            else in = Tin;

            //std::cout << n_singlu << std::endl;

            Tensor U,S,vT;
            S.Init({n_singlu},in.dtype()<=2?in.dtype()+2:in.dtype(),in.device()); // if type is complex, S should be real
            S.storage().set_zeros();
            if(is_U){ U.Init({in.shape()[0],n_singlu},in.dtype(),in.device()); U.storage().set_zeros(); }
            if(is_vT){ vT.Init({n_singlu,in.shape()[1]},in.dtype(),in.device()); vT.storage().set_zeros();}



            if(Tin.device()==Device.cpu){


                cytnx::linalg_internal::lii.Svd_ii[in.dtype()](in._impl->storage()._impl, 
                                                        U._impl->storage()._impl,
                                                        vT._impl->storage()._impl,  
                                                        S._impl->storage()._impl,in.shape()[0],in.shape()[1]);

                std::vector<Tensor> out;
                out.push_back(S);
                if(is_U) out.push_back(U);
                if(is_vT) out.push_back(vT);
                
                return out;

            }else{
                #ifdef UNI_GPU
                    checkCudaErrors(cudaSetDevice(in.device()));
                    cytnx::linalg_internal::lii.cuSvd_ii[in.dtype()](in._impl->storage()._impl,
                                                            U._impl->storage()._impl,
                                                            vT._impl->storage()._impl,
                                                            S._impl->storage()._impl,in.shape()[0],in.shape()[1]);

                    std::vector<Tensor> out;
                    out.push_back(S);
                    if(is_U) out.push_back(U);
                    if(is_vT) out.push_back(vT);
                    
                    return out;
                #else
                    cytnx_error_msg(true,"[Svd] fatal error,%s","try to call the gpu section without CUDA support.\n");
                    return std::vector<Tensor>();
                #endif
            }    

        }




    }//linalg namespace

}//cytnx namespace

namespace cytnx_extension{
    namespace xlinalg{
        using namespace cytnx;
        std::vector<cytnx_extension::CyTensor> Svd(const cytnx_extension::CyTensor &Tin, const bool &is_U, const bool &is_vT){
            if(Tin.is_blockform()){
                cytnx_error_msg(true,"[Svd][Developing] Svd for SparseCyTensor is developing.%s","\n");
            }else{
                // using rowrank to split the bond to form a matrix.
                cytnx_error_msg(Tin.Rowrank() < 1 || Tin.rank()==1,"[Svd][ERROR] Svd for DenseCyTensor should have rank>1 and Rowrank>0%s","\n");

               Tensor tmp;
               if(Tin.is_contiguous()) tmp = Tin.get_block_();
               else{ tmp = Tin.get_block(); tmp.contiguous_();}

               vector<cytnx_uint64> tmps = tmp.shape();
               vector<cytnx_int64> oldshape(tmps.begin(),tmps.end()); tmps.clear();
               vector<cytnx_int64> oldlabel = Tin.labels();

               // collapse as Matrix:
               cytnx_int64 rowdim = 1;
               for(cytnx_uint64 i=0;i<Tin.Rowrank();i++) rowdim*= tmp.shape()[i];
               tmp.reshape_({rowdim,-1});

               vector<Tensor> outT = cytnx::linalg::Svd(tmp,is_U,is_vT);
               
               int t=0;
               vector<cytnx_extension::CyTensor> outCyT(outT.size());

               //s
               cytnx_extension::CyTensor &Cy_S = outCyT[t];  
               cytnx_extension::Bond newBond(outT[t].shape()[0]);
               cytnx_int64 newlbl = -1;
               for(int i=0;i<oldlabel.size();i++){
                   if(oldlabel[i]<=newlbl) newlbl = oldlabel[i]-1;
               }
               Cy_S.Init({newBond,newBond},{newlbl,newlbl-1},1,Type.Double,Device.cpu,true); //it is just reference so no hurt to alias ^^
               Cy_S.put_block_(outT[t]);
               t++; 
               if(is_U){
                   cytnx_extension::CyTensor &Cy_U = outCyT[t]; 
                   vector<cytnx_int64> shapeU = vec_clone(oldshape,Tin.Rowrank());
                   shapeU.push_back(-1);
                   outT[t].reshape_(shapeU);
                   Cy_U.Init(outT[t],Tin.Rowrank()); 
                   vector<cytnx_int64> labelU = vec_clone(oldlabel,Tin.Rowrank());
                   labelU.push_back(Cy_S.labels()[0]);
                   Cy_U.set_labels(labelU);
                   t++; // U
               }

               if(is_vT){
                   cytnx_extension::CyTensor &Cy_vT = outCyT[t]; 
                   vector<cytnx_int64> shapevT(Tin.rank()-Tin.Rowrank()+1);
                   shapevT[0] = -1; memcpy(&shapevT[1],&oldshape[Tin.Rowrank()],sizeof(cytnx_int64)*(shapevT.size()-1));

                   outT[t].reshape_(shapevT);
                   Cy_vT.Init(outT[t],1); 
                   vector<cytnx_int64> labelvT(shapevT.size());
                   labelvT[0] = Cy_S.labels()[1]; memcpy(&labelvT[1],&oldlabel[Tin.Rowrank()],sizeof(cytnx_int64)*(labelvT.size()-1));
                   Cy_vT.set_labels(labelvT);
                   t++; // vT
               }

               // if tag, then update  the tagging informations
               if(Tin.is_tag()){
                   Cy_S.tag();                
                   t = 1;
                   if(is_U){
                        cytnx_extension::CyTensor &Cy_U = outCyT[t]; 
                        Cy_U._impl->_is_tag = true;
                        for(int i=0;i<Cy_U.Rowrank();i++){
                            Cy_U.bonds()[i].set_type(Tin.bonds()[i].type());
                        }
                        Cy_U.bonds().back().set_type(cytnx_extension::BD_BRA);
                        Cy_U._impl->_is_braket_form = Cy_U._impl->_update_braket();
                        t++;
                   }
                   if(is_vT){
                        cytnx_extension::CyTensor &Cy_vT = outCyT[t]; 
                        Cy_vT._impl->_is_tag = true;
                        Cy_vT.bonds()[0].set_type(cytnx_extension::BD_KET);
                        for(int i=1;i<Cy_vT.rank();i++){
                            Cy_vT.bonds()[i].set_type(Tin.bonds()[Tin.Rowrank()+i-1].type());
                        }
                        Cy_vT._impl->_is_braket_form = Cy_vT._impl->_update_braket();
                        t++;
                   }

               }// if tag

               return outCyT;


            }//is block form ?

        }//Svd


    }//linalg namespace
}//cytnx_extension namespace


