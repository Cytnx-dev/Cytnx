#include "linalg.hpp"
#include "Accessor.hpp"
#include <vector>
namespace cytnx{
    namespace linalg{
        typedef Accessor ac;
        std::vector<Tensor> Svd_truncate(const Tensor &Tin, const cytnx_uint64 &keepdim, const bool &is_U, const bool &is_vT){
            
            std::vector<Tensor> tmps = Svd(Tin,is_U,is_vT);
               
            cytnx_uint64 id=0;
            cytnx_error_msg(tmps[0].shape()[0] < keepdim,"[ERROR] keepdim should be <= the valid # of singular value, %d!\n",tmps[0].shape()[0]);

            tmps[id] = tmps[id].get({ac::range(0,keepdim)});
            
            if(is_U){
                id++;
                tmps[id] = tmps[id].get({ac::all(),ac::range(0,keepdim)});
            }
            if(is_vT){
                id++;
                tmps[id] = tmps[id].get({ac::range(0,keepdim),ac::all()});
            }
            return tmps;
        }            
    }
}


namespace cytnx{
    namespace linalg{
        using namespace std;
        typedef Accessor ac;
        std::vector<cytnx::UniTensor> Svd_truncate(const cytnx::UniTensor &Tin, const cytnx_uint64 &keepdim, const bool &is_U, const bool &is_vT){
            if(Tin.is_blockform()){
                cytnx_error_msg(true,"[Svd][Developing] Svd for SparseUniTensor is developing.%s","\n");
            }else{
               // using rowrank to split the bond to form a matrix.
               cytnx_error_msg((Tin.rowrank() < 1 || Tin.rank()==1),"[Svd][ERROR] Svd for DenseUniTensor should have rank>1 and rowrank>0%s","\n");
               
               Tensor tmp;
               if(Tin.is_contiguous()) tmp = Tin.get_block_();
               else{ tmp = Tin.get_block(); tmp.contiguous_();}
               

               vector<cytnx_uint64> tmps = tmp.shape();
               vector<cytnx_int64> oldshape(tmps.begin(),tmps.end()); tmps.clear();
               vector<cytnx_int64> oldlabel = Tin.labels();

               // collapse as Matrix:
               cytnx_int64 rowdim = 1;
               for(cytnx_uint64 i=0;i<Tin.rowrank();i++) rowdim*= tmp.shape()[i];
               tmp.reshape_({rowdim,-1});

               vector<Tensor> outT = cytnx::linalg::Svd(tmp,is_U,is_vT);
               if(Tin.is_contiguous()) tmp.reshape_(oldshape);                       
              

               int t=0;
               vector<cytnx::UniTensor> outCyT(outT.size());

               //s
               cytnx_error_msg(keepdim>outT[t].shape()[0],"[ERROR][Svd_truncate] keepdim should <= dimension of singular tensor%s","\n");

               cytnx::UniTensor &Cy_S = outCyT[t];  
               cytnx::Bond newBond(keepdim);
               cytnx_int64 newlbl = -1;
               for(int i=0;i<oldlabel.size();i++){
                   if(oldlabel[i]<=newlbl) newlbl = oldlabel[i]-1;
               }
               Cy_S.Init({newBond,newBond},{newlbl,newlbl-1},1,Type.Double,Device.cpu,true); //it is just reference so no hurt to alias ^^
               Tensor tmps2 = outT[t].get({ac::range(0,keepdim)});
               Cy_S.put_block_(tmps2);
               t++; 

               if(is_U){
                   cytnx::UniTensor &Cy_U = outCyT[t]; 
                   // shape
                   vector<cytnx_int64> shapeU = vec_clone(oldshape,Tin.rowrank());
                   shapeU.push_back(-1);

                   // trim mem
                   outT[t] = outT[t].get({ac::all(),ac::range(0,keepdim)});
                   outT[t].reshape_(shapeU);

                    
                   Cy_U.Init(outT[t],Tin.rowrank()); 
                   vector<cytnx_int64> labelU = vec_clone(oldlabel,Tin.rowrank());
                   labelU.push_back(Cy_S.labels()[0]);
                   Cy_U.set_labels(labelU);
                   t++; // U
               }

               if(is_vT){
                   cytnx::UniTensor &Cy_vT = outCyT[t]; 
                   //shape
                   vector<cytnx_int64> shapevT(Tin.rank()-Tin.rowrank()+1);
                   shapevT[0] = -1; memcpy(&shapevT[1],&oldshape[Tin.rowrank()],sizeof(cytnx_int64)*(shapevT.size()-1));

                   // trim mem
                   outT[t] = outT[t].get({ac::range(0,keepdim),ac::all()});
                   outT[t].reshape_(shapevT);

                   
                   Cy_vT.Init(outT[t],1); 
                   vector<cytnx_int64> labelvT(shapevT.size());
                   labelvT[0] = Cy_S.labels()[1]; memcpy(&labelvT[1],&oldlabel[Tin.rowrank()],sizeof(cytnx_int64)*(labelvT.size()-1));
                   Cy_vT.set_labels(labelvT);
                   t++; // vT
               }

               // if tag, then update  the tagging informations
               if(Tin.is_tag()){
                   Cy_S.tag();                
                   t = 1;
                   if(is_U){
                        cytnx::UniTensor &Cy_U = outCyT[t]; 
                        Cy_U._impl->_is_tag = true;
                        for(int i=0;i<Cy_U.rowrank();i++){
                            Cy_U.bonds()[i].set_type(Tin.bonds()[i].type());
                        }
                        Cy_U.bonds().back().set_type(cytnx::BD_BRA);
                        Cy_U._impl->_is_braket_form = Cy_U._impl->_update_braket();
                        t++;
                   }
                   if(is_vT){
                        cytnx::UniTensor &Cy_vT = outCyT[t]; 
                        Cy_vT._impl->_is_tag = true;
                        Cy_vT.bonds()[0].set_type(cytnx::BD_KET);
                        for(int i=1;i<Cy_vT.rank();i++){
                            Cy_vT.bonds()[i].set_type(Tin.bonds()[Tin.rowrank()+i-1].type());
                        }
                        Cy_vT._impl->_is_braket_form = Cy_vT._impl->_update_braket();
                        t++;
                   }

               }// if tag

               return outCyT;


            }//is block form ?
            
        }// Svd_truncate

    }// linalg
}//cytnx
