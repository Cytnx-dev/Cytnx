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


namespace cytnx_extension{
    namespace xlinalg{
        using namespace std;
        using namespace cytnx;
        typedef Accessor ac;
        std::vector<cytnx_extension::CyTensor> Svd_truncate(const cytnx_extension::CyTensor &Tin, const cytnx_uint64 &keepdim, const bool &is_U, const bool &is_vT){
            if(Tin.is_blockform()){
                cytnx_error_msg(true,"[Svd][Developing] Svd for SparseCyTensor is developing.%s","\n");
            }else{
                // using rowrank to split the bond to form a matrix.
                if(Tin.Rowrank() < 1 || Tin.rank()==1){
                    cytnx_error_msg(true,"[Svd][ERROR] Svd for DenseCyTensor should have rank>1 and Rowrank>0%s","\n");
                    return  std::vector<cytnx_extension::CyTensor>();
                }else{
                   if(Tin.is_tag()){
                        cytnx_error_msg(true,"[Svd][Developing] Svd for tagged CyTensor is developing.%s","\n");
                        return  std::vector<cytnx_extension::CyTensor>();
                   }else{
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
                       cytnx_error_msg(keepdim>outT[t].shape()[0],"[ERROR][Svd_truncate] keepdim should <= dimension of singular tensor%s","\n");

                       cytnx_extension::CyTensor &Cy_S = outCyT[t];  
                       cytnx_extension::Bond newBond(keepdim);
                       cytnx_int64 newlbl = -1;
                       for(int i=0;i<oldlabel.size();i++){
                           if(oldlabel[i]<=newlbl) newlbl = oldlabel[i]-1;
                       }
                       Cy_S.Init({newBond,newBond},{newlbl,newlbl-1},1,Type.Double,Device.cpu,true); //it is just reference so no hurt to alias ^^
                       Cy_S.put_block_(outT[t].get({ac::range(0,keepdim)}));
                       t++; 

                       if(is_U){
                           cytnx_extension::CyTensor &Cy_U = outCyT[t]; 
                           // shape
                           vector<cytnx_int64> shapeU = vec_clone(oldshape,Tin.Rowrank());
                           shapeU.push_back(-1);

                           // trim mem
                           outT[t] = outT[t].get({ac::all(),ac::range(0,keepdim)});
                           outT[t].reshape_(shapeU);

                            
                           Cy_U.Init(outT[t],Tin.Rowrank()); 
                           vector<cytnx_int64> labelU = vec_clone(oldlabel,Tin.Rowrank());
                           labelU.push_back(Cy_S.labels()[0]);
                           Cy_U.set_labels(labelU);
                           t++; // U
                       }

                       if(is_vT){
                           cytnx_extension::CyTensor &Cy_vT = outCyT[t]; 
                           //shape
                           vector<cytnx_int64> shapevT(Tin.rank()-Tin.Rowrank()+1);
                           shapevT[0] = -1; memcpy(&shapevT[1],&oldshape[Tin.Rowrank()],sizeof(cytnx_int64)*(shapevT.size()-1));

                           // trim mem
                           outT[t] = outT[t].get({ac::range(0,keepdim),ac::all()});
                           outT[t].reshape_(shapevT);

                           
                           Cy_vT.Init(outT[t],1); 
                           vector<cytnx_int64> labelvT(shapevT.size());
                           labelvT[0] = Cy_S.labels()[1]; memcpy(&labelvT[1],&oldlabel[Tin.Rowrank()],sizeof(cytnx_int64)*(labelvT.size()-1));
                           Cy_vT.set_labels(labelvT);
                           t++; // vT
                       }


                       return outCyT;

                   }// if tagged

                }// Rowrank is qualify? 
            }//is block form ?
            
        }// Svd_truncate

    }// xlinalg
}//cytnx_extension
