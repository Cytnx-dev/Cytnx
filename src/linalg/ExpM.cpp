#include "linalg.hpp"
#include "linalg_internal_interface.hpp"
#include "Generator.hpp"

#include <iostream>
#include <vector>

using namespace std;
namespace cytnx{
    namespace linalg{

                 
        Tensor ExpM(const Tensor &Tin, const cytnx_double &a, const cytnx_double &b){
            
            cytnx_error_msg(Tin.shape().size() != 2,"[ExpH] error, ExpH can only operate on rank-2 Tensor.%s","\n");
            cytnx_error_msg(!Tin.is_contiguous(), "[ExpH] error tensor must be contiguous. Call Contiguous_() or Contiguous() first%s","\n");
            
            
            cytnx_error_msg(Tin.shape()[0] != Tin.shape()[1],"[ExpH] error, ExpM can only operator on square Tensor (#row = #col%s","\n");


            vector<Tensor> su = cytnx::linalg::Eig(Tin,true);
            //cout << su[0] << su[1] << endl;
            Tensor s,u, ut;
            if(a==0)
                return cytnx::linalg::Diag(cytnx::ones(Tin.shape()[0]));

            if(b==0)
                s = cytnx::linalg::Exp(a*su[0]);
            else
                s = cytnx::linalg::Exp(a*su[0]+b);

            u = su[1];
            su.clear();

         
            //[Optim required]
            //cout << s << endl;
            s = cytnx::linalg::Diag(s);
            //cout << s << endl;
            //cout << u;
            ut = InvM(su[1]); 
        
            ut = cytnx::linalg::Matmul(s,ut);
            ut = cytnx::linalg::Matmul(u,ut);

            return ut;
                         
        }




    }//linalg namespace

}//cytnx namespace

namespace cytnx_extension{
    namespace xlinalg{
        CyTensor ExpM(const CyTensor &Tin, const cytnx_double &a, const cytnx_double &b){
            
            if(Tin.is_blockform()){
                cytnx_error_msg(true,"[Developing]%s","\n");
                


            }else{
                
                cytnx_error_msg((Tin.Rowrank()==0) || (Tin.Rowrank()==Tin.rank()),"[ERROR][ExpM] The total dimension of row-space and col-space should be equal!!%s","\n");
                //cytnx_error_msg(Tin.is_contiguous()==false,"[ERROR][ExpH] CyTensor must be contiguous. Call Contiguous_() or Contiguous() first%s","\n");
                //CyTensor out = Tin.clone();
                CyTensor out;
                if(Tin.is_contiguous()){
                    out = Tin.clone();
                }else{
                    out = Tin.contiguous();
                }
                cytnx_int64 Drow=1,Dcol=1;
                for(int i=0;i<Tin.Rowrank();i++){
                    Drow*=Tin.shape()[i];
                }
                for(int i=Tin.Rowrank();i<Tin.rank();i++){
                    Dcol*=Tin.shape()[i];
                }
                cytnx_error_msg(Drow!=Dcol,"[ERROR][ExpM] The total dimension of row-space and col-space should be equal!!%s","\n");
               
                out.get_block_().reshape_({Drow,Dcol});
                
                out.get_block_() = cytnx::linalg::ExpM(out.get_block_(),a,b);

                out.get_block_().reshape_(Tin.shape());

                return out;
            }            


        }
    }
}



