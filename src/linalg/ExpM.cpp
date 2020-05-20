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
            ut = Inv(su[1]); 
        
            ut = cytnx::linalg::Matmul(s,ut);
            ut = cytnx::linalg::Matmul(u,ut);

            return ut;
                         
        }




    }//linalg namespace

}//cytnx namespace



