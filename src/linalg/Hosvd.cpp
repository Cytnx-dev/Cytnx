#include "linalg.hpp"
#include "linalg_internal_interface.hpp"

#include <iostream>
#include <vector>

using namespace std;
namespace cytnx{
    namespace linalg{
         
        std::vector<Tensor> Hosvd(const Tensor &Tin, const std::vector<cytnx_uint64> &mode, const bool &is_core, const bool &is_Ls, const std::vector<cytnx_int64> &truncate_dim){
            
            //cytnx_error_msg(Tin.shape().size() != 2,"[Svd] error, Svd can only operate on rank-2 Tensor.%s","\n");
            //cytnx_error_msg(!Tin.is_contiguous(), "[Svd] error tensor must be contiguous. Call Contiguous_() or Contiguous() first%s","\n");
            cytnx_error_msg(true,"[Developing]%s","\n");
            return std::vector<Tensor>();

            

        }




    }//linalg namespace

}//cytnx namespace

namespace cytnx_extension{
    namespace xlinalg{
        using namespace cytnx;
        std::vector<cytnx_extension::CyTensor> Hosvd(const cytnx_extension::CyTensor &Tin, const std::vector<cytnx_uint64> &mode, const bool &is_core, const bool &is_Ls,const std::vector<cytnx_int64> &truncate_dim){


            //checking mode:
            cytnx_error_msg(mode.size()<2,"[Hosvd] error mode must be at least 2 element.%s","\n");
            if(truncate_dim.size()!=0){
                cytnx_error_msg(truncate_dim.size()!=mode.size(),"[Hosve] true, truncate_dim are given but len(truncate_dim) != len(mode)%s","\n");
            }

            cytnx_uint64 tot = 0;
            for(int i=0;i<mode.size();i++){
                cytnx_error_msg(mode[i]==0,"[Hosvd][ERROR], mode cannot have element=0%s","\n");
                tot += mode[i];
            }
            cytnx_error_msg(tot!=Tin.rank(),"[Hosvd] error the total sum of elements in mode should be equal to the rank of Tin.%s","\n");

            CyTensor in;
            if(!Tin.is_contiguous())
                in = Tin.contiguous();
            else
                in = Tin;

            std::vector<CyTensor> out;
            std::vector<CyTensor> Ls;

            if(Tin.is_tag()){
                cytnx_error_msg(true,"[ERROR][Hosvd] currently can only support regular CyTensor without tagged.%s","\n");
            }else{
                std::vector<cytnx_int64> perm;
                cytnx_uint64 oldRowrank = in.Rowrank();

                for(int i=0;i<mode.size();i++){
                    in.set_Rowrank(mode[i]);
                
                    if(truncate_dim.size()!=0){
                        auto tsvdout = Svd_truncate(in,truncate_dim[i],true,false);
                        out.push_back(tsvdout[1]);
                        if(is_Ls)
                            Ls.push_back(tsvdout[0]);
                    }else{
                        auto tsvdout = Svd(in,true,false);
                        out.push_back(tsvdout[1]);
                        if(is_Ls)
                            Ls.push_back(tsvdout[0]);
                    }
                    out.back().set_label(out.back().rank()-1,out.back().labels().back()-i);
                    perm.clear();                    
                    for(int j=mode[i];j<in.rank();j++)
                        perm.push_back(j);
                    for(int j=0;j<mode[i];j++){
                        perm.push_back(j);
                    }
                    in.permute_(perm);
                    in.contiguous_();

                }      
                in.set_Rowrank(oldRowrank);

                if(is_core){
                    CyTensor s = Contract(in,out[0]);
                    for(int i=1;i<out.size();i++){
                        s = Contract(s,out[i]);
                    }
                    out.push_back(s);
                }
     
                if(is_Ls){
                    for(int i=0;i<Ls.size();i++)
                        out.push_back(Ls[i]);
                }
             
            }
            return out;

        }//Hosvd


    }//linalg namespace
}//cytnx_extension namespace


