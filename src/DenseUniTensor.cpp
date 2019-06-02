#include "UniTensor.hpp"
#include "utils/utils.hpp"
namespace cytnx{


    void DenseUniTensor::Init(const std::vector<Bond> &bonds, const std::vector<cytnx_int64> &in_labels, const cytnx_int64 &Rowrank, const unsigned int &dtype,const int &device, const bool &is_diag){

                //check for all bonds
                bool is_tag =false;
                cytnx_uint32 N_ket = 0;
                if(bonds.size()!=0) is_tag = (bonds[0].type() != bondType::BD_REG);
                for(cytnx_uint64 i=0;i<bonds.size();i++){
                    //check 
                    cytnx_error_msg(bonds[i].qnums().size()!=0,"%s","[ERROR][DenseUniTensor] All bonds must have non symmetries.");
                    if(is_tag){
                        cytnx_error_msg(bonds[i].type() == bondType::BD_REG,"%s","[ERROR][DenseUniTensor] cannot mix tagged bond with un-tagged bond!%s","\n");
                        N_ket += cytnx_uint32(bonds[i].type() == bondType::BD_KET);
                    }else{
                        cytnx_error_msg(bonds[i].type() != bondType::BD_REG,"%s","[ERROR][DenseUniTensor] cannot mix tagged bond with un-tagged bond!%s","\n");
                    }

                }
                
                //check Rowrank
                if(is_tag){
                    if(Rowrank < 0){this->_Rowrank = N_ket;}
                    else{
                        cytnx_error_msg(Rowrank >= bonds.size(),"[ERROR] Rowrank cannot exceed total rank of Tensor.%s","\n");
                        this->_Rowrank = Rowrank;
                    }
                }else{ 
                    if(bonds.size()==0) this->_Rowrank = 0;    
                    else{
                        cytnx_error_msg(Rowrank <0, "[ERROR] initialize a non-symmetry, un-tagged tensor should assign a >=0 Rowrank.%s","\n");
                        cytnx_error_msg(Rowrank >= bonds.size(),"[ERROR] Rowrank cannot exceed total rank of Tensor.%s","\n");
                        this->_Rowrank = Rowrank;
                    }
                }
                                
                //check labels:
                if(in_labels.size()==0){
                    for(cytnx_int64 i=0;i<bonds.size();i++)
                        this->_labels.push_back(i);
    
                }else{
                    //check bonds & labels dim                 
                    cytnx_error_msg(bonds.size()!=in_labels.size(),"%s","[ERROR] labels must have same lenth as # of bonds.");

                    std::vector<cytnx_int64> tmp = vec_unique(in_labels);
                    cytnx_error_msg(tmp.size()!=in_labels.size(),"[ERROR] labels cannot contain duplicated elements.%s","\n");
                    this->_labels = in_labels;
                }

                if(is_diag){
                    cytnx_error_msg(bonds.size()!=2,"[ERROR] is_diag= ture should have the shape for initializing the UniTensor is square, 2-rank tensor.%s","\n");
                    cytnx_error_msg(bonds[0].dim() != bonds[1].dim(),"[ERROR] is_diag= ture should have the shape for initializing the UniTensor is square, 2-rank tensor.%s","\n");
                }


                //copy bonds, otherwise it will share objects:
                this->_bonds = vec_clone(bonds);
                this->_braket = this->_update_braket();


                //non symmetry, initialize memory.
                if(this->_bonds.size()==0){
                    //scalar:
                    this->_block = Tensor({1},dtype,device);
                }else{
                    if(is_diag){
                        this->_block = Tensor({_bonds[0].dim()},dtype,device);
                        this->_is_diag = is_diag;
                    }else{
                        std::vector<cytnx_uint64> _shape(bonds.size());
                        for(unsigned int i=0;i<_shape.size();i++)
                            _shape[i] = bonds[i].dim();

                        this->_block = Tensor(_shape,dtype,device);
                    }
                }          
    }





}

