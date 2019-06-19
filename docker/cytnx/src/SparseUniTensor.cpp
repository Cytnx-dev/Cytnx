#include "UniTensor.hpp"

namespace cytnx{


    void SparseUniTensor::Init(const std::vector<Bond> &bonds, const std::vector<cytnx_int64> &in_labels, const cytnx_int64 &Rowrank, const unsigned int &dtype,const int &device, const bool &is_diag){
        /*
                //check Symmetry for all bonds
                for(cytnx_uint64 i=0;i<bonds.size();i++){
                    //check 
                    cytnx_error_msg(bonds[i].qnums().size()!=0,"%s","[ERROR][SparseUniTensor] All bonds must have non symmetries.");
                }

                
                //check labels:
                if(in_labels.size()==0){
                    for(cytnx_int64 i=0;i<bonds.size();i++)
                        this->_labels.push_back(i);    
                }else{
                    //check bonds & labels dim                 
                    cytnx_error_msg(bonds.size()!=in_labels.size(),"%s","[ERROR] labels must have same lenth as # of bonds.");
                    std::vector<cytnx_uint64> tmp = vec_unique(in_labels);
                    cytnx_error_msg(tmp.size()!=in_labels,"[ERROR] labels cannot contain duplicated elements.%s","\n");
                    this->_labels = in_labels;
                }

                // device, dtype
                this->_device = device;
                this->_dtype = dtype;

                //non symmetry
                std::vector<cytnx_uint64> _shape(bonds.size());
                for(unsigned int i=0;i<_shape.size();i++)
                    _shape[i] = bonds[i].dim();
                this->_block = Tensor(_shape,dtype,device);
          */                      
    }





}

