#include "UniTensor.hpp"
#include "utils/utils.hpp"
#include "linalg/linalg.hpp"
#include "Generator.hpp"
#include <vector>
using namespace std;
namespace cytnx{


    void SparseUniTensor::Init(const std::vector<Bond> &bonds, const std::vector<cytnx_int64> &in_labels, const cytnx_int64 &Rowrank, const unsigned int &dtype,const int &device, const bool &is_diag){
        //the entering is already check all the bonds have symmetry.
        // need to check:
        // 1. the # of symmetry and their type across all bonds
        // 2. check if all bonds are non regular:

        //check Symmetry for all bonds
        cytnx_uint32 N_symmetry = bonds[0].Nsym();
        vector<Symmetry> tmpSyms = bonds[0]._syms_by_ref();
        cytnx_uint32 N_ket = 0;
        for(cytnx_uint64 i=0;i<bonds.size();i++){
            //check 
            cytnx_error_msg(bonds[i].type()==BD_REG,"[ERROR][SparseUniTensor] All bonds must be tagged for UniTensor with symmetries.%s","\n");
            //check rank-0 bond:
            cytnx_error_msg(bonds[i].dim()==0,"[ERROR][SparseUniTensor] All bonds must have dimension >=1%s","\n");
            //check symmetry and type:
            cytnx_error_msg(bonds[i].Nsym() != N_symmetry,"[ERROR][SparseUniTensor] inconsistant # of symmetry at bond: %d. # of symmetry should be %d\n",i,N_symmetry);
            for(cytnx_uint32 n=0;n<N_symmetry;n++){
                cytnx_error_msg(bonds[i]._syms_by_ref()[n] !=tmpSyms[n],"[ERROR][SparseUniTensor] symmetry mismatch at bond: %d, %s != %s\n",n,bonds[i]._syms_by_ref()[n].stype_str().c_str(),tmpSyms[n].stype_str().c_str());
            }
            N_ket += cytnx_uint32(bonds[i].type() == bondType::BD_KET);
        }
        
        //check Rowrank:
        cytnx_error_msg((N_ket<1)||(N_ket>bonds.size()-1),"[ERROR][SparseUniTensor] must have at least one ket-bond and one bra-bond.%s","\n");
        
        if(Rowrank<0){this->_Rowrank = N_ket;}
        else{
            cytnx_error_msg((Rowrank<1) || (Rowrank>bonds.size()-1),"[ERROR][SparseUniTensor] Rowrank must be >=1 and <=rank-1.%s","\n");
            this->_Rowrank = Rowrank;
            // update braket_form >>>
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
        cytnx_error_msg(is_diag,"[ERROR][SparseUniTensor] Cannot set is_diag=true when the UniTensor is with symmetry.%s","\n");

        //copy bonds, otherwise it will share objects:
        this->_bonds = vec_clone(bonds);
        this->_is_braket_form = this->_update_braket();

        //Symmetry, initialize memories for blocks.
                


    }





}

