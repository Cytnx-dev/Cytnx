#include "UniTensor.hpp"
#include "Accessor.hpp"
#include "utils/utils.hpp"
#include "utils/utils_internal_interface.hpp"
#include "linalg.hpp"
#include "Generator.hpp"
#include <vector>
#include "utils/vec_print.hpp"
using namespace std;
namespace cytnx{
    typedef Accessor ac;
    void AnyonUniTensor::Init(const std::vector<Bond> &bonds, const std::vector<cytnx_int64> &in_labels, const cytnx_int64 &rowrank, const unsigned int &dtype,const int &device, const bool &is_diag){
            cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");

    }


    vector<Bond> AnyonUniTensor::getTotalQnums(const bool &physical){
        
        cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");
        

    }
    void AnyonUniTensor::permute_(const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank,const bool &by_label){
        cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");

    }

    void AnyonUniTensor::print_diagram(const bool &bond_info){
        char *buffer = (char*)malloc(256*sizeof(char));

        sprintf(buffer,"-----------------------%s","\n");
        sprintf(buffer,"tensor Name : %s\n",this->_name.c_str());       std::cout << std::string(buffer);
        sprintf(buffer,"tensor Rank : %d\n",this->_labels.size());      std::cout << std::string(buffer);
        sprintf(buffer,"block_form  : true%s","\n");                    std::cout << std::string(buffer);
        sprintf(buffer,"valid bocks : %d\n",this->_blocks.size());      std::cout << std::string(buffer);
        sprintf(buffer,"on device   : %s\n",this->device_str().c_str());std::cout << std::string(buffer);

        cytnx_uint64 Nin = this->_rowrank;
        cytnx_uint64 Nout = this->_labels.size() - this->_rowrank;
        cytnx_uint64 vl;
        if(Nin > Nout) vl = Nin;
        else           vl = Nout;

        std::string bks;
        char *l = (char*)malloc(40*sizeof(char));
        char *llbl = (char*)malloc(40*sizeof(char));
        char *r = (char*)malloc(40*sizeof(char));
        char *rlbl = (char*)malloc(40*sizeof(char));
        
        sprintf(buffer,"braket_form : %s\n",this->_is_braket_form?"True":"False"); std::cout << std::string(buffer);
        sprintf(buffer,"        row               col %s","\n");                 std::cout << std::string(buffer);
        sprintf(buffer,"           x-------------x      %s","\n");                 std::cout << std::string(buffer);
        for(cytnx_uint64 i=0;i<vl;i++){
            sprintf(buffer,"           |             |     %s","\n"); std::cout << std::string(buffer);
            if(i<Nin){
                if(this->_bonds[i].type() == bondType::BD_KET) bks = " -->";
                else                                         bks = "*<--";
                memset(l,0,sizeof(char)*40);
                memset(llbl,0,sizeof(char)*40);
                sprintf(l,"%3d %s",this->_labels[i],bks.c_str());
                sprintf(llbl,"%-3d",this->_bonds[i].dim());
            }else{
                memset(l,0,sizeof(char)*40);
                memset(llbl,0,sizeof(char)*40);
                sprintf(l,"%s","        ");
                sprintf(llbl,"%s","   ");
            }
            if(i< Nout){
                if(this->_bonds[Nin+i].type() == bondType::BD_KET) bks = "<--*";
                else                                              bks = "--> ";
                memset(r,0,sizeof(char)*40);
                memset(rlbl,0,sizeof(char)*40);
                sprintf(r,"%s %-3d",bks.c_str(),this->_labels[Nin + i]);
                sprintf(rlbl,"%3d",this->_bonds[Nin + i].dim());
            }else{
                memset(r,0,sizeof(char)*40);
                memset(rlbl,0,sizeof(char)*40);
                sprintf(r,"%s","        ");
                sprintf(rlbl,"%s","   ");
            }
            sprintf(buffer,"   %s| %s     %s |%s\n",l,llbl,rlbl,r); std::cout << std::string(buffer);

        }
        sprintf(buffer,"           |             |     %s","\n"); std::cout << std::string(buffer);
        sprintf(buffer,"           x-------------x     %s","\n"); std::cout << std::string(buffer);


        if(bond_info){
            for(cytnx_uint64 i=0; i< this->_bonds.size();i++){
                sprintf(buffer,"lbl:%d ",this->_labels[i]); std::cout << std::string(buffer);
                std::cout << this->_bonds[i] << std::endl;
            }
        }

        fflush(stdout);
        free(l);
        free(llbl);
        free(r);
        free(rlbl);
        free(buffer);
    }

    boost::intrusive_ptr<UniTensor_base> AnyonUniTensor::contiguous(){
        cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");
        
    }
//=======================================================================
// at_for_sparse;
//=======================================================================
    

    cytnx_complex128& AnyonUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_complex128 &aux){
        cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");
        
    }
    const cytnx_complex128& AnyonUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_complex128 &aux) const{
        cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");
    }

    //-----------------------------------------
    cytnx_complex64&  AnyonUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_complex64 &aux){
        cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");
    }
    const cytnx_complex64&  AnyonUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_complex64 &aux) const{
        cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");
    }
    //-------------------------------------
    cytnx_double&     AnyonUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_double &aux){
        cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");
    }
    const cytnx_double&     AnyonUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_double &aux) const{
        cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");
    }
    //--------------------------------------
    cytnx_float&      AnyonUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_float &aux){

        cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");
    }
    const cytnx_float&      AnyonUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_float &aux) const{
        cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");
    }
    //--------------------------------------
    cytnx_uint64&      AnyonUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint64 &aux){
        cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");
    }
    const cytnx_uint64&      AnyonUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint64 &aux) const{
        cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");
    }
    //--------------------------------------
    cytnx_int64&      AnyonUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int64 &aux){
        cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");
    }
    const cytnx_int64&      AnyonUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int64 &aux) const{
        cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");
    }
    //--------------------------------------
    cytnx_uint32&      AnyonUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint32 &aux){
        cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");
    }
    const cytnx_uint32&      AnyonUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint32 &aux) const{
        cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");
    }
    //--------------------------------------
    cytnx_int32&      AnyonUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int32 &aux){
        cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");
    }
    const cytnx_int32&      AnyonUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int32 &aux) const{
        cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");
    }
    //--------------------------------------
    cytnx_uint16&      AnyonUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint16 &aux){
        cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");
    }
    const cytnx_uint16&      AnyonUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint16 &aux) const{
        cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");
    }
    //--------------------------------------
    cytnx_int16&      AnyonUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int16 &aux){
        cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");
    }
    const cytnx_int16&      AnyonUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int16 &aux) const{
        cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");
    }
    //================================
    bool AnyonUniTensor::elem_exists(const std::vector<cytnx_uint64> &locator) const{
        cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");
    }


    void AnyonUniTensor::Transpose_(){
        cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");
    }

    boost::intrusive_ptr<UniTensor_base> AnyonUniTensor::contract(const boost::intrusive_ptr<UniTensor_base> &rhs){
        cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");
        
    }


    void AnyonUniTensor::truncate_(const cytnx_int64 &bond_idx, const cytnx_uint64 &dim, const bool &by_label){
        cytnx_error_msg(true,"[ERROR] truncate for AnyonUniTensor is under developing!!%s","\n");
    }

    void AnyonUniTensor::_save_dispatch(std::fstream &f) const{
        cytnx_error_msg(true,"[ERROR] Save for AnyonUniTensor is under developing!!%s","\n");
    }
    void AnyonUniTensor::_load_dispatch(std::fstream &f){
        cytnx_error_msg(true,"[ERROR] Save for AnyonUniTensor is under developing!!%s","\n");
    }



}//namespace cytnx
