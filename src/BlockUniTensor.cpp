#include "UniTensor.hpp"
#include "Accessor.hpp"
#include "utils/utils.hpp"
#include "utils/utils_internal_interface.hpp"
#include "linalg.hpp"
#include "Generator.hpp"
#include <vector>
#include "utils/vec_print.hpp"
#include "utils/vec_concatenate.hpp"
#include <map>
#include <stack>
#ifdef UNI_OMP
  #include <omp.h>
#endif

using namespace std;
namespace cytnx {
  typedef Accessor ac;
  void BlockUniTensor::Init(const std::vector<Bond> &bonds, const std::vector<string> &in_labels,
                             const cytnx_int64 &rowrank, const unsigned int &dtype,
                             const int &device, const bool &is_diag, const bool &no_alloc) {
    // the entering is already check all the bonds have symmetry.
    //  need to check:
    //  1. the # of symmetry and their type across all bonds
    //  2. check if all bonds are non regular:

    // check Symmetry for all bonds
    cytnx_uint32 N_symmetry = bonds[0].Nsym();
    vector<Symmetry> tmpSyms = bonds[0].syms();

    cytnx_uint32 N_ket = 0;
    for (cytnx_uint64 i = 0; i < bonds.size(); i++) {
      // check
      cytnx_error_msg(
        bonds[i].type() == BD_REG,
        "[ERROR][BlockUniTensor] All bonds must be tagged for UniTensor with symmetries.%s", "\n");


      cytnx_error_msg(
        bonds[i]._impl->_degs.size() == 0,
        "[ERROR][BlockUniTensor] All bonds must be in new format for BlockUniTensor!.%s", "\n");

      // check rank-0 bond:
      cytnx_error_msg(bonds[i].dim() == 0,
                      "[ERROR][BlockUniTensor] All bonds must have dimension >=1%s", "\n");
      // check symmetry and type:
      cytnx_error_msg(bonds[i].Nsym() != N_symmetry,
                      "[ERROR][BlockUniTensor] inconsistant # of symmetry at bond: %d. # of "
                      "symmetry should be %d\n",
                      i, N_symmetry);
      for (cytnx_uint32 n = 0; n < N_symmetry; n++) {
        cytnx_error_msg(bonds[i].syms()[n] != tmpSyms[n],
                        "[ERROR][BlockUniTensor] symmetry mismatch at bond: %d, %s != %s\n", n,
                        bonds[i].syms()[n].stype_str().c_str(), tmpSyms[n].stype_str().c_str());
      }
      N_ket += cytnx_uint32(bonds[i].type() == bondType::BD_KET);
    }

    // check rowrank:
    cytnx_error_msg((N_ket < 1) || (N_ket > bonds.size() - 1),
                    "[ERROR][BlockUniTensor] must have at least one ket-bond and one bra-bond.%s",
                    "\n");


    if (rowrank < 0) {
      this->_rowrank = N_ket;
      //this->_inner_rowrank = N_ket;
    } else {
      cytnx_error_msg((rowrank < 1) || (rowrank > bonds.size() - 1),
                      "[ERROR][BlockUniTensor] rowrank must be >=1 and <=rank-1.%s", "\n");
      this->_rowrank = rowrank;
      //this->_inner_rowrank = rowrank;
      // update braket_form >>>
    }


    // check labels:
    if (in_labels.size() == 0) {
      for (cytnx_int64 i = 0; i < bonds.size(); i++) this->_labels.push_back(to_string(i));

    } else {
      // check bonds & labels dim
      cytnx_error_msg(bonds.size() != in_labels.size(), "%s",
                      "[ERROR] labels must have same lenth as # of bonds.");

      std::vector<string> tmp = vec_unique(in_labels);
      cytnx_error_msg(tmp.size() != in_labels.size(),
                      "[ERROR] labels cannot contain duplicated elements.%s", "\n");
      this->_labels = in_labels;
    }

    cytnx_error_msg(is_diag,"[ERROR][BlockUniTensor] Cannot set is_diag=true when the UniTensor is with symmetry.%s","\n");
    this->_is_diag = is_diag;

    // copy bonds, otherwise it will share objects:
    this->_bonds = vec_clone(bonds);
    this->_is_braket_form = this->_update_braket();

    
    // checking how many blocks are there, and the size:
    std::vector<cytnx_uint64> Loc(this->_bonds.size(),0);
    std::vector<cytnx_int64> tot_qns(this->_bonds[0].Nsym()); // use first bond to determine symmetry size
    
    std::vector<cytnx_uint64> size(this->_bonds.size());
    bool fin=false;
    while(1){
       
        //get elem
        //cytnx::vec_print(std::cout , Loc);
        this->_fx_get_total_fluxs(Loc, this->_bonds[0].syms(),tot_qns);

        //std::cout << "Loc: ";
        //cytnx::vec_print_simple(std::cout, Loc);    
        //std::cout << "tot_flx: ";
        //cytnx::vec_print_simple(std::cout, tot_qns);
                
        //if exists:
        if( std::all_of(tot_qns.begin(),tot_qns.end(), [](const int &i){return i==0;}) ){
            //get size & init block!
            for(cytnx_int32 i=0;i<Loc.size();i++){
                size[i] = this->_bonds[i]._impl->_degs[Loc[i]];
            }
            this->_blocks.push_back(zeros(size,dtype,device));

            // push its loc
            this->_inner_to_outer_idx.push_back(Loc);

        }



        while(Loc.size()!=0){
            Loc.back()+=1;
            if(Loc.back()==this->_bonds[Loc.size()-1]._impl->_qnums.size()){Loc.pop_back();}
            else{
                for(int i=0;i<this->_bonds.size()-Loc.size();i++){
                    Loc.push_back(0);
                }
                break;
            }
        }
            
        if(Loc.size()==0) break;
        

    }
    
      
  }

  void BlockUniTensor::print_blocks(const bool &full_info) const{
    std::ostream &os = std::cout;

    os << "-------- start of print ---------\n";
    char *buffer = (char *)malloc(sizeof(char) * 1024);
    sprintf(buffer, "Tensor name: %s\n", this->_name.c_str());
    os << std::string(buffer);
    if (this->_is_tag) sprintf(buffer, "braket_form : %s\n", this->_is_braket_form ? "True" : "False");
    os << std::string(buffer);
    sprintf(buffer, "is_diag    : %s\n", this->_is_diag ? "True" : "False");
    os << std::string(buffer);
    sprintf(buffer, "[OVERALL] contiguous : %s\n", this->is_contiguous() ? "True" : "False");
    os << std::string(buffer);

    /*
    os << "Symmetries: ";
    for(int s=0;s<this->_bonds[0].Nsym();s++)
        os << this->_bonds[0]._impl->_syms[s].stype_str() << " ";
    os << endl;
    */

    // print each blocks with its qnum! 
    for(int b=0;b<this->_blocks.size();b++){
        os << "========================\n";
        os << "BLOCK [#" << b << "]\n"; 
        os << "Qn for each axis:\n";
        for(int s=0;s<this->_bonds[0].Nsym();s++){
            os << this->_bonds[0]._impl->_syms[s].stype_str() << ": "; 
            for(int l=0;l<this->_blocks[b].shape().size();l++){
                os << this->_bonds[l]._impl->_qnums[this->_inner_to_outer_idx[b][l]][s] << "\t";
            } 
            os << endl; 
        }
        if(full_info)
            os << this->_blocks[b];
        else{
            os << "dtype: " << Type.getname(this->_blocks[b].dtype()) << endl;
            os << "device: " << Device.getname(this->_blocks[b].device()) << endl;
            os << "contiguous: " << this->_blocks[b].is_contiguous() << endl;
            os << "shape: ";
            vec_print_simple(os,this->_blocks[b].shape());

        }
    }

    /*
      auto tmp_qnums = in.get_blocks_qnums();
      std::vector<Tensor> tmp = in.get_blocks_(true);
      sprintf(buffer, "BLOCKS:: %s", "\n");
      os << std::string(buffer);
      os << "=============\n";

      if (!in.is_contiguous()) {
        cytnx_warning_msg(
          true,
          "[WARNING][Symmetric] cout/print UniTensor on a non-contiguous UniTensor. the blocks "
          "appears here could be different than the current shape of UniTensor.%s",
          "\n");
      }
      for (cytnx_uint64 i = 0; i < tmp.size(); i++) {
        os << "Qnum:" << tmp_qnums[i] << std::endl;
        os << tmp[i] << std::endl;
        os << "=============\n";
      }
      os << "-------- end of print ---------\n";
    */
    free(buffer);
  }

  void BlockUniTensor::print_diagram(const bool &bond_info) {
    char *buffer = (char *)malloc(1024 * sizeof(char));
    unsigned int BUFFsize = 100;

    sprintf(buffer, "-----------------------%s", "\n");
    std::cout << std::string(buffer);
    sprintf(buffer, "tensor Name : %s\n", this->_name.c_str());
    std::cout << std::string(buffer);
    sprintf(buffer, "tensor Rank : %d\n", this->_labels.size());
    std::cout << std::string(buffer);
    //sprintf(buffer, "block_form  : true%s", "\n");
    //std::cout << std::string(buffer);
    sprintf(buffer, "contiguous  : %s\n", this->is_contiguous() ? "True" : "False");
    std::cout << std::string(buffer);
    sprintf(buffer, "valid bocks : %d\n", this->_blocks.size());
    std::cout << std::string(buffer);
    //sprintf(buffer, "is diag   : %s\n", this->is_diag() ? "True" : "False");
    //std::cout << std::string(buffer);
    sprintf(buffer, "on device   : %s\n", this->device_str().c_str());
    std::cout << std::string(buffer);

    cytnx_uint64 Nin = this->_rowrank;
    cytnx_uint64 Nout = this->_labels.size() - this->_rowrank;
    cytnx_uint64 vl;
    if (Nin > Nout)
      vl = Nin;
    else
      vl = Nout;

    std::string bks;
    char *l = (char *)malloc(BUFFsize * sizeof(char));
    char *llbl = (char *)malloc(BUFFsize * sizeof(char));
    char *r = (char *)malloc(BUFFsize * sizeof(char));
    char *rlbl = (char *)malloc(BUFFsize * sizeof(char));

    int Space_Llabel_max=0, Space_Ldim_max=0, Space_Rdim_max =0;
    //quickly checking the size for each line, only check the largest! 
    
    for (cytnx_uint64 i = 0; i < vl; i++) {
        if(i<Nin){
            if(Space_Llabel_max < this->_labels[i].size()) Space_Llabel_max = this->_labels[i].size();
            if(Space_Ldim_max < to_string(this->_bonds[i].dim()).size()) Space_Ldim_max = to_string(this->_bonds[i].dim()).size();
        }
        if(i<Nout){
            if(Space_Rdim_max < to_string(this->_bonds[Nin+i].dim()).size()) Space_Rdim_max = to_string(this->_bonds[Nin+i].dim()).size();
        }
    }
    string LallSpace = (string(" ")*(Space_Llabel_max+3+1));
    string MallSpace = string(" ")*(1 + Space_Ldim_max + 5 + Space_Rdim_max+1);
    string M_dashes  = string("-")*(1 + Space_Ldim_max + 5 + Space_Rdim_max+1);
    
    std::string tmpss;
    sprintf(buffer, "%s row %s col %s",LallSpace.c_str(),MallSpace.c_str(),"\n");
    std::cout << std::string(buffer);
    sprintf(buffer, "%s    -%s-    %s",LallSpace.c_str(),M_dashes.c_str(),"\n");
    std::cout << std::string(buffer);
    for (cytnx_uint64 i = 0; i < vl; i++) {
      sprintf(buffer, "%s    |%s|    %s",LallSpace.c_str(),MallSpace.c_str(),"\n");
      std::cout << std::string(buffer);

      if (i < Nin) {
        if (this->_bonds[i].type() == bondType::BD_KET)
          bks = " -->";
        else
          bks = "*<--";
        memset(l, 0, sizeof(char) * BUFFsize);
        memset(llbl, 0, sizeof(char) * BUFFsize);
        tmpss = this->_labels[i] + std::string(" ")*(Space_Llabel_max-this->_labels[i].size());
        sprintf(l, "%s %s", tmpss.c_str(), bks.c_str());
        tmpss = to_string(this->_bonds[i].dim()) + std::string(" ")*(Space_Ldim_max-to_string(this->_bonds[i].dim()).size());
        sprintf(llbl, "%s", tmpss.c_str());
      } else {
        memset(l, 0, sizeof(char) * BUFFsize);
        memset(llbl, 0, sizeof(char) * BUFFsize);
        tmpss = std::string(" ")*(Space_Llabel_max+5);
        sprintf(l, "%s",tmpss.c_str());
        tmpss = std::string(" ")*(Space_Ldim_max);
        sprintf(llbl, "%s",tmpss.c_str());
      }
      if (i < Nout) {
        if (this->_bonds[Nin + i].type() == bondType::BD_KET)
          bks = "<--*";
        else
          bks = "--> ";
        memset(r, 0, sizeof(char) * BUFFsize);
        memset(rlbl, 0, sizeof(char) * BUFFsize);
        
        sprintf(r, "%s %s", bks.c_str(), this->_labels[Nin + i].c_str());
        
        tmpss = to_string(this->_bonds[Nin+i].dim()) + std::string(" ")*(Space_Rdim_max-to_string(this->_bonds[Nin+i].dim()).size());
        sprintf(rlbl, "%s", tmpss.c_str());

      } else {
        memset(r, 0, sizeof(char) * BUFFsize);
        memset(rlbl, 0, sizeof(char) * BUFFsize);
        sprintf(r, "%s", "        ");
        tmpss = std::string(" ")*Space_Rdim_max;
        sprintf(rlbl, "%s",tmpss.c_str());
      }
      sprintf(buffer, "   %s| %s     %s |%s\n", l, llbl, rlbl, r);
      std::cout << std::string(buffer);
    }
    sprintf(buffer, "%s    |%s|    %s",LallSpace.c_str(),MallSpace.c_str(),"\n");
    std::cout << std::string(buffer);
    sprintf(buffer, "%s    -%s-    %s",LallSpace.c_str(),M_dashes.c_str(),"\n");
    std::cout << std::string(buffer);
    sprintf(buffer, "%s", "\n");
    std::cout << std::string(buffer);

    if (bond_info) {
      for (cytnx_uint64 i = 0; i < this->_bonds.size(); i++) {
        // sprintf(buffer, "lbl:%d ", this->_labels[i]);
        sprintf(buffer, "lbl:%s ", this->_labels[i].c_str());
        std::cout << std::string(buffer);
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



}  // namespace cytnx
