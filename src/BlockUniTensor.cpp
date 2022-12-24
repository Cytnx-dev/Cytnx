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
        //cout << "start!" << endl;
        //cytnx::vec_print_simple(std::cout , Loc);
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
            if(Loc.back()==this->_bonds[Loc.size()-1]._impl->_qnums.size()-1){
                Loc.pop_back(); 
                continue;
            }
            else{
                Loc.back()+=1;
                //cout << "+1 at loc:" << Loc.size()-1 <<endl;
                while(Loc.size()!=this->_bonds.size()){
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
        os << "  |-Qn indices for each axis:\n   {\t";
        for(int s=0;s<this->_inner_to_outer_idx[b].size();s++){
            os << this->_inner_to_outer_idx[b][s] << "\t";
        }
        os << "}" << endl;
        os << "\t";
        for(int s=0;s<this->_bonds.size();s++){
            os << ((this->_bonds[s].type()>0)?"OUT":"IN") << "\t";
        }
        os << endl;
        os << "  |-Qn for each axis:\n";
        for(int s=0;s<this->_bonds[0].Nsym();s++){
            os << " " <<this->_bonds[0]._impl->_syms[s].stype_str() << ":\t"; 
            for(int l=0;l<this->_blocks[b].shape().size();l++){
                os << std::showpos << this->_bonds[l]._impl->_qnums[this->_inner_to_outer_idx[b][l]][s] << "\t";
            } 
            os << std::noshowpos << endl; 
        }
        
        if(full_info)
            os << this->_blocks[b];
        else{
            os << "  |-dtype:\t" << Type.getname(this->_blocks[b].dtype()) << endl;
            os << "  |-device:\t" << Device.getname(this->_blocks[b].device()) << endl;
            os << "  |-contiguous:\t" << (this->_blocks[b].is_contiguous()? "True" : "False") << endl;
            os << "  |-shape:\t";
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

    boost::intrusive_ptr<UniTensor_base> BlockUniTensor::contiguous() {
        if(this->is_contiguous()){
            boost::intrusive_ptr<UniTensor_base> out(this);
            return out;
        } else{
            BlockUniTensor *tmp = new BlockUniTensor();
            tmp = this->clone_meta(true,true);
            tmp->_blocks.resize(this->_blocks.size());
            for(unsigned int b=0;b<this->_blocks.size();b++){
                if(this->_blocks[b].is_contiguous()){
                    tmp->_blocks[b] = this->_blocks[b].clone();
                }else{
                    tmp->_blocks[b] = this->_blocks[b].contiguous();
                }
            }
            boost::intrusive_ptr<UniTensor_base> out(tmp);
            return out;
        }
    }

  std::vector<Symmetry> BlockUniTensor::syms() const { return this->_bonds[0].syms(); }


  boost::intrusive_ptr<UniTensor_base> BlockUniTensor::permute(
    const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank, const bool &by_label) {
    
    BlockUniTensor *out_raw = this->clone_meta(true,true);
    out_raw ->_blocks.resize(this->_blocks.size());

    std::vector<cytnx_uint64> mapper_u64;
    if (by_label) {
      // cytnx_error_msg(true,"[Developing!]%s","\n");
      std::vector<std::string>::iterator it;
      for (cytnx_uint64 i = 0; i < mapper.size(); i++) {
        it = std::find(out_raw->_labels.begin(), out_raw->_labels.end(), std::to_string(mapper[i]));
        cytnx_error_msg(it == out_raw->_labels.end(),
                        "[ERROR] label %d does not exist in current UniTensor.\n", mapper[i]);
        mapper_u64.push_back(std::distance(out_raw->_labels.begin(), it));
      }

    } else {
      mapper_u64 = std::vector<cytnx_uint64>(mapper.begin(), mapper.end());
    }


    out_raw->_bonds = vec_map(vec_clone(out_raw->bonds()), mapper_u64);  // this will check validity
    out_raw->_labels = vec_map(out_raw->labels(), mapper_u64);


    //inner_to_outer permute!
    for(cytnx_int64 b=0;b<this->_inner_to_outer_idx.size();b++){
        out_raw->_inner_to_outer_idx[b] = vec_map(out_raw->_inner_to_outer_idx[b], mapper_u64);
        out_raw->_blocks[b] = this->_blocks[b].permute(mapper_u64); 
    }

    if(out_raw->_is_diag){
        cytnx_error_msg(true,"[ERROR][BlockUniTensor] currently do not support permute for is_diag=true for BlockUniTensor!%s","\n");
    }else{
        if(rowrank >=0){
            cytnx_error_msg((rowrank >= out_raw->_bonds.size()) || (rowrank < 1),
                            "[ERROR][BlockUniTensor] rowrank cannot exceed the rank of UniTensor-1, and should be >=1.%s",
                            "\n");
            out_raw->_rowrank = rowrank;

        }
        out_raw->_is_braket_form = out_raw->_update_braket();
    }
    boost::intrusive_ptr<UniTensor_base> out(out_raw);

    return out;
  }

  boost::intrusive_ptr<UniTensor_base> BlockUniTensor::permute(
    const std::vector<std::string> &mapper, const cytnx_int64 &rowrank) {

        BlockUniTensor *out_raw = this->clone_meta(true,true);
        out_raw ->_blocks.resize(this->_blocks.size());

        std::vector<cytnx_uint64> mapper_u64;
        // cytnx_error_msg(true,"[Developing!]%s","\n");
        std::vector<string>::iterator it;
        for (cytnx_uint64 i = 0; i < mapper.size(); i++) {
          it = std::find(out_raw->_labels.begin(), out_raw->_labels.end(), mapper[i]);
          cytnx_error_msg(it == out_raw->_labels.end(),
                          "[ERROR] label %s does not exist in current UniTensor.\n", mapper[i]);
          mapper_u64.push_back(std::distance(out_raw->_labels.begin(), it));
        }

    
        out_raw->_bonds = vec_map(vec_clone(out_raw->bonds()), mapper_u64);  // this will check validity
        out_raw->_labels = vec_map(out_raw->labels(), mapper_u64);


        if(out_raw->_is_diag){
            //inner_to_outer permute!
            for(cytnx_int64 b=0;b<this->_inner_to_outer_idx.size();b++){
                out_raw->_inner_to_outer_idx[b] = vec_map(out_raw->_inner_to_outer_idx[b], mapper_u64);
                //??out_raw->_blocks[b] = this->_blocks[b].permute(mapper_u64);
            }

            cytnx_error_msg(true,"[ERROR][BlockUniTensor] currently do not support permute for is_diag=true for BlockUniTensor!%s","\n");
        }else{
            //inner_to_outer permute!
            for(cytnx_int64 b=0;b<this->_inner_to_outer_idx.size();b++){
                out_raw->_inner_to_outer_idx[b] = vec_map(out_raw->_inner_to_outer_idx[b], mapper_u64);
                out_raw->_blocks[b] = this->_blocks[b].permute(mapper_u64);
            }

            if(rowrank >=0){
                cytnx_error_msg((rowrank >= out_raw->_bonds.size()) || (rowrank < 1),
                                "[ERROR][BlockUniTensor] rowrank cannot exceed the rank of UniTensor-1, and should be >=1.%s",
                                "\n");
                out_raw->_rowrank = rowrank;

            }
            out_raw->_is_braket_form = out_raw->_update_braket();
        }
        boost::intrusive_ptr<UniTensor_base> out(out_raw);

        return out;


  }

  void BlockUniTensor::permute_(const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank,
                                 const bool &by_label) {
    std::vector<cytnx_uint64> mapper_u64;
    if (by_label) {
      // cytnx_error_msg(true,"[Developing!]%s","\n");
      std::vector<string>::iterator it;
      for (cytnx_uint64 i = 0; i < mapper.size(); i++) {
        it = std::find(this->_labels.begin(), this->_labels.end(), std::to_string(mapper[i]));
        cytnx_error_msg(it == this->_labels.end(),
                        "[ERROR] label %d does not exist in current UniTensor.\n", mapper[i]);
        mapper_u64.push_back(std::distance(this->_labels.begin(), it));
      }

    } else {
      mapper_u64 = std::vector<cytnx_uint64>(mapper.begin(), mapper.end());
    }

    this->_bonds = vec_map(vec_clone(this->bonds()), mapper_u64);  // this will check validity
    this->_labels = vec_map(this->labels(), mapper_u64);

    if(this->_is_diag){

        for(cytnx_int64 b=0;b<this->_inner_to_outer_idx.size();b++){
            this->_inner_to_outer_idx[b] = vec_map(this->_inner_to_outer_idx[b], mapper_u64);
            //??out_raw->_blocks[b] = this->_blocks[b].permute(mapper_u64);
        }
        cytnx_error_msg(true,"[ERROR][BlockUniTensor] currently do not support permute for is_diag=true for BlockUniTensor!%s","\n");
    }else{
        //inner_to_outer permute!
        for(cytnx_int64 b=0;b<this->_inner_to_outer_idx.size();b++){
            this->_inner_to_outer_idx[b] = vec_map(this->_inner_to_outer_idx[b], mapper_u64);
            this->_blocks[b].permute_(mapper_u64);
        }

        if (rowrank >= 0) {
            cytnx_error_msg((rowrank >= this->_bonds.size()) || (rowrank < 1),
                                "[ERROR][BlockUniTensor] rowrank cannot exceed the rank of UniTensor-1, and should be >=1.%s",
                                "\n");
                this->_rowrank = rowrank;
        }
        this->_is_braket_form = this->_update_braket();
    }

  }

  void BlockUniTensor::permute_(const std::vector<std::string> &mapper,
                                const cytnx_int64 &rowrank) {

    std::vector<cytnx_uint64> mapper_u64;
    // cytnx_error_msg(true,"[Developing!]%s","\n");
    std::vector<std::string>::iterator it;
    for (cytnx_uint64 i = 0; i < mapper.size(); i++) {
      it = std::find(this->_labels.begin(), this->_labels.end(), mapper[i]);
      cytnx_error_msg(it == this->_labels.end(),
                      "[ERROR] label %d does not exist in current UniTensor.\n", mapper[i]);
      mapper_u64.push_back(std::distance(this->_labels.begin(), it));
    }

    this->_bonds = vec_map(vec_clone(this->bonds()), mapper_u64);  // this will check validity
    this->_labels = vec_map(this->labels(), mapper_u64);

    if(this->_is_diag){

        for(cytnx_int64 b=0;b<this->_inner_to_outer_idx.size();b++){
            this->_inner_to_outer_idx[b] = vec_map(this->_inner_to_outer_idx[b], mapper_u64);
            //??out_raw->_blocks[b] = this->_blocks[b].permute(mapper_u64);
        }
        cytnx_error_msg(true,"[ERROR][BlockUniTensor] currently do not support permute for is_diag=true for BlockUniTensor!%s","\n");
    }else{
        //inner_to_outer permute!
        for(cytnx_int64 b=0;b<this->_inner_to_outer_idx.size();b++){
            this->_inner_to_outer_idx[b] = vec_map(this->_inner_to_outer_idx[b], mapper_u64);
            this->_blocks[b].permute_(mapper_u64);
        }

        if (rowrank >= 0) {
            cytnx_error_msg((rowrank >= this->_bonds.size()) || (rowrank < 1),
                                "[ERROR][BlockUniTensor] rowrank cannot exceed the rank of UniTensor-1, and should be >=1.%s",
                                "\n");
                this->_rowrank = rowrank;
        }
        this->_is_braket_form = this->_update_braket();
    }



  }

  boost::intrusive_ptr<UniTensor_base> BlockUniTensor::relabels(
    const std::vector<string> &new_labels) {
    BlockUniTensor *tmp = this->clone_meta(true, true);
    tmp->_blocks = this->_blocks;
    tmp->set_labels(new_labels);
    boost::intrusive_ptr<UniTensor_base> out(tmp);
    return out;
  }
  boost::intrusive_ptr<UniTensor_base> BlockUniTensor::relabels(
    const std::vector<cytnx_int64> &new_labels) {
    vector<string> vs(new_labels.size());
    transform(new_labels.begin(), new_labels.end(), vs.begin(),
              [](cytnx_int64 x) -> string { return to_string(x); });
    //std::cout << "entry" << endl;
    return relabels(vs);
  }

  boost::intrusive_ptr<UniTensor_base> BlockUniTensor::relabel(const cytnx_int64 &inx,
                                                                const cytnx_int64 &new_label,
                                                                const bool &by_label) {
    BlockUniTensor *tmp = this->clone_meta(true, true);
    tmp->_blocks = this->_blocks;
    tmp->set_label(inx, new_label, by_label);
    boost::intrusive_ptr<UniTensor_base> out(tmp);
    return out;
  }
  boost::intrusive_ptr<UniTensor_base> BlockUniTensor::relabel(const cytnx_int64 &inx,
                                                                const string &new_label) {
    BlockUniTensor *tmp = this->clone_meta(true, true);
    tmp->_blocks = this->_blocks;
    tmp->set_label(inx, new_label);
    boost::intrusive_ptr<UniTensor_base> out(tmp);
    return out;
  }
  boost::intrusive_ptr<UniTensor_base> BlockUniTensor::relabel(const string &inx,
                                                                const string &new_label) {
    BlockUniTensor *tmp = this->clone_meta(true, true);
    tmp->_blocks = this->_blocks;
    tmp->set_label(inx, new_label);
    boost::intrusive_ptr<UniTensor_base> out(tmp);
    return out;
  }
  boost::intrusive_ptr<UniTensor_base> BlockUniTensor::relabel(const cytnx_int64 &inx,
                                                                const cytnx_int64 &new_label) {
    BlockUniTensor *tmp = this->clone_meta(true, true);
    tmp->_blocks = this->_blocks;
    tmp->set_label(inx, new_label);
    boost::intrusive_ptr<UniTensor_base> out(tmp);
    return out;
  }



  boost::intrusive_ptr<UniTensor_base> BlockUniTensor::contract(
      const boost::intrusive_ptr<UniTensor_base> &rhs, const bool &mv_elem_self,
      const bool &mv_elem_rhs){
    // checking type
    cytnx_error_msg(rhs->uten_type() != UTenType.Block,
                    "[ERROR] cannot contract symmetry-block UniTensor with other type of UniTensor%s",
                    "\n");

    //checking symmetry:
    cytnx_error_msg(this->syms() != rhs->syms(),
                    "[ERROR] two UniTensor have different symmetry type cannot contract.%s", "\n");


    // get common labels:
    std::vector<string> comm_labels;
    std::vector<cytnx_uint64> comm_idx1, comm_idx2;
    vec_intersect_(comm_labels, this->labels(), rhs->labels(), comm_idx1, comm_idx2);

    

    if (comm_idx1.size() == 0) {
        
        // output instance;
        BlockUniTensor *tmp = new BlockUniTensor();
        BlockUniTensor *Rtn = (BlockUniTensor*)rhs.get();
        std::vector<string> out_labels;
        std::vector<Bond> out_bonds;
        cytnx_int64 out_rowrank;


        //no-common label:
        vec_concatenate_(out_labels, this->labels(), rhs->labels());
        for (cytnx_uint64 i = 0; i < this->_bonds.size(); i++)
            out_bonds.push_back(this->_bonds[i].clone());
        for (cytnx_uint64 i = 0; i < rhs->_bonds.size(); i++)
            out_bonds.push_back(rhs->_bonds[i].clone());

        out_rowrank = this->rowrank() + rhs->rowrank();
        vec_concatenate_(out_labels, this->_labels, rhs->_labels);
        
        //cout << out_bonds;
        tmp->Init(out_bonds,out_labels, out_rowrank, this->dtype(), this->device(), this->is_diag());
        
        //tmp->_name = this->_name + "+" + rhs->_name;        

        //check each valid block:
        std::vector<cytnx_uint64> Lidx(this->_bonds.size()); //buffer
        std::vector<cytnx_uint64> Ridx(rhs->_bonds.size());  //buffer
        for(cytnx_int32 b=0;b<tmp->_blocks.size();b++){
            memcpy(&Lidx[0], &tmp->_inner_to_outer_idx[b][0],sizeof(cytnx_uint64)*this->_bonds.size());
            memcpy(&Ridx[0], &tmp->_inner_to_outer_idx[b][this->_bonds.size()],sizeof(cytnx_uint64)*rhs->_bonds.size());
        
            auto IDL = vec_argwhere(this->_inner_to_outer_idx,Lidx);
            auto IDR = vec_argwhere(Rtn->_inner_to_outer_idx,Ridx);

            /*
            cout << b << endl;
            //vec_print_simple(std::cout,tmp->_inner_to_outer_idx[b]);
            //vec_print_simple(std::cout,Lidx);
            //vec_print_simple(std::cout,Ridx);
            vec_print_simple(std::cout,IDL);
            vec_print_simple(std::cout,IDR);
            */
            if(User_debug){
                if(IDL.size()==IDR.size()){
                    cytnx_error_msg(IDL.size()>1,"[ERROR][BlockUniTensor] IDL has more than two ambiguous location!%s","\n");
                    cytnx_error_msg(IDR.size()>1,"[ERROR][BlockUniTensor] IDL has more than two ambiguous location!%s","\n");
                    
                }else{
                    cytnx_error_msg(true,"[ERROR] duplication, something wrong!%s","\n");
                 
                }
            }
            if(IDL.size()){
                auto tmpR = Rtn->_blocks[IDR[0]];
                std::vector<cytnx_uint64> shape_L =
                    vec_concatenate(this->_blocks[IDL[0]].shape(), std::vector<cytnx_uint64>(tmpR.shape().size(), 1));

                auto tmpL = this->_blocks[IDL[0]].reshape(shape_L);
                auto Ott = linalg::Kron(tmpL,tmpR,false,true);
                //checking:
                cytnx_error_msg(Ott.shape()!=tmp->_blocks[b].shape(),"[ERROR] mismatching shape!%s","\n");
                tmp->_blocks[b] = Ott;
            } 

        }         

        boost::intrusive_ptr<UniTensor_base> out(tmp);
        return out;
           

    }else{
        //first, get common index!
        
        // check qnums & type:
        for (int i = 0; i < comm_labels.size(); i++) {
            if (User_debug){
              cytnx_error_msg(this->_bonds[comm_idx1[i]].qnums() != rhs->_bonds[comm_idx2[i]].qnums(),
                              "[ERROR] contract bond @ label %d have qnum mismatch.\n", comm_labels[i]);
              cytnx_error_msg(this->_bonds[comm_idx1[i]].getDegeneracies() != rhs->_bonds[comm_idx2[i]].getDegeneracies(),
                              "[ERROR] contract bond @ label %d have degeneracies mismatch.\n", comm_labels[i]);
            }
            cytnx_error_msg(this->_bonds[comm_idx1[i]].type() + rhs->_bonds[comm_idx2[i]].type(),
                            "[ERROR] BRA can only contract with KET. invalid @ label: %d\n",
                            comm_labels[i]);
        }
        
        // proc meta, labels:
        std::vector<cytnx_uint64> non_comm_idx1 =
        vec_erase(utils_internal::range_cpu(this->rank()), comm_idx1);
        std::vector<cytnx_uint64> non_comm_idx2 =
        vec_erase(utils_internal::range_cpu(rhs->rank()), comm_idx2);

        std::vector<cytnx_int64> _shadow_comm_idx1(comm_idx1.size()), _shadow_comm_idx2(comm_idx2.size());
        memcpy(_shadow_comm_idx1.data(),comm_idx1.data(),sizeof(cytnx_int64)*comm_idx1.size());
        memcpy(_shadow_comm_idx2.data(),comm_idx2.data(),sizeof(cytnx_int64)*comm_idx2.size());


        
        if ((non_comm_idx1.size() == 0) && (non_comm_idx2.size() == 0)) {
            // All the legs are contracted, the return will be a scalar
            
            // output instance;
            DenseUniTensor *tmp = new DenseUniTensor();
            
            boost::intrusive_ptr<UniTensor_base> Lperm = this->permute(_shadow_comm_idx1);
            boost::intrusive_ptr<UniTensor_base> Rperm = rhs->permute(_shadow_comm_idx2);
            
            BlockUniTensor *Lperm_raw = (BlockUniTensor*)Lperm.get();
            BlockUniTensor *Rperm_raw = (BlockUniTensor*)Rperm.get();
            

            //pair the block and contract using vectordot!
            // naive way!
            for(unsigned int b=0;b<Lperm_raw->_blocks.size();b++){
                for(unsigned int a=0;a<Rperm_raw->_blocks.size();a++){
                    if(Lperm_raw->_inner_to_outer_idx[b] == Rperm_raw->_inner_to_outer_idx[a]){
                        if(tmp->_block.dtype()==Type.Void)
                            tmp->_block = linalg::Vectordot(Lperm_raw->_blocks[b].flatten(),Rperm_raw->_blocks[a].flatten()); 
                        else
                            tmp->_block += linalg::Vectordot(Lperm_raw->_blocks[b].flatten(),Rperm_raw->_blocks[a].flatten());
                        
                        std::cout << b << " " << a << endl;                        


                    } 
                }
            }
            
            tmp->_rowrank = 0;
            tmp->_is_tag = false;
            /*
            if(mv_elem_self){
                // calculate reverse mapper:
                std::vector<cytnx_uint64> inv_mapperL(comm_idx1.size());
                for (int i = 0; i < comm_idx1.size(); i++) {
                  inv_mapperL[comm_idx1[i]] = i;
                }
                for(unsigned int b=0;b<this->_blocks.size();b++){
                    this->_blocks[b].permute_(comm_idx1);
                    this->_blocks[b].contiguous_();
                    this->_blocks[b].permute_(inv_mapperL);
                }
            }

            if(mv_elem_rhs){
                BlockUniTensor *Rtn = (BlockUniTensor*)rhs.get();
                // calculate reverse mapper:
                std::vector<cytnx_uint64> inv_mapperR(comm_idx2.size());
                for (int i = 0; i < comm_idx2.size(); i++) {
                  inv_mapperR[comm_idx2[i]] = i;
                }
                for(unsigned int b=0;b<Rtn->_blocks.size();b++){
                    Rtn->_blocks[b].permute_(comm_idx2);
                    Rtn->_blocks[b].contiguous_();
                    Rtn->_blocks[b].permute_(inv_mapperR);
                }
            }
            */
            boost::intrusive_ptr<UniTensor_base> out(tmp);
            return out;
            

        }else{
            //cytnx_error_msg(true,"developing!%s","\n");
            BlockUniTensor *tmp = new BlockUniTensor();
            BlockUniTensor *Rtn = (BlockUniTensor*)rhs.get();
            std::vector<string> out_labels;
            std::vector<Bond> out_bonds;
            cytnx_int64 out_rowrank;

            // these two cannot omp parallel, due to intrusive_ptr
            for (cytnx_uint64 i = 0; i < non_comm_idx1.size(); i++)
                out_bonds.push_back(this->_bonds[non_comm_idx1[i]].clone());
            for (cytnx_uint64 i = 0; i < non_comm_idx2.size(); i++)
                out_bonds.push_back(rhs->_bonds[non_comm_idx2[i]].clone());

            vec_concatenate_(out_labels, vec_clone(this->_labels, non_comm_idx1),
                       vec_clone(rhs->_labels, non_comm_idx2));

            out_rowrank = this->rowrank() + rhs->rowrank();
            for (cytnx_uint64 i = 0; i < comm_idx1.size(); i++)
              if (comm_idx1[i] < this->_rowrank) out_rowrank--;
            for (cytnx_uint64 i = 0; i < comm_idx2.size(); i++)
              if (comm_idx2[i] < rhs->_rowrank) out_rowrank--;

            // Initialize!!
            tmp->Init(out_bonds,out_labels, out_rowrank, this->dtype(), this->device(), this->is_diag());
 

            // now, build the itoi table:
            std::vector< std::vector<cytnx_uint64> > itoiL_common(this->_blocks.size()), itoiR_common(Rtn->_blocks.size());
            std::vector< std::vector<cytnx_uint64> > Bkk;

            for(cytnx_int64 a=0;a<this->_blocks.size();a++){
                itoiL_common[a] = vec_clone(this->_inner_to_outer_idx[a],comm_idx1);
            }

            for(cytnx_int64 b=0;b<this->_blocks.size();b++){
                itoiR_common[b] = vec_clone(Rtn->_inner_to_outer_idx[b],comm_idx2);
            }

            std::vector<cytnx_uint64> Lgbuffer;
            for(cytnx_int64 a=0;a<this->_blocks.size();a++){
                for(cytnx_int64 b=0;b<Rtn->_blocks.size();b++){
                    //check if common index are the same:
                    if(itoiL_common[a] == itoiR_common[b]){
                        //std::cout << "[contract] " << a <<" " << b << endl;
                        
                        vec_concatenate_(Lgbuffer, vec_clone(this->_inner_to_outer_idx[a],non_comm_idx1)
                                                 , vec_clone(Rtn->_inner_to_outer_idx[b],non_comm_idx2));

                        //find Lgbuffer in tmp, which specify the target block!
                        auto it = std::find(tmp->_inner_to_outer_idx.begin(),tmp->_inner_to_outer_idx.end(),Lgbuffer);
                        if(it != tmp->_inner_to_outer_idx.end()){
                            cytnx_int64 targ_b = it - tmp->_inner_to_outer_idx.begin();
                            //cout << "  "  << "targ blk_id:" << targ_b << endl;
                            tmp->_blocks[targ_b] += linalg::Tensordot(this->_blocks[a], Rtn->_blocks[b], comm_idx1, comm_idx2,
                                          mv_elem_self, mv_elem_rhs);

                        }else{
                            cytnx_error_msg(true,"[ERROR][BlockUniTensor] trying to contract L.blk [%d] with R.blk [%d] but no target blk found!\n",a,b); 
                        }

                    }
                }
            }
            
            boost::intrusive_ptr<UniTensor_base> out(tmp);
            return out;



        } // does it contract all the bond?

        cytnx_error_msg(true,"something wrong!%s","\n");

    } // does it contract all the bond?



  };


  void BlockUniTensor::Transpose_(){
    // modify tag, and reverse qnum:
    for (int i = 0; i < this->bonds().size(); i++) {
      this->bonds()[i].redirect_();
    }

  };
  void BlockUniTensor::Trace_(const cytnx_int64 &a, const cytnx_int64 &b, const bool &by_label){

    // 1) from label to indx.
    cytnx_int64 ida, idb;

    if (by_label) {
      ida = vec_where(this->_labels, std::to_string(a));
      idb = vec_where(this->_labels, std::to_string(b));
    } else {
      cytnx_error_msg(a < 0 || b < 0, "[ERROR] invalid index a, b%s", "\n");
      cytnx_error_msg(a >= this->rank() || b >= this->rank(), "[ERROR] index out of bound%s", "\n");
      ida = a;
      idb = b;
    }

    this->Trace_(ida,idb);

  }
  
  void BlockUniTensor::Trace_(const std::string &a, const std::string &b){
    // 1) from label to indx.
    cytnx_int64 ida, idb;

    ida = vec_where(this->_labels, a);
    idb = vec_where(this->_labels, b);

    this->Trace_(ida,idb);
  }
  void BlockUniTensor::Trace_(const cytnx_int64 &a, const cytnx_int64 &b){

    cytnx_int64 ida = a;
    cytnx_int64 idb = b;

    // check if indices are the same:
    cytnx_error_msg(ida == idb,
                    "[ERROR][BlockUniTensor::Trace_] index a and index b should not be the same.%s",
                    "\n");

    // check if two bonds type are contractable:
    cytnx_error_msg(this->_bonds[ida].type() == this->_bonds[idb].type(),"[ERROR] BD_BRA/BD_OUT can only contract with BD_KET/BD_IN%s","\n");

    // check if two bonds dimension matches:
    cytnx_error_msg(
      this->_bonds[ida]._impl->_degs != this->_bonds[idb]._impl->_degs,
      "[ERROR][BlockUniTensor::Trace_] The dimension of two bond for trace does not match!%s",
      "\n");

    // check if two bonds qnum matches:
    cytnx_error_msg(
      this->_bonds[ida]._impl->_qnums != this->_bonds[idb]._impl->_qnums,
      "[ERROR][BlockUniTensor::Trace_] The quantum numbers of two bond for trace does not match!%s",
      "\n");


    // update rowrank:
    cytnx_int64 tmpRk = this->_rowrank;
    if (ida < tmpRk) this->_rowrank--;
    if (idb < tmpRk) this->_rowrank--;

    //trace the block!
    for(cytnx_int64 i=0;i<this->_blocks.size();i++){
        this->_blocks[i] = this->_blocks[i].Trace(ida,idb);
    }


    //// deal with Metas:

    // 1) remove the bond, labels:
    if (ida > idb) std::swap(ida, idb);
    this->_bonds.erase(this->_bonds.begin() + idb);
    this->_bonds.erase(this->_bonds.begin() + ida);
    this->_labels.erase(this->_labels.begin() + idb);
    this->_labels.erase(this->_labels.begin() + ida);


    // 2) inner to outer:
    for(cytnx_int64 i=0;i < this->_inner_to_outer_idx.size();i++){
        this->_inner_to_outer_idx[i].erase(this->_inner_to_outer_idx[i].begin() + idb);
        this->_inner_to_outer_idx[i].erase(this->_inner_to_outer_idx[i].begin() + ida);
    }

  }


  Tensor BlockUniTensor::Norm() const{
        Scalar t;
        if (this->_blocks.size()) {
          t = linalg::Norm(this->_blocks[0]).item();
          t *= t;
          for (int blk = 1; blk < this->_blocks.size(); blk++) {
            Scalar tmp = linalg::Norm(this->_blocks[blk]).item();
            t += tmp * tmp;
          }

        } else {
          t = Scalar(0, Type.Double);
        }

        t = sqrt(t);
        Tensor R({1}, t.dtype());

        R(0) = t;
        return R;
  }


  // helper function:
  void BlockUniTensor::_fx_locate_elem(cytnx_int64 &bidx, std::vector<cytnx_uint64> &loc_in_T,const std::vector<cytnx_uint64> &locator) const {
    // 1. check if out of range:
    cytnx_error_msg(locator.size() != this->_bonds.size(),
                    "[ERROR] len(locator) does not match the rank of tensor.%s", "\n");


    for (int i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(locator[i] >= this->_bonds[i].dim(),
                      "[ERROR][SparseUniTensor][elem_exists] locator @index: %d out of range.\n",
                      i);
    }

    // 2. calculate the location is in which qindices:
    loc_in_T = locator;
    std::vector<cytnx_uint64> qindices(loc_in_T.size());
    for(int i=0;i<this->_bonds.size();i++){
        for(int d=0;d<this->_bonds[i]._impl->_degs.size();d++){
            if(loc_in_T[i] >= this->_bonds[i]._impl->_degs[d]) loc_in_T[i] -= this->_bonds[i]._impl->_degs[d];
            else{qindices[i] = d; break;}
        }
    }

    auto it = std::find(this->_inner_to_outer_idx.begin(),this->_inner_to_outer_idx.end(),qindices);
    
    if(it == this->_inner_to_outer_idx.end()) bidx = -1;
    else bidx = it - this->_inner_to_outer_idx.begin();
  }



  bool BlockUniTensor::elem_exists(const std::vector<cytnx_uint64> &locator) const{
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx,loc_in_T,locator); 
    return !(bidx < 0);
  }

  //-------------------------------------------
  // at_for_sparse
  Scalar::Sproxy BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator){
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx,loc_in_T,locator); 
    if(bidx<0){
        return Scalar::Sproxy(this->NullRefTensor.storage()._impl,0);
    }else{
        return this->_blocks[bidx].at(loc_in_T);
    }
  }
  cytnx_complex128 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                    const cytnx_complex128 &aux){
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx,loc_in_T,locator); 
    return this->_blocks[bidx].at<cytnx_complex128>(loc_in_T);
  }
  cytnx_complex64 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                   const cytnx_complex64 &aux){
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx,loc_in_T,locator); 
    return this->_blocks[bidx].at<cytnx_complex64>(loc_in_T);

  }
  cytnx_double &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_double &aux){
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx,loc_in_T,locator); 
    return this->_blocks[bidx].at<cytnx_double>(loc_in_T);

  }
  cytnx_float &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_float &aux){
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx,loc_in_T,locator); 
    return this->_blocks[bidx].at<cytnx_float>(loc_in_T);
  }
  cytnx_uint64 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint64 &aux){
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx,loc_in_T,locator); 
    return this->_blocks[bidx].at<cytnx_uint64>(loc_in_T);
  }
  cytnx_int64 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int64 &aux){
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx,loc_in_T,locator); 
    return this->_blocks[bidx].at<cytnx_int64>(loc_in_T);
  }
  cytnx_uint32 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint32 &aux){
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx,loc_in_T,locator); 
    return this->_blocks[bidx].at<cytnx_uint32>(loc_in_T);
  }
  cytnx_int32 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int32 &aux){
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx,loc_in_T,locator); 
    return this->_blocks[bidx].at<cytnx_int32>(loc_in_T);
  }
  cytnx_uint16 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint16 &aux){
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx,loc_in_T,locator); 
    return this->_blocks[bidx].at<cytnx_uint16>(loc_in_T);
  }
  cytnx_int16 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int16 &aux){
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx,loc_in_T,locator); 
    return this->_blocks[bidx].at<cytnx_int16>(loc_in_T);
  }


  const Scalar::Sproxy BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator) const{
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx,loc_in_T,locator); 
    if(bidx<0){
        return Scalar::Sproxy(this->NullRefTensor.storage()._impl,0);
    }else{
        return this->_blocks[bidx].at(loc_in_T);
    }
  }
  const cytnx_complex128 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                    const cytnx_complex128 &aux)const {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx,loc_in_T,locator); 
    return this->_blocks[bidx].at<cytnx_complex128>(loc_in_T);
  }
  const cytnx_complex64 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                   const cytnx_complex64 &aux)const {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx,loc_in_T,locator); 
    return this->_blocks[bidx].at<cytnx_complex64>(loc_in_T);

  }
  const cytnx_double &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_double &aux)const {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx,loc_in_T,locator); 
    return this->_blocks[bidx].at<cytnx_double>(loc_in_T);

  }
  const cytnx_float &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_float &aux)const {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx,loc_in_T,locator); 
    return this->_blocks[bidx].at<cytnx_float>(loc_in_T);
  }
  const cytnx_uint64 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint64 &aux)const {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx,loc_in_T,locator); 
    return this->_blocks[bidx].at<cytnx_uint64>(loc_in_T);
  }
  const cytnx_int64 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int64 &aux)const {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx,loc_in_T,locator); 
    return this->_blocks[bidx].at<cytnx_int64>(loc_in_T);
  }
  const cytnx_uint32 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint32 &aux)const {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx,loc_in_T,locator); 
    return this->_blocks[bidx].at<cytnx_uint32>(loc_in_T);
  }
  const cytnx_int32 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int32 &aux)const {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx,loc_in_T,locator); 
    return this->_blocks[bidx].at<cytnx_int32>(loc_in_T);
  }
  const cytnx_uint16 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_uint16 &aux)const {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx,loc_in_T,locator); 
    return this->_blocks[bidx].at<cytnx_uint16>(loc_in_T);
  }
  const cytnx_int16 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator, const cytnx_int16 &aux)const {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx,loc_in_T,locator); 
    return this->_blocks[bidx].at<cytnx_int16>(loc_in_T);
  }


  void BlockUniTensor::_save_dispatch(std::fstream &f) const {
    // cytnx_error_msg(true,"[ERROR] Save for SparseUniTensor is under developing!!%s","\n");

    cytnx_uint64 Nblocks = this->_blocks.size();
    f.write((char *)&Nblocks, sizeof(cytnx_uint64));

    // save inner_to_outer_idx:
    for(unsigned int b=0;b<Nblocks;b++){
        f.write((char*)&this->_inner_to_outer_idx[b][0],sizeof(cytnx_uint64)*this->_bonds.size());
    }
    for (unsigned int i = 0; i < this->_blocks.size(); i++) {
      this->_blocks[i]._Save(f);
    }
  }

  void BlockUniTensor::_load_dispatch(std::fstream &f) {
    // cytnx_error_msg(true,"[ERROR] Save for SparseUniTensor is under developing!!%s","\n");

    cytnx_uint64 Nblocks;
    f.read((char *)&Nblocks, sizeof(cytnx_uint64));

    this->_inner_to_outer_idx = std::vector< std::vector<cytnx_uint64> >(Nblocks,std::vector<cytnx_uint64>(this->_bonds.size()));       
    // read inner_to_outer_idx:
    for(unsigned int b=0;b<Nblocks;b++){
        f.read((char*)&this->_inner_to_outer_idx[b][0],sizeof(cytnx_uint64)*this->_bonds.size());
    }
    this->_blocks.resize(Nblocks);

    for (unsigned int i = 0; i < this->_blocks.size(); i++) {
      this->_blocks[i]._Load(f);
    }
  }


  void BlockUniTensor::truncate_(const cytnx_int64 &bond_idx, const cytnx_uint64 &q_index,
                           const bool &by_label){

        cytnx_int64 bidx = bond_idx;
        if(by_label){
            auto it = std::find(this->_labels.begin(), this->_labels.end(), to_string(bond_idx));
            cytnx_error_msg(it == this->_labels.end(),
                    "[ERROR] label [%d] does not exist in current UniTensor.\n", bond_idx);
            bidx = it - this->_labels.begin();
        }

        cytnx_error_msg((bidx>=this->_labels.size())|| (bidx < 0), "[ERROR][BlockUniTensor][truncate_] bond_idx out of bound.%s","\n");   
        cytnx_error_msg(q_index >= this->_bonds[bidx].qnums().size(), "[ERROR][BlockUniTensor][truncate_] q_index out of bound @ specify Bond @[%d].\n",bidx);   

        cytnx_error_msg(this->_bonds[bidx].qnums().size()==1,"[ERROR][BlockUniTensor][truncate_] cannot remove the only qnums on a given Bond!%s","\n");

        this->_bonds[bidx]._impl->_rm_qnum(q_index);

        //traversal all blocks, find all blocks that need to remove:
        std::vector<cytnx_uint64> locs;
        for(cytnx_int64 b=0;b<this->_blocks.size();b++){
            if(this->_inner_to_outer_idx[b][bidx] == q_index) locs.push_back(b);
        }

        //remove!
        vec_erase_(this->_inner_to_outer_idx,locs);
        vec_erase_(this->_blocks,locs);



  }
  void BlockUniTensor::truncate_(const std::string &bond_idx, const cytnx_uint64 &q_index){
    auto it = std::find(this->_labels.begin(), this->_labels.end(), bond_idx);
    cytnx_error_msg(it == this->_labels.end(),
                    "[ERROR] label [%s] does not exist in current UniTensor.\n", bond_idx.c_str());

    cytnx_int64 idx = it - this->_labels.begin();
    this->truncate_(idx,q_index,false);
  }
  void BlockUniTensor::truncate_(const cytnx_int64 &bond_idx, const cytnx_uint64 &q_index){
    this->truncate_(bond_idx,q_index,false);
  }



  void BlockUniTensor::Mul_(const Scalar &rhs) {
    // cytnx_error_msg(true,"[ERROR] cannot perform arithmetic on all tagged tensor, @spase
    // unitensor%s","\n");
    for (cytnx_int64 i = 0; i < this->_blocks.size(); i++) {
      this->_blocks[i] *= rhs;
    }
  }

  void BlockUniTensor::Div_(const Scalar &rhs) {
    // cytnx_error_msg(true,"[ERROR] cannot perform arithmetic on all tagged tensor, @spase
    // unitensor%s","\n");
    for (cytnx_int64 i = 0; i < this->_blocks.size(); i++) {
      this->_blocks[i] /= rhs;
    }
  }


  void BlockUniTensor::Add_(const boost::intrusive_ptr<UniTensor_base> &rhs){
    //checking Type:
    cytnx_error_msg(rhs->uten_type()!=UTenType.Block,"[ERROR] cannot add two UniTensor with different type/format.%s","\n");

    BlockUniTensor* Rtn = (BlockUniTensor*)rhs.get();
    
    // 1) check each bond.
    cytnx_error_msg(this->_bonds.size()!=Rtn->_bonds.size(),"[ERROR] cannot add two BlockUniTensor with different rank!%s","\n");
    for(cytnx_int64 i=0;i<this->_bonds.size();i++){
        cytnx_error_msg(this->_bonds[i] != Rtn->_bonds[i],"[ERROR] Bond @ index: %d does not match. Therefore cannot perform Add of two UniTensor\n",i);
    }

    // 2) finding the blocks (they might be not in the same order!
    for(cytnx_int64 b=0;b<this->_blocks.size();b++){
        for(cytnx_int64 a=0;a<Rtn->_blocks.size();a++){
            if(this->_inner_to_outer_idx[b] == Rtn->_inner_to_outer_idx[(b+a)%Rtn->_blocks.size()]){
                this->_blocks[b] += Rtn->_blocks[(b+a)%Rtn->_blocks.size()];
                break;
            }
        }
    }

  }

  void BlockUniTensor::Mul_(const boost::intrusive_ptr<UniTensor_base> &rhs){
    //checking Type:
    cytnx_error_msg(rhs->uten_type()!=UTenType.Block,"[ERROR] cannot add two UniTensor with different type/format.%s","\n");

    BlockUniTensor* Rtn = (BlockUniTensor*)rhs.get();

    // 1) check each bond.
    cytnx_error_msg(this->_bonds.size()!=Rtn->_bonds.size(),"[ERROR] cannot add two BlockUniTensor with different rank!%s","\n");
    for(cytnx_int64 i=0;i<this->_bonds.size();i++){
        cytnx_error_msg(this->_bonds[i] != Rtn->_bonds[i],"[ERROR] Bond @ index: %d does not match. Therefore cannot perform Add of two UniTensor\n",i);
    }

    // 2) finding the blocks (they might be not in the same order!
    for(cytnx_int64 b=0;b<this->_blocks.size();b++){
        for(cytnx_int64 a=0;a<Rtn->_blocks.size();a++){
            if(this->_inner_to_outer_idx[b] == Rtn->_inner_to_outer_idx[(b+a)%Rtn->_blocks.size()]){
                this->_blocks[b] *= Rtn->_blocks[(b+a)%Rtn->_blocks.size()];
                break;
            }
        }
    }

  }

  void BlockUniTensor::Sub_(const boost::intrusive_ptr<UniTensor_base> &rhs){
    //checking Type:
    cytnx_error_msg(rhs->uten_type()!=UTenType.Block,"[ERROR] cannot add two UniTensor with different type/format.%s","\n");

    BlockUniTensor* Rtn = (BlockUniTensor*)rhs.get();

    // 1) check each bond.
    cytnx_error_msg(this->_bonds.size()!=Rtn->_bonds.size(),"[ERROR] cannot add two BlockUniTensor with different rank!%s","\n");
    for(cytnx_int64 i=0;i<this->_bonds.size();i++){
        cytnx_error_msg(this->_bonds[i] != Rtn->_bonds[i],"[ERROR] Bond @ index: %d does not match. Therefore cannot perform Add of two UniTensor\n",i);
    }

    // 2) finding the blocks (they might be not in the same order!
    for(cytnx_int64 b=0;b<this->_blocks.size();b++){
        for(cytnx_int64 a=0;a<Rtn->_blocks.size();a++){
            if(this->_inner_to_outer_idx[b] == Rtn->_inner_to_outer_idx[(b+a)%Rtn->_blocks.size()]){
                this->_blocks[b] -= Rtn->_blocks[(b+a)%Rtn->_blocks.size()];
                break;
            }
        }
    }

  }



}  // namespace cytnx
