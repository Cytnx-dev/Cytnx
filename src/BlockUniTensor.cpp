#include "UniTensor.hpp"
#include "Accessor.hpp"
#include "utils/utils.hpp"
#include "linalg.hpp"
#include "Generator.hpp"
#include <vector>
#include "utils/vec_print.hpp"
#include "utils/vec_concatenate.hpp"
#include <map>
#include <boost/unordered_map.hpp>
#include <stack>
using namespace std;

#ifdef BACKEND_TORCH
#else

  #ifdef UNI_OMP
    #include <omp.h>
  #endif
namespace cytnx {
  typedef Accessor ac;
  void BlockUniTensor::Init(const std::vector<Bond> &bonds, const std::vector<string> &in_labels,
                            const cytnx_int64 &rowrank, const unsigned int &dtype,
                            const int &device, const bool &is_diag, const bool &no_alloc,
                            const std::string &name) {
    this->_name = name;
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

    if (rowrank == -1) {
      this->_rowrank = N_ket;
      // this->_inner_rowrank = N_ket;
    } else {
      if (is_diag) {
        cytnx_error_msg(rowrank != 1,
                        "[ERROR][BlockUniTensor] rowrank must be = 1 when is_diag = true.%s", "\n");
      } else {
        cytnx_error_msg((rowrank < 0) || (rowrank > bonds.size()),
                        "[ERROR][BlockUniTensor] rowrank must be >=0 and <=rank.%s", "\n");
      }
      this->_rowrank = rowrank;
      // this->_inner_rowrank = rowrank;
      //  update braket_form >>>
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

    // cytnx_error_msg(is_diag,"[ERROR][BlockUniTensor] Cannot set is_diag=true when the UniTensor
    // is with symmetry.%s","\n");
    if (is_diag) {
      cytnx_error_msg(bonds.size() != 2,
                      "[ERROR][BlockUniTensor] is_diag = true must be rank-2 with one in-bond and "
                      "one out-bond.%s",
                      "\n");
      cytnx_error_msg(
        bonds[0].type() == bonds[1].type(),
        "[ERROR][BlockUniTensor] is_diag=true must have one in-bond and oue out-bond.%s", "\n");
      if (rowrank != 1, "[ERROR][BlockUniTensor] is_diag = true must have rowrank=1.%s", "\n")
        ;

      // checking basis!
      cytnx_error_msg(
        bonds[0].redirect() != bonds[1],
        "[ERROR][BlockUniTensor] is_diag=true the in-bond and out-bond basis must match!%s", "\n");
    }
    this->_is_diag = is_diag;

    // copy bonds, otherwise it will share objects:
    this->_bonds = vec_clone(bonds);
    this->_is_braket_form = this->_update_braket();

    // vector<cytnx_uint64> blocklens;
    // vector<vector<cytnx_uint64>> blocksizes;
    // cytnx_uint64 totblocksize = 0;

    if (this->_is_diag) {
      for (int b = 0; b < this->_bonds[0].qnums().size(); b++) {
        this->_inner_to_outer_idx.push_back({(cytnx_uint64)b, (cytnx_uint64)b});
        if (!no_alloc) {
          this->_blocks.push_back(zeros(this->_bonds[0]._impl->_degs[b], dtype, device));
        } else {
          this->_blocks.push_back(Tensor({this->_bonds[0]._impl->_degs[b]}, dtype, device, false));
        }
      }

    } else {
      // checking how many blocks are there, and the size:
      std::vector<cytnx_uint64> Loc(this->_bonds.size(), 0);
      std::vector<cytnx_int64> tot_qns(
        this->_bonds[0].Nsym());  // use first bond to determine symmetry size
      std::vector<cytnx_uint64> size(this->_bonds.size());
      bool fin = false;
      while (1) {
        // get elem
        // cout << "start!" << endl;
        // cytnx::vec_print_simple(std::cout , Loc);
        this->_fx_get_total_fluxs(Loc, this->_bonds[0].syms(), tot_qns);

        // std::cout << "Loc: ";
        // cytnx::vec_print_simple(std::cout, Loc);
        // std::cout << "tot_flx: ";
        // cytnx::vec_print_simple(std::cout, tot_qns);

        // if exists:
        if (std::all_of(tot_qns.begin(), tot_qns.end(), [](const int &i) { return i == 0; })) {
          // get size & init block!
          if (!no_alloc) {
            // cytnx_uint64 blockNelem = 1;
            for (cytnx_int32 i = 0; i < Loc.size(); i++) {
              size[i] = this->_bonds[i]._impl->_degs[Loc[i]];
              // blockNelem *= size[i];
            }
            this->_blocks.push_back(zeros(size, dtype, device));
            // blocklens.push_back(blockNelem);
            // blocksizes.push_back(size);
            // totblocksize += blockNelem;
          } else {
            for (cytnx_int32 i = 0; i < Loc.size(); i++) {
              size[i] = this->_bonds[i]._impl->_degs[Loc[i]];
            }
            this->_blocks.push_back(Tensor(size, dtype, device, false));
          }
          // push its loc
          this->_inner_to_outer_idx.push_back(Loc);
        }

        while (Loc.size() != 0) {
          if (Loc.back() == this->_bonds[Loc.size() - 1]._impl->_qnums.size() - 1) {
            Loc.pop_back();
            continue;
          } else {
            Loc.back() += 1;
            // cout << "+1 at loc:" << Loc.size()-1 <<endl;
            while (Loc.size() != this->_bonds.size()) {
              Loc.push_back(0);
            }
            break;
          }
        }

        if (Loc.size() == 0) break;
      }

      // if(!no_alloc){
      //   cytnx_uint64 offset=0;

      //   char* ptr = (char*)utils_internal::Calloc_cpu(
      //     totblocksize+blocklens.size()*STORAGE_DEFT_SZ,
      //     Type.typeSize(dtype));
      //   for(cytnx_int64 k=0;k<blocklens.size();k++){
      //     cytnx_uint64 cap=0;
      //     if (blocklens[k] % STORAGE_DEFT_SZ) {
      //       cap = ((unsigned long long)((blocklens[k]) / STORAGE_DEFT_SZ) + 1) * STORAGE_DEFT_SZ;
      //     } else {
      //       cap = blocklens[k];
      //     }
      //     this->_blocks.push_back(Tensor(Storage(ptr+(offset*Type.typeSize(dtype)),
      //         blocklens[k],dtype,device,true,cap),blocksizes[k],dtype,device));
      //     offset+=cap;
      //   }
      // }
    }  // is_diag?
  }

  void beauty_print_block(std::ostream &os, const cytnx_uint64 &Nin, const cytnx_uint64 &Nout,
                          const std::vector<cytnx_uint64> &qn_indices,
                          const std::vector<Bond> &bonds, const Tensor &block) {
    cytnx_uint64 Total_line = Nin < Nout ? Nout : Nin;

    std::vector<std::string> Lside(Total_line);
    std::vector<std::string> Rside(Total_line);
    std::vector<std::string> MidL(Total_line);
    std::vector<std::string> MidR(Total_line);
    cytnx_uint64 Lmax = 0;
    cytnx_uint64 mL = 0;
    cytnx_uint64 mR = 0;

    for (int i = 0; i < Total_line; i++) {
      // Lside:
      if (i < Nin) {
        Lside[i] += "[" + to_string(qn_indices[i]) + "] ";
        for (int s = 0; s < bonds[0].Nsym(); s++) {
          Lside[i] += bonds[0]._impl->_syms[s].stype_str() + "(" +
                      to_string(bonds[i]._impl->_qnums[qn_indices[i]][s]) + ")";
        }
        if (Lmax < Lside[i].size()) Lmax = Lside[i].size();

        MidL[i] += to_string(block.shape()[i]);
        if (mL < MidL[i].size()) mL = MidL[i].size();
      }

      // Rside:
      if (i < Nout) {
        Rside[i] += "[" + to_string(qn_indices[Nin + i]) + "] ";
        for (int s = 0; s < bonds[0].Nsym(); s++) {
          Rside[i] += bonds[0]._impl->_syms[s].stype_str() + "(" +
                      to_string(bonds[Nin + i]._impl->_qnums[qn_indices[Nin + i]][s]) + ")";
        }
        // check if is_diag = true:
        if (block.shape().size() == 1 && bonds.size() == 2)
          MidR[i] += to_string(block.shape()[i]);
        else
          MidR[i] += to_string(block.shape()[Nin + i]);
        if (mR < MidR[i].size()) mR = MidR[i].size();
      }
    }

    // filling space:
    for (int i = 0; i < Total_line; i++) {
      if (Lside[i].size() < Lmax) {
        Lside[i] += string(" ") * (Lmax - Lside[i].size());
      }
      if (MidL[i].size() < mL) {
        MidL[i] += string(" ") * (mL - MidL[i].size());
      }
      if (MidR[i].size() < mR) {
        MidR[i] += string(" ") * (mR - MidR[i].size());
      }
    }

    // starting printing:
    //  3spacing, Lmax , 5 for arrow
    std::string empty_line =
      (std::string(" ") * (3 + Lmax + 5)) + "| " + std::string(" ") * (mL + 5 + mR) + " |";
    os << (std::string(" ") * (3 + Lmax + 5)) << std::string("-") * (4 + mL + mR + 5) << endl;
    os << empty_line << endl;

    std::string bks;
    for (int i = 0; i < Total_line; i++) {
      os << "   " << Lside[i];
      // arrow:
      if (i < Nin) {
        if (bonds[i].type() == bondType::BD_KET)
          bks = "  -->";
        else
          bks = " *<--";
      } else {
        bks = "     ";
      }
      os << bks << "| " << MidL[i] << "     " << MidR[i] << " |";
      if (i < Nout) {
        if (bonds[Nin + i].type() == bondType::BD_KET)
          bks = "<--* ";
        else
          bks = "-->  ";
      } else {
        bks = "";
      }

      os << bks << Rside[i] << endl;
      os << empty_line << endl;
    }

    os << (std::string(" ") * (3 + Lmax + 5)) << std::string("-") * (4 + mL + mR + 5) << endl;
  }
  void BlockUniTensor::print_block(const cytnx_int64 &idx, const bool &full_info) const {
    cytnx_error_msg(
      (idx < 0) || (idx >= this->_blocks.size()),
      "[ERROR] index [%d] out of bound. should be >0 and < number of available blocks %d\n", idx,
      this->_blocks.size());

    std::ostream &os = std::cout;

    os << "========================\n";
    if (this->_is_diag) os << " *is_diag: True\n";
    os << "BLOCK [#" << idx << "]\n";
    /*
    os << "  |-Qn indices for each axis:\n   {\t";
    for(int s=0;s<this->_inner_to_outer_idx[idx].size();s++){
        os << this->_inner_to_outer_idx[idx][s] << "\t";
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
        for(int l=0;l<this->_blocks[idx].shape().size();l++){
            os << std::showpos <<
    this->_bonds[l]._impl->_qnums[this->_inner_to_outer_idx[idx][l]][s] << "\t";
        }
        os << std::noshowpos << endl;
    }
    */
    os << " |- []   : Qn index \n";
    os << " |- Sym(): Qnum of correspond symmetry\n";
    beauty_print_block(os, this->_rowrank, this->_labels.size() - this->_rowrank,
                       this->_inner_to_outer_idx[idx], this->_bonds, this->_blocks[idx]);

    if (full_info)
      os << this->_blocks[idx];
    else {
      os << "  |-dtype:\t" << Type.getname(this->_blocks[idx].dtype()) << endl;
      os << "  |-device:\t" << Device.getname(this->_blocks[idx].device()) << endl;
      os << "  |-contiguous:\t" << (this->_blocks[idx].is_contiguous() ? "True" : "False") << endl;
      os << "  |-shape:\t";
      vec_print_simple(os, this->_blocks[idx].shape());
    }
  }

  void BlockUniTensor::print_blocks(const bool &full_info) const {
    std::ostream &os = std::cout;

    os << "-------- start of print ---------\n";
    char *buffer = (char *)malloc(sizeof(char) * 10240);
    sprintf(buffer, "Tensor name: %s\n", this->_name.c_str());
    os << std::string(buffer);
    if (this->_is_tag)
      sprintf(buffer, "braket_form : %s\n", this->_is_braket_form ? "True" : "False");
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
    for (int b = 0; b < this->_blocks.size(); b++) {
      this->print_block(b, full_info);
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
    char *buffer = (char *)malloc(10240 * sizeof(char));
    unsigned int BUFFsize = 100;

    sprintf(buffer, "-----------------------%s", "\n");
    std::cout << std::string(buffer);
    sprintf(buffer, "tensor Name : %s\n", this->_name.c_str());
    std::cout << std::string(buffer);
    sprintf(buffer, "tensor Rank : %d\n", this->_labels.size());
    std::cout << std::string(buffer);
    // sprintf(buffer, "block_form  : true%s", "\n");
    // std::cout << std::string(buffer);
    sprintf(buffer, "contiguous  : %s\n", this->is_contiguous() ? "True" : "False");
    std::cout << std::string(buffer);
    sprintf(buffer, "valid blocks : %d\n", this->_blocks.size());
    std::cout << std::string(buffer);
    sprintf(buffer, "is diag   : %s\n", this->is_diag() ? "True" : "False");
    std::cout << std::string(buffer);
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

    int Space_Llabel_max = 0, Space_Ldim_max = 0, Space_Rdim_max = 0;
    // quickly checking the size for each line, only check the largest!

    for (cytnx_uint64 i = 0; i < vl; i++) {
      if (i < Nin) {
        if (Space_Llabel_max < this->_labels[i].size()) Space_Llabel_max = this->_labels[i].size();
        if (Space_Ldim_max < to_string(this->_bonds[i].dim()).size())
          Space_Ldim_max = to_string(this->_bonds[i].dim()).size();
      }
      if (i < Nout) {
        if (Space_Rdim_max < to_string(this->_bonds[Nin + i].dim()).size())
          Space_Rdim_max = to_string(this->_bonds[Nin + i].dim()).size();
      }
    }
    string LallSpace = (string(" ") * (Space_Llabel_max + 3 + 1));
    string MallSpace = string(" ") * (1 + Space_Ldim_max + 5 + Space_Rdim_max + 1);
    string M_dashes = string("-") * (1 + Space_Ldim_max + 5 + Space_Rdim_max + 1);

    std::string tmpss;
    sprintf(buffer, "%s row %s col %s", LallSpace.c_str(), MallSpace.c_str(), "\n");
    std::cout << std::string(buffer);
    sprintf(buffer, "%s    -%s-    %s", LallSpace.c_str(), M_dashes.c_str(), "\n");
    std::cout << std::string(buffer);
    for (cytnx_uint64 i = 0; i < vl; i++) {
      sprintf(buffer, "%s    |%s|    %s", LallSpace.c_str(), MallSpace.c_str(), "\n");
      std::cout << std::string(buffer);

      if (i < Nin) {
        if (this->_bonds[i].type() == bondType::BD_KET)
          bks = " -->";
        else
          bks = "*<--";
        memset(l, 0, sizeof(char) * BUFFsize);
        memset(llbl, 0, sizeof(char) * BUFFsize);
        tmpss = this->_labels[i] + std::string(" ") * (Space_Llabel_max - this->_labels[i].size());
        sprintf(l, "%s %s", tmpss.c_str(), bks.c_str());
        tmpss = to_string(this->_bonds[i].dim()) +
                std::string(" ") * (Space_Ldim_max - to_string(this->_bonds[i].dim()).size());
        sprintf(llbl, "%s", tmpss.c_str());
      } else {
        memset(l, 0, sizeof(char) * BUFFsize);
        memset(llbl, 0, sizeof(char) * BUFFsize);
        tmpss = std::string(" ") * (Space_Llabel_max + 5);
        sprintf(l, "%s", tmpss.c_str());
        tmpss = std::string(" ") * (Space_Ldim_max);
        sprintf(llbl, "%s", tmpss.c_str());
      }
      if (i < Nout) {
        if (this->_bonds[Nin + i].type() == bondType::BD_KET)
          bks = "<--*";
        else
          bks = "--> ";
        memset(r, 0, sizeof(char) * BUFFsize);
        memset(rlbl, 0, sizeof(char) * BUFFsize);

        sprintf(r, "%s %s", bks.c_str(), this->_labels[Nin + i].c_str());

        tmpss = to_string(this->_bonds[Nin + i].dim()) +
                std::string(" ") * (Space_Rdim_max - to_string(this->_bonds[Nin + i].dim()).size());
        sprintf(rlbl, "%s", tmpss.c_str());

      } else {
        memset(r, 0, sizeof(char) * BUFFsize);
        memset(rlbl, 0, sizeof(char) * BUFFsize);
        sprintf(r, "%s", "        ");
        tmpss = std::string(" ") * Space_Rdim_max;
        sprintf(rlbl, "%s", tmpss.c_str());
      }
      sprintf(buffer, "   %s| %s     %s |%s\n", l, llbl, rlbl, r);
      std::cout << std::string(buffer);
    }
    sprintf(buffer, "%s    |%s|    %s", LallSpace.c_str(), MallSpace.c_str(), "\n");
    std::cout << std::string(buffer);
    sprintf(buffer, "%s    -%s-    %s", LallSpace.c_str(), M_dashes.c_str(), "\n");
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
    if (this->is_contiguous()) {
      boost::intrusive_ptr<UniTensor_base> out(this);
      return out;
    } else {
      BlockUniTensor *tmp = new BlockUniTensor();
      tmp = this->clone_meta(true, true);
      tmp->_blocks.resize(this->_blocks.size());
      for (unsigned int b = 0; b < this->_blocks.size(); b++) {
        if (this->_blocks[b].is_contiguous()) {
          tmp->_blocks[b] = this->_blocks[b].clone();
        } else {
          tmp->_blocks[b] = this->_blocks[b].contiguous();
        }
      }
      boost::intrusive_ptr<UniTensor_base> out(tmp);
      return out;
    }
  }

  std::vector<Symmetry> BlockUniTensor::syms() const { return this->_bonds[0].syms(); }

  boost::intrusive_ptr<UniTensor_base> BlockUniTensor::permute(
    const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank) {
    BlockUniTensor *out_raw = this->clone_meta(true, true);
    out_raw->_blocks.resize(this->_blocks.size());

    std::vector<cytnx_uint64> mapper_u64;

    mapper_u64 = std::vector<cytnx_uint64>(mapper.begin(), mapper.end());
    // checking:
    for (int i = 0; i < mapper_u64.size(); i++) {
      cytnx_error_msg(mapper_u64[i] >= this->rank(), "[ERROR] index %d out of bound!\n",
                      mapper_u64[i]);
    }

    out_raw->_bonds = vec_map(vec_clone(out_raw->bonds()), mapper_u64);  // this will check validity
    out_raw->_labels = vec_map(out_raw->labels(), mapper_u64);

    if (out_raw->_is_diag) {
      // cytnx_error_msg(true,"[ERROR][BlockUniTensor] currently do not support permute for
      // is_diag=true for BlockUniTensor!%s","\n");
      if (rowrank >= 0)
        cytnx_error_msg(rowrank != 1, "[ERROR][BlockUniTensor] is_diag=true must have rowrank=1.%s",
                        "\n");
      out_raw->_is_braket_form = out_raw->_update_braket();

    } else {
      // inner_to_outer permute!
      for (cytnx_int64 b = 0; b < this->_inner_to_outer_idx.size(); b++) {
        out_raw->_inner_to_outer_idx[b] = vec_map(out_raw->_inner_to_outer_idx[b], mapper_u64);
        out_raw->_blocks[b] = this->_blocks[b].permute(mapper_u64);
      }

      if (rowrank >= 0) {
        cytnx_error_msg((rowrank >= out_raw->_bonds.size()) || (rowrank < 1),
                        "[ERROR][BlockUniTensor] rowrank cannot exceed the rank of UniTensor-1, "
                        "and should be >=1.%s",
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
    BlockUniTensor *out_raw = this->clone_meta(true, true);
    out_raw->_blocks.resize(this->_blocks.size());

    std::vector<cytnx_int64> mapper_i64;
    // cytnx_error_msg(true,"[Developing!]%s","\n");
    std::vector<string>::iterator it;
    for (cytnx_int64 i = 0; i < mapper.size(); i++) {
      it = std::find(out_raw->_labels.begin(), out_raw->_labels.end(), mapper[i]);
      cytnx_error_msg(it == out_raw->_labels.end(),
                      "[ERROR] label %s does not exist in current UniTensor.\n", mapper[i].c_str());
      mapper_i64.push_back(std::distance(out_raw->_labels.begin(), it));
    }

    return this->permute(mapper_i64, rowrank);
  }

  void BlockUniTensor::permute_(const std::vector<cytnx_int64> &mapper,
                                const cytnx_int64 &rowrank) {
    std::vector<cytnx_uint64> mapper_u64;

    mapper_u64 = std::vector<cytnx_uint64>(mapper.begin(), mapper.end());
    // checking:
    for (int i = 0; i < mapper_u64.size(); i++) {
      cytnx_error_msg(mapper_u64[i] >= this->rank(), "[ERROR] index %d out of bound!\n",
                      mapper_u64[i]);
    }

    this->_bonds = vec_map(vec_clone(this->bonds()), mapper_u64);  // this will check validity
    this->_labels = vec_map(this->labels(), mapper_u64);

    if (this->_is_diag) {
      if (rowrank >= 0)
        cytnx_error_msg(rowrank != 1, "[ERROR][BlockUniTensor] is_diag=true must have rowrank=1.%s",
                        "\n");
      this->_is_braket_form = this->_update_braket();

    } else {
      // inner_to_outer permute!
      for (cytnx_int64 b = 0; b < this->_inner_to_outer_idx.size(); b++) {
        this->_inner_to_outer_idx[b] = vec_map(this->_inner_to_outer_idx[b], mapper_u64);
        this->_blocks[b].permute_(mapper_u64);
      }

      if (rowrank >= 0) {
        cytnx_error_msg((rowrank >= this->_bonds.size()) || (rowrank < 1),
                        "[ERROR][BlockUniTensor] rowrank cannot exceed the rank of UniTensor-1, "
                        "and should be >=1.%s",
                        "\n");
        this->_rowrank = rowrank;
      }
      this->_is_braket_form = this->_update_braket();
    }
  }

  void BlockUniTensor::permute_(const std::vector<std::string> &mapper,
                                const cytnx_int64 &rowrank) {
    std::vector<cytnx_int64> mapper_i64;
    // cytnx_error_msg(true,"[Developing!]%s","\n");
    std::vector<std::string>::iterator it;
    for (cytnx_uint64 i = 0; i < mapper.size(); i++) {
      it = std::find(this->_labels.begin(), this->_labels.end(), mapper[i]);
      cytnx_error_msg(it == this->_labels.end(),
                      "[ERROR] label %d does not exist in current UniTensor.\n", mapper[i].c_str());
      mapper_i64.push_back(std::distance(this->_labels.begin(), it));
    }

    this->permute_(mapper_i64, rowrank);
  }

  boost::intrusive_ptr<UniTensor_base> BlockUniTensor::relabel(
    const std::vector<string> &new_labels) {
    BlockUniTensor *tmp = this->clone_meta(true, true);
    tmp->_blocks = this->_blocks;
    tmp->set_labels(new_labels);
    boost::intrusive_ptr<UniTensor_base> out(tmp);
    return out;
  }

  boost::intrusive_ptr<UniTensor_base> BlockUniTensor::relabel(
    const std::vector<std::string> &old_labels, const std::vector<std::string> &new_labels) {
    BlockUniTensor *tmp = this->clone_meta(true, true);
    tmp->_blocks = this->_blocks;
    tmp->relabel_(old_labels, new_labels);
    boost::intrusive_ptr<UniTensor_base> out(tmp);
    return out;
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
    const std::vector<std::string> &old_labels, const std::vector<std::string> &new_labels) {
    BlockUniTensor *tmp = this->clone_meta(true, true);
    tmp->_blocks = this->_blocks;
    tmp->relabels_(old_labels, new_labels);
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

  boost::intrusive_ptr<UniTensor_base> BlockUniTensor::contract(
    const boost::intrusive_ptr<UniTensor_base> &rhs, const bool &mv_elem_self,
    const bool &mv_elem_rhs) {
    // checking type
    cytnx_error_msg(
      rhs->uten_type() != UTenType.Block,
      "[ERROR] cannot contract symmetry-block UniTensor with other type of UniTensor%s", "\n");

    // checking symmetry:
    cytnx_error_msg(this->syms() != rhs->syms(),
                    "[ERROR] two UniTensor have different symmetry type cannot contract.%s", "\n");

    // get common labels:
    std::vector<string> comm_labels;
    std::vector<cytnx_uint64> comm_idx1, comm_idx2;
    vec_intersect_(comm_labels, this->labels(), rhs->labels(), comm_idx1, comm_idx2);

    if (comm_idx1.size() == 0) {
      // output instance;
      BlockUniTensor *tmp = new BlockUniTensor();
      BlockUniTensor *Rtn = (BlockUniTensor *)rhs.get();
      std::vector<string> out_labels;
      std::vector<Bond> out_bonds;
      cytnx_int64 out_rowrank;

      // no-common label:
      vec_concatenate_(out_labels, this->labels(), rhs->labels());
      for (cytnx_uint64 i = 0; i < this->_bonds.size(); i++)
        out_bonds.push_back(this->_bonds[i].clone());
      for (cytnx_uint64 i = 0; i < rhs->_bonds.size(); i++)
        out_bonds.push_back(rhs->_bonds[i].clone());

      out_rowrank = this->rowrank() + rhs->rowrank();
      vec_concatenate_(out_labels, this->_labels, rhs->_labels);

      // cout << out_bonds;
      tmp->Init(out_bonds, out_labels, out_rowrank, this->dtype(), this->device(), false);

      // tmp->_name = this->_name + "+" + rhs->_name;

      // check each valid block:
      std::vector<cytnx_uint64> Lidx(this->_bonds.size());  // buffer
      std::vector<cytnx_uint64> Ridx(rhs->_bonds.size());  // buffer
      for (cytnx_int32 b = 0; b < tmp->_blocks.size(); b++) {
        memcpy(&Lidx[0], &tmp->_inner_to_outer_idx[b][0],
               sizeof(cytnx_uint64) * this->_bonds.size());
        memcpy(&Ridx[0], &tmp->_inner_to_outer_idx[b][this->_bonds.size()],
               sizeof(cytnx_uint64) * rhs->_bonds.size());

        auto IDL = vec_argwhere(this->_inner_to_outer_idx, Lidx);
        auto IDR = vec_argwhere(Rtn->_inner_to_outer_idx, Ridx);

        /*
        cout << b << endl;
        //vec_print_simple(std::cout,tmp->_inner_to_outer_idx[b]);
        //vec_print_simple(std::cout,Lidx);
        //vec_print_simple(std::cout,Ridx);
        vec_print_simple(std::cout,IDL);
        vec_print_simple(std::cout,IDR);
        */
        if (User_debug) {
          if (IDL.size() == IDR.size()) {
            cytnx_error_msg(IDL.size() > 1,
                            "[ERROR][BlockUniTensor] IDL has more than two ambiguous location!%s",
                            "\n");
            cytnx_error_msg(IDR.size() > 1,
                            "[ERROR][BlockUniTensor] IDL has more than two ambiguous location!%s",
                            "\n");

          } else {
            cytnx_error_msg(true, "[ERROR] duplication, something wrong!%s", "\n");
          }
        }
        if (IDL.size()) {
          auto tmpR = Rtn->is_diag() ? linalg::Diag(Rtn->_blocks[IDR[0]]) : Rtn->_blocks[IDR[0]];
          auto tmpL = this->is_diag() ? linalg::Diag(this->_blocks[IDL[0]]) : this->_blocks[IDL[0]];
          std::vector<cytnx_uint64> shape_L =
            vec_concatenate(tmpL.shape(), std::vector<cytnx_uint64>(tmpR.shape().size(), 1));

          tmpL = tmpL.reshape(shape_L);
          auto Ott = linalg::Kron(tmpL, tmpR, false, true);
          // checking:
          cytnx_error_msg(Ott.shape() != tmp->_blocks[b].shape(), "[ERROR] mismatching shape!%s",
                          "\n");
          tmp->_blocks[b] = Ott;
        }
      }

      boost::intrusive_ptr<UniTensor_base> out(tmp);
      return out;
    } else {
      // first, get common index!

      // check qnums & type:
      for (int i = 0; i < comm_labels.size(); i++) {
        if (User_debug) {
          cytnx_error_msg(this->_bonds[comm_idx1[i]].qnums() != rhs->_bonds[comm_idx2[i]].qnums(),
                          "[ERROR] contract bond @ label %s have qnum mismatch.\n",
                          comm_labels[i].c_str());
          cytnx_error_msg(this->_bonds[comm_idx1[i]].getDegeneracies() !=
                            rhs->_bonds[comm_idx2[i]].getDegeneracies(),
                          "[ERROR] contract bond @ label %s have degeneracies mismatch.\n",
                          comm_labels[i].c_str());
        }
        cytnx_error_msg(this->_bonds[comm_idx1[i]].type() + rhs->_bonds[comm_idx2[i]].type(),
                        "[ERROR] BRA can only contract with KET. invalid @ label: %s\n",
                        comm_labels[i].c_str());
      }

      // proc meta, labels:
      std::vector<cytnx_uint64> non_comm_idx1 = vec_erase(vec_range(this->rank()), comm_idx1);
      std::vector<cytnx_uint64> non_comm_idx2 = vec_erase(vec_range(rhs->rank()), comm_idx2);

      if ((non_comm_idx1.size() == 0) && (non_comm_idx2.size() == 0)) {
        std::vector<cytnx_int64> _shadow_comm_idx1(comm_idx1.size()),
          _shadow_comm_idx2(comm_idx2.size());
        memcpy(_shadow_comm_idx1.data(), comm_idx1.data(), sizeof(cytnx_int64) * comm_idx1.size());
        memcpy(_shadow_comm_idx2.data(), comm_idx2.data(), sizeof(cytnx_int64) * comm_idx2.size());
        // All the legs are contracted, the return will be a scalar

        // output instance;
        DenseUniTensor *tmp = new DenseUniTensor();

        boost::intrusive_ptr<UniTensor_base> Lperm = this->permute(_shadow_comm_idx1);
        boost::intrusive_ptr<UniTensor_base> Rperm = rhs->permute(_shadow_comm_idx2);

        BlockUniTensor *Lperm_raw = (BlockUniTensor *)Lperm.get();
        BlockUniTensor *Rperm_raw = (BlockUniTensor *)Rperm.get();

        // pair the block and contract using vectordot!
        //  naive way!
        for (unsigned int b = 0; b < Lperm_raw->_blocks.size(); b++) {
          for (unsigned int a = 0; a < Rperm_raw->_blocks.size(); a++) {
            if (Lperm_raw->_inner_to_outer_idx[b] == Rperm_raw->_inner_to_outer_idx[a]) {
              if (tmp->_block.dtype() == Type.Void)
                tmp->_block = linalg::Vectordot(Lperm_raw->_blocks[b].flatten(),
                                                Rperm_raw->_blocks[a].flatten());
              else
                tmp->_block += linalg::Vectordot(Lperm_raw->_blocks[b].flatten(),
                                                 Rperm_raw->_blocks[a].flatten());

              // std::cout << b << " " << a << endl;
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

      } else {
        BlockUniTensor *tmp = new BlockUniTensor();
        BlockUniTensor *Rtn = (BlockUniTensor *)rhs.get();
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

  #ifdef UNI_MKL
        // Initialize!!
        if (true or
            (this->dtype() != Type.Double and this->dtype() != Type.ComplexDouble) and
              (this->dtype() != Type.Float and this->dtype() != Type.ComplexFloat) or
            this->is_diag() or Rtn->is_diag()) {
          tmp->Init(out_bonds, out_labels, out_rowrank, this->dtype(), this->device(), false,
                    false);
        } else {
          tmp->Init(out_bonds, out_labels, out_rowrank, this->dtype(), this->device(), false, true);
        }
  #else
        tmp->Init(out_bonds, out_labels, out_rowrank, this->dtype(), this->device(), false, false);
  #endif

        // now, build the itoi table:
        std::vector<std::vector<cytnx_uint64>> itoiL_common(this->_blocks.size()),
          itoiR_common(Rtn->_blocks.size());

        for (cytnx_int64 a = 0; a < this->_blocks.size(); a++) {
          itoiL_common[a] = vec_clone(this->_inner_to_outer_idx[a], comm_idx1);
        }

        boost::unordered_map<std::vector<cytnx_uint64>, std::vector<cytnx_uint64>> mp;
        boost::unordered_map<std::vector<cytnx_uint64>, cytnx_uint64> mpC;

        for (cytnx_int64 b = 0; b < Rtn->_blocks.size(); b++) {
          itoiR_common[b] = vec_clone(Rtn->_inner_to_outer_idx[b], comm_idx2);
          if (!mp[itoiR_common[b]].size())
            mp[itoiR_common[b]] = std::vector<cytnx_uint64>(1, b);
          else
            mp[itoiR_common[b]].push_back(b);
        }
        for (cytnx_int64 b = 0; b < tmp->_blocks.size(); b++) {
          mpC[tmp->_inner_to_outer_idx[b]] = b;
        }

        std::vector<cytnx_uint64> Lgbuffer;
        std::vector<cytnx_uint64> itoiR_idx;
        std::vector<cytnx_uint64> oldshapeL;
        std::vector<std::vector<cytnx_uint64>> oldshapeR(Rtn->_blocks.size(),
                                                         std::vector<cytnx_uint64>());
        std::vector<std::vector<cytnx_uint64>> oldshapeC;
        std::vector<bool> reshaped(tmp->_blocks.size(), false);
        for (cytnx_int64 a = 0; a < tmp->_blocks.size(); a++) {
          oldshapeC.push_back(tmp->_blocks[a].shape());
        }
        std::vector<cytnx_uint64> mapperL, inv_mapperL(this->rank());
        std::vector<cytnx_uint64> mapperR, inv_mapperR(rhs->rank());
        vec_concatenate_(mapperL, non_comm_idx1, comm_idx1);
        vec_concatenate_(mapperR, comm_idx2, non_comm_idx2);
        for (int aa = 0; aa < mapperL.size(); aa++) {
          inv_mapperL[mapperL[aa]] = aa;
        }
        for (int aa = 0; aa < mapperR.size(); aa++) {
          inv_mapperR[mapperR[aa]] = aa;
        }

        if (this->is_diag() != Rtn->is_diag()) {
          for (cytnx_int64 a = 0; a < this->_blocks.size(); a++) {
            cytnx_int64 comm_dim = 1;
            itoiR_idx = mp[itoiL_common[a]];
            for (cytnx_uint64 b : itoiR_idx) {
              Lgbuffer.resize(non_comm_idx1.size() + non_comm_idx2.size());
              for (cytnx_uint64 cc = 0; cc < non_comm_idx1.size(); cc++) {
                Lgbuffer[cc] = this->_inner_to_outer_idx[a][non_comm_idx1[cc]];
              }
              for (cytnx_uint64 cc = non_comm_idx1.size();
                   cc < non_comm_idx1.size() + non_comm_idx2.size(); cc++) {
                Lgbuffer[cc] =
                  Rtn->_inner_to_outer_idx[b][non_comm_idx2[cc - non_comm_idx1.size()]];
              }
              cytnx_int64 targ_b = mpC[Lgbuffer];
              tmp->_blocks[targ_b] += linalg::Tensordot_dg(this->_blocks[a], Rtn->_blocks[b],
                                                           comm_idx1, comm_idx2, this->is_diag());
            }
          }
        } else {
          std::vector<char> transs(Rtn->_blocks.size(), 'N');
          std::vector<blas_int> ms(Rtn->_blocks.size(), 0), ns(Rtn->_blocks.size(), 0),
            ks(Rtn->_blocks.size(), 0);
          std::vector<void *> LMems(Rtn->_blocks.size(), 0), RMems(Rtn->_blocks.size(), 0),
            CMems(Rtn->_blocks.size(), 0);
          std::vector<blas_int> group_size(Rtn->_blocks.size(), 1);
          std::vector<Scalar> alphas(Rtn->_blocks.size(), 1.0);
          std::vector<Scalar> betas(Rtn->_blocks.size(), 0.0);

          BlockUniTensor *tmp_Rtn = Rtn;

          // check if all sub-tensor are same dtype and device
          if (User_debug) {
            bool all_sub_tensor_same_dtype = true;
            bool all_sub_tensor_same_device = true;
            for (cytnx_int64 a = 0; a < this->_blocks.size(); a++) {
              if (this->_blocks[a].dtype() != this->_blocks[0].dtype())
                all_sub_tensor_same_dtype = false;
              if (this->_blocks[a].device() != this->_blocks[0].device())
                all_sub_tensor_same_device = false;
            }
            cytnx_error_msg(
              all_sub_tensor_same_dtype,
              "[ERROR] cannot perform contraction on sub-Tensors with different dtype.%s", "\n");
            cytnx_error_msg(
              all_sub_tensor_same_device,
              "[ERROR] cannot perform contraction on sub-Tensors with different device.%s", "\n");
            all_sub_tensor_same_dtype = true;
            all_sub_tensor_same_device = true;
            for (cytnx_int64 a = 0; a < Rtn->_blocks.size(); a++) {
              if (Rtn->_blocks[a].dtype() != Rtn->_blocks[0].dtype())
                all_sub_tensor_same_dtype = false;
              if (Rtn->_blocks[a].device() != Rtn->_blocks[0].device())
                all_sub_tensor_same_device = false;
            }
            cytnx_error_msg(
              all_sub_tensor_same_dtype,
              "[ERROR] cannot perform contraction on sub-Tensors with different dtype.%s", "\n");
            cytnx_error_msg(
              all_sub_tensor_same_device,
              "[ERROR] cannot perform contraction on sub-Tensors with different device.%s", "\n");
          }
  #ifdef UNI_MKL
          // If the dtype of this and Rtn are different, we need to cast to the common dtype
          if (this->dtype() != Rtn->dtype()) {
            BlockUniTensor *tmpp = Rtn->clone_meta(true, true);
            tmpp->_blocks.resize(Rtn->_blocks.size());
            for (cytnx_int64 blk = 0; blk < Rtn->_blocks.size(); blk++) {
              tmpp->_blocks[blk] = Rtn->_blocks[blk].astype(this->dtype());
            }
            tmp_Rtn = tmpp;
          }
          // First select left block to do gemm
          for (cytnx_int64 a = 0; a < this->_blocks.size(); a++) {
            cytnx_int64 comm_dim = 1;
            // get the indices of right blocks that *can* contract with this->_blocks[a]
            itoiR_idx = mp[itoiL_common[a]];
            for (cytnx_uint64 aa = 0; aa < comm_idx1.size(); aa++) {
              comm_dim *= this->_blocks[a].shape()[comm_idx1[aa]];
            }
            // permute&reshape this->_blocks[a]
            this->_blocks[a].permute_(mapperL);
            oldshapeL = this->_blocks[a].shape();
            this->_blocks[a].reshape_({-1, comm_dim});
            // loop over all right blocks that can contract with this->_blocks[a]
            for (cytnx_uint64 binx = 0; binx < itoiR_idx.size(); binx++) {
              // get the index of the right block
              cytnx_uint64 b = itoiR_idx[binx];
              // permute&reshape Rtn->_blocks[b]
              tmp_Rtn->_blocks[b].permute_(mapperR);
              oldshapeR[b] = tmp_Rtn->_blocks[b].shape();
              tmp_Rtn->_blocks[b].reshape_({comm_dim, -1});
              // prepare to find the target block
              Lgbuffer.resize(non_comm_idx1.size() + non_comm_idx2.size());
              for (cytnx_uint64 cc = 0; cc < non_comm_idx1.size(); cc++) {
                Lgbuffer[cc] = this->_inner_to_outer_idx[a][non_comm_idx1[cc]];
              }
              for (cytnx_uint64 cc = non_comm_idx1.size();
                   cc < non_comm_idx1.size() + non_comm_idx2.size(); cc++) {
                Lgbuffer[cc] =
                  tmp_Rtn->_inner_to_outer_idx[b][non_comm_idx2[cc - non_comm_idx1.size()]];
              }
              // target block index
              cytnx_int64 targ_b = mpC[Lgbuffer];
              betas[binx] = 1.0;
              // if the target block is not initialized, call to gemm with beta=0
              if (!reshaped[targ_b]) {
                tmp->_blocks[targ_b].reshape_({(cytnx_int64)this->_blocks[a].shape()[0],
                                               (cytnx_int64)tmp_Rtn->_blocks[b].shape()[1]});
                reshaped[targ_b] = true;
                betas[binx] = 0.0;
              }
              // prepare to call gemm_batch
              if (false and (tmp->dtype() <= 4 and this->dtype() <= 4 and tmp_Rtn->dtype() <= 4) and
                  (tmp->dtype() != Type.Void and this->dtype() != Type.Void and
                   tmp_Rtn->dtype() != Type.Void)) {
                ms[binx] = this->_blocks[a].shape()[0];
                ns[binx] = tmp_Rtn->_blocks[b].shape()[1];
                ks[binx] = comm_dim;
                LMems[binx] = this->_blocks[a].storage()._impl->Mem;
                RMems[binx] = tmp_Rtn->_blocks[b].storage()._impl->Mem;
                CMems[binx] = tmp->_blocks[targ_b].storage()._impl->Mem;
              } else {
                tmp->_blocks[targ_b] += linalg::Matmul(this->_blocks[a], tmp_Rtn->_blocks[b])
                                          .reshape(tmp->_blocks[targ_b].shape());
              }
            }
            // mkl_set_interface_layer(MKL_INTERFACE_ILP64);

            blas_int group_count = itoiR_idx.size();
            if (false and (tmp->dtype() <= 4 and this->dtype() <= 4 and tmp_Rtn->dtype() <= 4) and
                (tmp->dtype() != Type.Void and this->dtype() != Type.Void and
                 tmp_Rtn->dtype() != Type.Void)) {
              group_size.resize(group_count, 1);
              linalg::__Gemm_Batch(transs, transs, ms, ns, ks, alphas, (const void **)LMems.data(),
                                   (const void **)RMems.data(), betas, (void **)CMems.data(),
                                   group_count, group_size, this->dtype(), tmp->device());
            }
            // restore the shape&permutation of this->_blocks[a]
            for (cytnx_uint64 binx = 0; binx < itoiR_idx.size(); binx++) {
              cytnx_uint64 b = itoiR_idx[binx];

              tmp_Rtn->_blocks[b].reshape_(oldshapeR[b]);
              tmp_Rtn->_blocks[b].permute_(inv_mapperR);
            }

            this->_blocks[a].reshape_(oldshapeL);
            this->_blocks[a].permute_(inv_mapperL);
          }
          // restore the shape of tmp->_blocks
          for (cytnx_int64 a = 0; a < tmp->_blocks.size(); a++) {
            tmp->_blocks[a].reshape_(oldshapeC[a]);
            if (!reshaped[a]) {
              // if targ_block is not result of any block contraction, set to zeros
              tmp->_blocks[a].storage().set_zeros();
            }
          }

          // if Rtn dtype is casted, delete the tmp_Rtn
          if (this->dtype() != Rtn->dtype()) {
            delete tmp_Rtn;
          }
        }
  #else
          // First select left block to do gemm
          for (cytnx_int64 a = 0; a < this->_blocks.size(); a++) {
            cytnx_int64 comm_dim = 1;
            // get the indices of right blocks that *can* contract with this->_blocks[a]
            itoiR_idx = mp[itoiL_common[a]];
            for (cytnx_uint64 aa = 0; aa < comm_idx1.size(); aa++) {
              comm_dim *= this->_blocks[a].shape()[comm_idx1[aa]];
            }
            // permute&reshape this->_blocks[a]
            this->_blocks[a].permute_(mapperL);
            oldshapeL = this->_blocks[a].shape();
            this->_blocks[a].reshape_({-1, comm_dim});
            // loop over all right blocks that can contract with this->_blocks[a]
            for (cytnx_uint64 binx = 0; binx < itoiR_idx.size(); binx++) {
              // get the index of the right block
              cytnx_uint64 b = itoiR_idx[binx];
              // permute&reshape Rtn->_blocks[b]
              Rtn->_blocks[b].permute_(mapperR);
              oldshapeR[b] = Rtn->_blocks[b].shape();
              Rtn->_blocks[b].reshape_({comm_dim, -1});
              // prepare to find the target block
              Lgbuffer.resize(non_comm_idx1.size() + non_comm_idx2.size());
              for (cytnx_uint64 cc = 0; cc < non_comm_idx1.size(); cc++) {
                Lgbuffer[cc] = this->_inner_to_outer_idx[a][non_comm_idx1[cc]];
              }
              for (cytnx_uint64 cc = non_comm_idx1.size();
                   cc < non_comm_idx1.size() + non_comm_idx2.size(); cc++) {
                Lgbuffer[cc] =
                  Rtn->_inner_to_outer_idx[b][non_comm_idx2[cc - non_comm_idx1.size()]];
              }
              // target block index
              cytnx_int64 targ_b = mpC[Lgbuffer];
              tmp->_blocks[targ_b] += linalg::Matmul(this->_blocks[a], Rtn->_blocks[b])
                                        .reshape(tmp->_blocks[targ_b].shape());
            }
            // restore the shape&permutation of this->_blocks[a]
            for (cytnx_uint64 binx = 0; binx < itoiR_idx.size(); binx++) {
              cytnx_uint64 b = itoiR_idx[binx];

              Rtn->_blocks[b].reshape_(oldshapeR[b]);
              Rtn->_blocks[b].permute_(inv_mapperR);
            }

            this->_blocks[a].reshape_(oldshapeL);
            this->_blocks[a].permute_(inv_mapperL);
          }
          // // restore the shape of tmp->_blocks
          // for(cytnx_int64 a=0;a<tmp->_blocks.size();a++){
          //   tmp->_blocks[a].reshape_(oldshapeC[a]);
          //   if(!reshaped[a]){
          //     // if targ_block is not result of any block contraction, set to zeros
          //     tmp->_blocks[a].storage().set_zeros();
          //   }
          // }
        }
  #endif

        boost::intrusive_ptr<UniTensor_base> out(tmp);
        return out;

      }  // does it contract all the bond?

      cytnx_error_msg(true, "something wrong!%s", "\n");

    }  // does it contract all the bond?
  };

  void BlockUniTensor::Transpose_() {
    // modify tag
    for (int i = 0; i < this->bonds().size(); i++) {
      this->bonds()[i].redirect_();
      // this->bonds()[i].qnums() = this->bonds()[i].calc_reverse_qnums();
    }
  };

  void BlockUniTensor::normalize_() {
    Scalar out(0, this->dtype());
    for (auto &block : this->_blocks) {
      out += Scalar(linalg::Pow(linalg::Norm(block), 2).item());
    }
    out = sqrt(out);
    for (auto &block : this->_blocks) {
      block /= out;
    }
  };

  void BlockUniTensor::Trace_(const std::string &a, const std::string &b) {
    // 1) from label to indx.
    cytnx_int64 ida, idb;

    ida = vec_where(this->_labels, a);
    idb = vec_where(this->_labels, b);

    this->Trace_(ida, idb);
  }
  void BlockUniTensor::Trace_(const cytnx_int64 &a, const cytnx_int64 &b) {
    cytnx_int64 ida = a;
    cytnx_int64 idb = b;

    // check if indices are the same:
    cytnx_error_msg(a < 0 || b < 0, "[ERROR] invalid index a, b%s", "\n");
    cytnx_error_msg(a >= this->rank() || b >= this->rank(), "[ERROR] index out of bound%s", "\n");

    cytnx_error_msg(ida == idb,
                    "[ERROR][BlockUniTensor::Trace_] index a and index b should not be the same.%s",
                    "\n");

    // check if two bonds type are contractable:
    cytnx_error_msg(this->_bonds[ida].type() == this->_bonds[idb].type(),
                    "[ERROR] BD_BRA/BD_OUT can only contract with BD_KET/BD_IN%s", "\n");

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

    // 1) remove the bond, labels:
    if (ida > idb) std::swap(ida, idb);
    this->_bonds.erase(this->_bonds.begin() + idb);
    this->_bonds.erase(this->_bonds.begin() + ida);
    this->_labels.erase(this->_labels.begin() + idb);
    this->_labels.erase(this->_labels.begin() + ida);

    // trace the block!
    std::vector<Tensor> new_blocks;
    vec2d<cytnx_uint64> new_itoi;
    if (this->_labels.size() == 0) {
      // if there is no leg left, leaving only one block, and let API to handle the
      // BlockUniTensor->DenseUniTensor!
      new_blocks.push_back(zeros(1, this->dtype(), this->device()));
      for (cytnx_int64 i = 0; i < this->_blocks.size(); i++) {
        if (this->_inner_to_outer_idx[i][ida] == this->_inner_to_outer_idx[i][idb]) {
          if (this->is_diag())
            new_blocks.back() += linalg::Sum(this->_blocks[i]);
          else
            new_blocks.back() += this->_blocks[i].Trace(ida, idb);
        }
      }

    } else {
      std::map<std::vector<cytnx_uint64>, cytnx_uint64> tmap;
      std::map<std::vector<cytnx_uint64>, cytnx_uint64>::iterator itr;
      for (cytnx_int64 i = 0; i < this->_blocks.size(); i++) {
        // std::cout << "blk: " << i << std::endl;
        if (this->_inner_to_outer_idx[i][ida] == this->_inner_to_outer_idx[i][idb]) {
          auto s = this->_inner_to_outer_idx[i];
          s.erase(s.begin() + idb);
          s.erase(s.begin() + ida);
          auto itr = tmap.find(s);
          if (itr != tmap.end())
            new_blocks[itr->second] += this->_blocks[i].Trace(ida, idb);
          else {
            tmap[s] = new_blocks.size();
            new_blocks.push_back(this->_blocks[i].Trace(ida, idb));
            new_itoi.push_back(s);
          }
        }
      }
    }

    this->_blocks = new_blocks;
    this->_inner_to_outer_idx = new_itoi;
  }

  Tensor BlockUniTensor::Norm() const {
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
  void BlockUniTensor::_fx_locate_elem(cytnx_int64 &bidx, std::vector<cytnx_uint64> &loc_in_T,
                                       const std::vector<cytnx_uint64> &locator) const {
    // 1. check if out of range:
    cytnx_error_msg(locator.size() != this->_bonds.size(),
                    "[ERROR] len(locator) does not match the rank of tensor.%s", "\n");

    for (int i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(locator[i] >= this->_bonds[i].dim(),
                      "[ERROR][BlockUniTensor][elem_exists] locator @index: %d out of range.\n", i);
    }

    // 2. calculate the location is in which qindices:
    if (this->is_diag()) {
      if (locator[0] != locator[1])
        bidx = -1;
      else {
        loc_in_T.push_back(locator[0]);
        std::vector<cytnx_uint64> qindices(2);
        // its diag, so we can just use single bond!
        for (int d = 0; d < this->_bonds[0]._impl->_degs.size(); d++) {
          if (loc_in_T[0] >= this->_bonds[0]._impl->_degs[d])
            loc_in_T[0] -= this->_bonds[0]._impl->_degs[d];
          else {
            qindices[0] = qindices[1] = d;
            break;
          }
        }
        auto it =
          std::find(this->_inner_to_outer_idx.begin(), this->_inner_to_outer_idx.end(), qindices);
        if (it == this->_inner_to_outer_idx.end())
          bidx = -1;
        else
          bidx = it - this->_inner_to_outer_idx.begin();
      }

    } else {
      loc_in_T = locator;
      std::vector<cytnx_uint64> qindices(loc_in_T.size());
      for (int i = 0; i < this->_bonds.size(); i++) {
        for (int d = 0; d < this->_bonds[i]._impl->_degs.size(); d++) {
          if (loc_in_T[i] >= this->_bonds[i]._impl->_degs[d])
            loc_in_T[i] -= this->_bonds[i]._impl->_degs[d];
          else {
            qindices[i] = d;
            break;
          }
        }
      }

      auto it =
        std::find(this->_inner_to_outer_idx.begin(), this->_inner_to_outer_idx.end(), qindices);

      if (it == this->_inner_to_outer_idx.end())
        bidx = -1;
      else
        bidx = it - this->_inner_to_outer_idx.begin();
    }
  }

  bool BlockUniTensor::elem_exists(const std::vector<cytnx_uint64> &locator) const {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx, loc_in_T, locator);
    return !(bidx < 0);
  }

  //-------------------------------------------
  // at_for_sparse
  Scalar::Sproxy BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator) {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx, loc_in_T, locator);
    if (bidx < 0) {
      return Scalar::Sproxy(this->NullRefTensor.storage()._impl, 0);
    } else {
      return this->_blocks[bidx].at(loc_in_T);
    }
  }
  cytnx_complex128 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                  const cytnx_complex128 &aux) {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx, loc_in_T, locator);
    return this->_blocks[bidx].at<cytnx_complex128>(loc_in_T);
  }
  cytnx_complex64 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                 const cytnx_complex64 &aux) {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx, loc_in_T, locator);
    return this->_blocks[bidx].at<cytnx_complex64>(loc_in_T);
  }
  cytnx_double &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                              const cytnx_double &aux) {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx, loc_in_T, locator);
    return this->_blocks[bidx].at<cytnx_double>(loc_in_T);
  }
  cytnx_float &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                             const cytnx_float &aux) {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx, loc_in_T, locator);
    return this->_blocks[bidx].at<cytnx_float>(loc_in_T);
  }
  cytnx_uint64 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                              const cytnx_uint64 &aux) {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx, loc_in_T, locator);
    return this->_blocks[bidx].at<cytnx_uint64>(loc_in_T);
  }
  cytnx_int64 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                             const cytnx_int64 &aux) {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx, loc_in_T, locator);
    return this->_blocks[bidx].at<cytnx_int64>(loc_in_T);
  }
  cytnx_uint32 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                              const cytnx_uint32 &aux) {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx, loc_in_T, locator);
    return this->_blocks[bidx].at<cytnx_uint32>(loc_in_T);
  }
  cytnx_int32 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                             const cytnx_int32 &aux) {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx, loc_in_T, locator);
    return this->_blocks[bidx].at<cytnx_int32>(loc_in_T);
  }
  cytnx_uint16 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                              const cytnx_uint16 &aux) {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx, loc_in_T, locator);
    return this->_blocks[bidx].at<cytnx_uint16>(loc_in_T);
  }
  cytnx_int16 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                             const cytnx_int16 &aux) {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx, loc_in_T, locator);
    return this->_blocks[bidx].at<cytnx_int16>(loc_in_T);
  }

  const Scalar::Sproxy BlockUniTensor::at_for_sparse(
    const std::vector<cytnx_uint64> &locator) const {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx, loc_in_T, locator);
    if (bidx < 0) {
      return Scalar::Sproxy(this->NullRefTensor.storage()._impl, 0);
    } else {
      return this->_blocks[bidx].at(loc_in_T);
    }
  }
  const cytnx_complex128 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                        const cytnx_complex128 &aux) const {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx, loc_in_T, locator);
    return this->_blocks[bidx].at<cytnx_complex128>(loc_in_T);
  }
  const cytnx_complex64 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                       const cytnx_complex64 &aux) const {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx, loc_in_T, locator);
    return this->_blocks[bidx].at<cytnx_complex64>(loc_in_T);
  }
  const cytnx_double &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                    const cytnx_double &aux) const {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx, loc_in_T, locator);
    return this->_blocks[bidx].at<cytnx_double>(loc_in_T);
  }
  const cytnx_float &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                   const cytnx_float &aux) const {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx, loc_in_T, locator);
    return this->_blocks[bidx].at<cytnx_float>(loc_in_T);
  }
  const cytnx_uint64 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                    const cytnx_uint64 &aux) const {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx, loc_in_T, locator);
    return this->_blocks[bidx].at<cytnx_uint64>(loc_in_T);
  }
  const cytnx_int64 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                   const cytnx_int64 &aux) const {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx, loc_in_T, locator);
    return this->_blocks[bidx].at<cytnx_int64>(loc_in_T);
  }
  const cytnx_uint32 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                    const cytnx_uint32 &aux) const {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx, loc_in_T, locator);
    return this->_blocks[bidx].at<cytnx_uint32>(loc_in_T);
  }
  const cytnx_int32 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                   const cytnx_int32 &aux) const {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx, loc_in_T, locator);
    return this->_blocks[bidx].at<cytnx_int32>(loc_in_T);
  }
  const cytnx_uint16 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                    const cytnx_uint16 &aux) const {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx, loc_in_T, locator);
    return this->_blocks[bidx].at<cytnx_uint16>(loc_in_T);
  }
  const cytnx_int16 &BlockUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                   const cytnx_int16 &aux) const {
    cytnx_int64 bidx;
    std::vector<cytnx_uint64> loc_in_T;
    this->_fx_locate_elem(bidx, loc_in_T, locator);
    return this->_blocks[bidx].at<cytnx_int16>(loc_in_T);
  }

  void BlockUniTensor::_save_dispatch(std::fstream &f) const {
    // cytnx_error_msg(true,"[ERROR] Save for SparseUniTensor is under developing!!%s","\n");

    cytnx_uint64 Nblocks = this->_blocks.size();
    f.write((char *)&Nblocks, sizeof(cytnx_uint64));

    // save inner_to_outer_idx:
    for (unsigned int b = 0; b < Nblocks; b++) {
      f.write((char *)&this->_inner_to_outer_idx[b][0], sizeof(cytnx_uint64) * this->_bonds.size());
    }
    for (unsigned int i = 0; i < this->_blocks.size(); i++) {
      this->_blocks[i]._Save(f);
    }
  }

  void BlockUniTensor::_load_dispatch(std::fstream &f) {
    // cytnx_error_msg(true,"[ERROR] Save for SparseUniTensor is under developing!!%s","\n");

    cytnx_uint64 Nblocks;
    f.read((char *)&Nblocks, sizeof(cytnx_uint64));

    this->_inner_to_outer_idx = std::vector<std::vector<cytnx_uint64>>(
      Nblocks, std::vector<cytnx_uint64>(this->_bonds.size()));
    // read inner_to_outer_idx:
    for (unsigned int b = 0; b < Nblocks; b++) {
      f.read((char *)&this->_inner_to_outer_idx[b][0], sizeof(cytnx_uint64) * this->_bonds.size());
    }
    this->_blocks.resize(Nblocks);

    for (unsigned int i = 0; i < this->_blocks.size(); i++) {
      this->_blocks[i]._Load(f);
    }
  }

  void BlockUniTensor::truncate_(const cytnx_int64 &bond_idx, const cytnx_uint64 &q_index) {
    cytnx_error_msg(
      this->is_diag(),
      "[ERROR][BlockUniTensor][truncate_] cannot use truncate_ when is_diag() = true.%s", "\n");
    cytnx_int64 bidx = bond_idx;

    cytnx_error_msg((bidx >= this->_labels.size()) || (bidx < 0),
                    "[ERROR][BlockUniTensor][truncate_] bond_idx out of bound.%s", "\n");
    cytnx_error_msg(
      q_index >= this->_bonds[bidx].qnums().size(),
      "[ERROR][BlockUniTensor][truncate_] q_index out of bound @ specify Bond @[%d].\n", bidx);

    cytnx_error_msg(
      this->_bonds[bidx].qnums().size() == 1,
      "[ERROR][BlockUniTensor][truncate_] cannot remove the only qnums on a given Bond!%s", "\n");

    this->_bonds[bidx]._impl->_rm_qnum(q_index);

    // traversal all blocks, find all blocks that need to remove:
    std::vector<cytnx_uint64> locs;
    for (cytnx_int64 b = 0; b < this->_blocks.size(); b++) {
      if (this->_inner_to_outer_idx[b][bidx] == q_index) locs.push_back(b);
    }

    // remove!
    vec_erase_(this->_inner_to_outer_idx, locs);
    vec_erase_(this->_blocks, locs);
  }
  void BlockUniTensor::truncate_(const std::string &bond_idx, const cytnx_uint64 &q_index) {
    auto it = std::find(this->_labels.begin(), this->_labels.end(), bond_idx);
    cytnx_error_msg(it == this->_labels.end(),
                    "[ERROR] label [%s] does not exist in current UniTensor.\n", bond_idx.c_str());

    cytnx_int64 idx = it - this->_labels.begin();
    this->truncate_(idx, q_index);
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

  void BlockUniTensor::Add_(const boost::intrusive_ptr<UniTensor_base> &rhs) {
    // checking Type:
    cytnx_error_msg(rhs->uten_type() != UTenType.Block,
                    "[ERROR] cannot add two UniTensor with different type/format.%s", "\n");

    BlockUniTensor *Rtn = (BlockUniTensor *)rhs.get();

    // 1) check each bond.
    cytnx_error_msg(this->_bonds.size() != Rtn->_bonds.size(),
                    "[ERROR] cannot add two BlockUniTensor with different rank!%s", "\n");
    for (cytnx_int64 i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(
        this->_bonds[i] != Rtn->_bonds[i],
        "[ERROR] Bond @ index: %d does not match. Therefore cannot perform Add of two UniTensor\n",
        i);
    }

    cytnx_error_msg(this->is_diag() != Rtn->is_diag(),
                    "[ERROR] cannot add BlockUniTensor with is_diag=true and is_diag=false.%s",
                    "\n");

    // 2) finding the blocks (they might be not in the same order!
    for (cytnx_int64 b = 0; b < this->_blocks.size(); b++) {
      for (cytnx_int64 a = 0; a < Rtn->_blocks.size(); a++) {
        if (this->_inner_to_outer_idx[b] ==
            Rtn->_inner_to_outer_idx[(b + a) % Rtn->_blocks.size()]) {
          this->_blocks[b] += Rtn->_blocks[(b + a) % Rtn->_blocks.size()];
          break;
        }
      }
    }
  }

  void BlockUniTensor::Mul_(const boost::intrusive_ptr<UniTensor_base> &rhs) {
    // checking Type:
    cytnx_error_msg(rhs->uten_type() != UTenType.Block,
                    "[ERROR] cannot add two UniTensor with different type/format.%s", "\n");

    BlockUniTensor *Rtn = (BlockUniTensor *)rhs.get();

    // 1) check each bond.
    cytnx_error_msg(this->_bonds.size() != Rtn->_bonds.size(),
                    "[ERROR] cannot add two BlockUniTensor with different rank!%s", "\n");
    for (cytnx_int64 i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(
        this->_bonds[i] != Rtn->_bonds[i],
        "[ERROR] Bond @ index: %d does not match. Therefore cannot perform Add of two UniTensor\n",
        i);
    }

    cytnx_error_msg(this->is_diag() != Rtn->is_diag(),
                    "[ERROR] cannot add BlockUniTensor with is_diag=true and is_diag=false.%s",
                    "\n");

    // 2) finding the blocks (they might be not in the same order!
    for (cytnx_int64 b = 0; b < this->_blocks.size(); b++) {
      for (cytnx_int64 a = 0; a < Rtn->_blocks.size(); a++) {
        if (this->_inner_to_outer_idx[b] ==
            Rtn->_inner_to_outer_idx[(b + a) % Rtn->_blocks.size()]) {
          this->_blocks[b] *= Rtn->_blocks[(b + a) % Rtn->_blocks.size()];
          break;
        }
      }
    }
  }

  void BlockUniTensor::Sub_(const boost::intrusive_ptr<UniTensor_base> &rhs) {
    // checking Type:
    cytnx_error_msg(rhs->uten_type() != UTenType.Block,
                    "[ERROR] cannot add two UniTensor with different type/format.%s", "\n");

    BlockUniTensor *Rtn = (BlockUniTensor *)rhs.get();

    // 1) check each bond.
    cytnx_error_msg(this->_bonds.size() != Rtn->_bonds.size(),
                    "[ERROR] cannot add two BlockUniTensor with different rank!%s", "\n");
    for (cytnx_int64 i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(
        this->_bonds[i] != Rtn->_bonds[i],
        "[ERROR] Bond @ index: %d does not match. Therefore cannot perform Add of two UniTensor\n",
        i);
    }

    cytnx_error_msg(this->is_diag() != Rtn->is_diag(),
                    "[ERROR] cannot add BlockUniTensor with is_diag=true and is_diag=false.%s",
                    "\n");

    // 2) finding the blocks (they might be not in the same order!
    for (cytnx_int64 b = 0; b < this->_blocks.size(); b++) {
      for (cytnx_int64 a = 0; a < Rtn->_blocks.size(); a++) {
        if (this->_inner_to_outer_idx[b] ==
            Rtn->_inner_to_outer_idx[(b + a) % Rtn->_blocks.size()]) {
          this->_blocks[b] -= Rtn->_blocks[(b + a) % Rtn->_blocks.size()];
          break;
        }
      }
    }
  }

  void BlockUniTensor::_fx_group_duplicates(
    const std::vector<cytnx_uint64> &dup_bond_idxs,
    const std::vector<std::vector<cytnx_uint64>> &idx_mappers) {
    // checking the bonds that are duplicates
    // auto mod_idxs = dup_bond_idxs; std::sort(mod_idx.begin(),mod_idx.end());

    // generating new inner_to_outer_idx:
    std::vector<std::vector<cytnx_uint64>> tmp_inner_to_outer_idx;

    // process one by one:
    for (cytnx_int64 bn = 0; bn < dup_bond_idxs.size(); bn++) {
      // cout << "BOND:" << dup_bond_idxs[bn] << endl;
      // cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n";
      tmp_inner_to_outer_idx = this->_inner_to_outer_idx;

      for (cytnx_int64 i = 0; i < this->_inner_to_outer_idx.size(); i++) {
        tmp_inner_to_outer_idx[i][dup_bond_idxs[bn]] =
          idx_mappers[bn][this->_inner_to_outer_idx[i][dup_bond_idxs[bn]]];
      }

      std::vector<int> mask(this->_blocks.size());
      std::vector<Tensor> new_blocks;
      std::vector<std::vector<cytnx_uint64>> new_inner_to_outer_idx;

      std::vector<cytnx_uint64> no_combine;  // same for each bond!
      for (cytnx_uint64 i = 0; i < this->rank(); i++) {
        if (i != dup_bond_idxs[bn]) no_combine.push_back(i);
      }

      for (cytnx_int64 b = 0; b < this->_blocks.size(); b++) {
        if (mask[b] == 1) continue;
        mask[b] = 1;
        new_blocks.push_back(this->_blocks[b]);
        new_inner_to_outer_idx.push_back(tmp_inner_to_outer_idx[b]);
        for (cytnx_int64 a = b + 1; a < this->_blocks.size(); a++) {
          if (mask[a] == 1) continue;
          if (tmp_inner_to_outer_idx[a] == tmp_inner_to_outer_idx[b]) {
            // need to combine two!
            // checking which bonds does not need to combine!
            mask[a] = 1;
            /*
            std::cout << "CALL DS:\n";
            std::cout << no_combine << std::endl;
            std::cout << "targ: old/new itoi:\n";
            std::cout << this->_inner_to_outer_idx[b] << std::endl;
            std::cout << tmp_inner_to_outer_idx[b] << std::endl;
            std::cout << "----------\n" << std::endl;
            std::cout << "src: old/new itoi:\n";
            std::cout << this->_inner_to_outer_idx[a] << std::endl;
            std::cout << tmp_inner_to_outer_idx[a] << std::endl;
            std::cout << "----------\n" << std::endl;
            std::cout << new_blocks.back().shape() << std::endl;
            std::cout << this->_blocks[a].shape() << std::endl;
            std::cout << "=============\n" << std::endl;
            */
            new_blocks.back() = linalg::Directsum(new_blocks.back(), this->_blocks[a], no_combine);
          }
        }
      }  // traversal each block!

      this->_blocks = new_blocks;
      this->_inner_to_outer_idx = new_inner_to_outer_idx;
    }
  }

  void BlockUniTensor::group_basis_() {
    std::vector<cytnx_uint64> has_dup;
    std::vector<std::vector<cytnx_uint64>> idx_mappers;
    for (cytnx_uint64 i = 0; i < this->_bonds.size(); i++) {
      if (this->_bonds[i].has_duplicate_qnums()) {
        has_dup.push_back(i);
        idx_mappers.push_back(this->_bonds[i].group_duplicates_());
      }
    }

    // this modify _inner_to_outer_idx and blocks!
    this->_fx_group_duplicates(has_dup, idx_mappers);
  }

  // Deprecated, internal use only
  void BlockUniTensor::combineBonds(const std::vector<cytnx_int64> &indicators, const bool &force) {
    cytnx_error_msg(this->is_diag(),
                    "[ERROR][BlockUniTensor] cannot combineBonds when is_diag = true!%s", "\n");

    cytnx_error_msg(indicators.size() < 2, "[ERROR] the number of bonds to combine must be > 1%s",
                    "\n");
    std::vector<cytnx_int64>::iterator it;
    std::vector<cytnx_int64> idx_mapper;
    idx_mapper.reserve(this->rank());
    // std::vector<cytnx_int64> new_shape_aft_perm;
    // new_shape_aft_perm.reserve(this->rank()-indicators.size()+1);

    // idx_mapper = std::vector<cytnx_uint64>(indicators.begin(), indicators.end());

    cytnx_error_msg(this->_is_diag,
                    "[ERROR] cannot combineBond on a is_diag=True UniTensor. suggestion: try "
                    "UniTensor.to_dense()/to_dense_() first.%s [NOTE] this is BlockUniTensor, so "
                    "currently under developing!\n",
                    "\n");

    // get the mapper:
    int cnt = 0;
    int idor;
    for (int i = 0; i < this->rank(); i++) {
      if (cnt == indicators.size()) {
        idx_mapper.push_back(i);
        // new_shape_aft_perm.push_back(0);
      } else {
        if (std::find(indicators.begin(), indicators.end(), i) == indicators.end()) {
          idx_mapper.push_back(i);
          // new_shape_aft_perm.push_back(0);
        } else {
          if (i == indicators[0]) {
            // new_shape_aft_perm.push_back(-1);
            idor = idx_mapper.size();  // new_shape_aft_perm.size();
            for (int j = 0; j < indicators.size(); j++) idx_mapper.push_back(indicators[j]);
          }
          cnt += 1;
        }
      }
    }
    // std::cout << idx_mapper << std::endl;
    // std::cout << new_shape_aft_perm << std::endl;

    this->permute_(idx_mapper);
    this->contiguous_();

    // group bonds:
    std::vector<Bond> new_bonds;
    std::vector<cytnx_uint64> cb_stride(indicators.size());
    // std::cout << "idor" << idor << std::endl;
    // std::cout << "rank" << this->rank() << std::endl;
    for (int i = 0; i < this->rank(); i++) {
      if (i == idor) {
        Bond tmp = this->_bonds[i];
        cb_stride[0] = this->_bonds[i].qnums().size();
        for (int j = 1; j < indicators.size(); j++) {
          cb_stride[j] = this->_bonds[i + j].qnums().size();
          if (force)
            tmp._impl->force_combineBond_(this->_bonds[i + j]._impl, false);  // no grouping
          else
            tmp.combineBond_(this->_bonds[i + j], false);  // no grouping
        }
        new_bonds.push_back(tmp);
        i += indicators.size() - 1;

      } else {
        new_bonds.push_back(this->_bonds[i]);
      }
    }

    // remove labels:
    this->_labels.erase(this->_labels.begin() + idor + 1,
                        this->_labels.begin() + idor + 1 + indicators.size() - 1);
    this->_bonds = new_bonds;

    // reshape each blocks, and update_inner_to_outer_idx:
    // process stride:
    memcpy(&cb_stride[0], &cb_stride[1], sizeof(cytnx_uint64) * (cb_stride.size() - 1));
    // for(int i=cb_stride.size()-2;i>=0;i--){
    //     cb_stride[i] = cb_stride[i+1];
    // }
    cb_stride.back() = 1;
    for (int i = cb_stride.size() - 2; i >= 0; i--) {
      cb_stride[i] *= cb_stride[i + 1];
    }

    std::vector<cytnx_int64> new_shape;
    new_shape.reserve(this->rank());
    for (int b = 0; b < this->_blocks.size(); b++) {
      new_shape.clear();
      for (int i = 0; i < this->_blocks[b].shape().size(); i++) {
        if (i == idor) {
          i += indicators.size() - 1;
          new_shape.push_back(-1);
        } else {
          new_shape.push_back(this->_blocks[b].shape()[i]);
        }
      }
      this->_blocks[b].reshape_(new_shape);
    }

    // cout<<"AAAAAAAAAAAAAAAAAAAAAAA"<<this->get_qindices(2)<<endl;
    // cout<<"AAAAAAAAAAAAAAAAAAAAAAA"<<this->bonds()<<endl;

    for (int b = 0; b < this->_blocks.size(); b++) {
      this->_inner_to_outer_idx[b][idor] *= cb_stride[0];
      for (int i = idor + 1; i < idor + indicators.size(); i++) {
        this->_inner_to_outer_idx[b][idor] += this->_inner_to_outer_idx[b][i] * cb_stride[i - idor];
      }
      if (idor + indicators.size() < this->_inner_to_outer_idx[b].size()) {
        memcpy(
          &this->_inner_to_outer_idx[b][idor + 1],
          &this->_inner_to_outer_idx[b][idor + indicators.size()],
          sizeof(cytnx_uint64) * (this->_inner_to_outer_idx[b].size() - idor - indicators.size()));
      }
      this->_inner_to_outer_idx[b].resize(this->rank());
    }
    // std::cout << this->_inner_to_outer_idx << std::endl;

    // check rowrank:
    if (this->_rowrank >= this->rank()) this->_rowrank = this->rank();

    this->_is_braket_form = this->_update_braket();

    // cout<<"BBBBBBBBBBBBBBBBBBBBBBB"<<this->get_qindices(2)<<endl;
    // cout<<"BBBBBBBBBBBBBBBBBBBBBBB"<<this->bonds()<<endl;
    // regroup:
    this->group_basis_();
  }

  void BlockUniTensor::combineBond(const std::vector<std::string> &indicators, const bool &force) {
    cytnx_error_msg(indicators.size() < 2, "[ERROR] the number of bonds to combine must be > 1%s",
                    "\n");
    std::vector<std::string>::iterator it;
    std::vector<cytnx_int64> idx_mapper;
    // find the index of label:
    for (cytnx_uint64 i = 0; i < indicators.size(); i++) {
      it = std::find(this->_labels.begin(), this->_labels.end(), indicators[i]);
      cytnx_error_msg(it == this->_labels.end(), "[ERROR] labels not found in current UniTensor%s",
                      "\n");
      idx_mapper.push_back(std::distance(this->_labels.begin(), it));
    }
    this->combineBonds(idx_mapper, force);
  }

  void BlockUniTensor::combineBonds(const std::vector<std::string> &indicators, const bool &force) {
    this->combineBond(indicators, force);
  }

  // Deprecated
  void BlockUniTensor::combineBonds(const std::vector<cytnx_int64> &indicators, const bool &force,
                                    const bool &by_label) {
    cytnx_error_msg(indicators.size() < 2, "[ERROR] the number of bonds to combine must be > 1%s",
                    "\n");
    std::vector<std::string>::iterator it;
    std::vector<cytnx_int64> idx_mapper;
    if (by_label) {
      // find the index of label:
      for (cytnx_uint64 i = 0; i < indicators.size(); i++) {
        it = std::find(this->_labels.begin(), this->_labels.end(), std::to_string(indicators[i]));
        cytnx_error_msg(it == this->_labels.end(),
                        "[ERROR] labels not found in current UniTensor%s", "\n");
        idx_mapper.push_back(std::distance(this->_labels.begin(), it));
      }

    } else {
      idx_mapper = indicators;
    }
    this->combineBonds(idx_mapper, force);
  }

  void _BK_from_DN(BlockUniTensor *ths, DenseUniTensor *rhs, const bool &force) {
    if (!force) {
      // more checking:
      if (int(rhs->bond_(0).type()) != bondType::BD_NONE) {
        for (int i = 0; i < rhs->bonds().size(); i++) {
          cytnx_error_msg(ths->bond_(i).type() != rhs->bond_(i).type(),
                          "[ERROR] conversion DenseUT -> BlockUT cannot be made, because "
                          "force=false, BOTH have directional Bond, and direction mismatch.%s",
                          "\n");
        }
      }
    }

    cytnx_uint64 total_elem = rhs->_block.storage().size();

    std::vector<cytnx_uint64> stride_rhs(rhs->shape().size(), 1);
    for (int i = (rhs->rank() - 2); i >= 0; i--) {
      stride_rhs[i] = stride_rhs[i + 1] * rhs->shape()[i + 1];
    }

    // moving element:
    for (cytnx_uint64 i = 0; i < total_elem; i++) {
      auto cart = c2cartesian(i, stride_rhs);
      auto elem = ths->at_for_sparse(cart);
      if (elem.exists()) {
        elem = rhs->_block.at(cart);
      } else {
        if (!force)
          if (abs(Scalar(rhs->_block.at(cart))) > 1e-14) {
            cytnx_error_msg(true,
                            "[ERROR] force = false, trying to convert DenseUT to BlockUT that "
                            "violate the symmetry structure.%s",
                            "\n");
          }
      }
    }
  }

  void _BK_from_BK(BlockUniTensor *ths, BlockUniTensor *rhs, const bool &force) {
    cytnx_error_msg(true, "[ERROR] BlockUT-> BlockUT not implemented.%s", "\n");
  }

  void BlockUniTensor::from_(const boost::intrusive_ptr<UniTensor_base> &rhs, const bool &force) {
    // checking shape:
    cytnx_error_msg(this->shape() != rhs->shape(), "[ERROR][from_] shape does not match.%s", "\n");

    if (rhs->uten_type() == UTenType.Dense) {
      _BK_from_DN(this, (DenseUniTensor *)(rhs.get()), force);
    } else if (rhs->uten_type() == UTenType.Block) {
      _BK_from_BK(this, (BlockUniTensor *)(rhs.get()), force);
    } else {
      cytnx_error_msg(true, "[ERROR] unsupport conversion of UniTensor from %s => BlockUniTensor\n",
                      UTenType.getname(rhs->uten_type()).c_str());
    }
  }

}  // namespace cytnx

#endif  // BACKEND_TORCH
