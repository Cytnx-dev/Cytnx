#include "UniTensor.hpp"
#include "utils/utils.hpp"
#include "linalg/linalg.hpp"
#include "Generator.hpp"
#include <vector>
using namespace std;
namespace cytnx {

  void SparseUniTensor::Init(const std::vector<Bond> &bonds,
                             const std::vector<cytnx_int64> &in_labels, const cytnx_int64 &Rowrank,
                             const unsigned int &dtype, const int &device, const bool &is_diag) {
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
        "[ERROR][SparseUniTensor] All bonds must be tagged for UniTensor with symmetries.%s", "\n");
      // check rank-0 bond:
      cytnx_error_msg(bonds[i].dim() == 0,
                      "[ERROR][SparseUniTensor] All bonds must have dimension >=1%s", "\n");
      // check symmetry and type:
      cytnx_error_msg(bonds[i].Nsym() != N_symmetry,
                      "[ERROR][SparseUniTensor] inconsistant # of symmetry at bond: %d. # of "
                      "symmetry should be %d\n",
                      i, N_symmetry);
      for (cytnx_uint32 n = 0; n < N_symmetry; n++) {
        cytnx_error_msg(bonds[i].syms()[n] != tmpSyms[n],
                        "[ERROR][SparseUniTensor] symmetry mismatch at bond: %d, %s != %s\n", n,
                        bonds[i].syms()[n].stype_str().c_str(), tmpSyms[n].stype_str().c_str());
      }
      N_ket += cytnx_uint32(bonds[i].type() == bondType::BD_KET);
    }

    // check Rowrank:
    cytnx_error_msg((N_ket < 1) || (N_ket > bonds.size() - 1),
                    "[ERROR][SparseUniTensor] must have at least one ket-bond and one bra-bond.%s",
                    "\n");

    if (Rowrank < 0) {
      this->_Rowrank = N_ket;
    } else {
      cytnx_error_msg((Rowrank < 1) || (Rowrank > bonds.size() - 1),
                      "[ERROR][SparseUniTensor] Rowrank must be >=1 and <=rank-1.%s", "\n");
      this->_Rowrank = Rowrank;
      // update braket_form >>>
    }

    // check labels:
    if (in_labels.size() == 0) {
      for (cytnx_int64 i = 0; i < bonds.size(); i++) this->_labels.push_back(i);

    } else {
      // check bonds & labels dim
      cytnx_error_msg(bonds.size() != in_labels.size(), "%s",
                      "[ERROR] labels must have same lenth as # of bonds.");

      std::vector<cytnx_int64> tmp = vec_unique(in_labels);
      cytnx_error_msg(tmp.size() != in_labels.size(),
                      "[ERROR] labels cannot contain duplicated elements.%s", "\n");
      this->_labels = in_labels;
    }
    cytnx_error_msg(
      is_diag,
      "[ERROR][SparseUniTensor] Cannot set is_diag=true when the UniTensor is with symmetry.%s",
      "\n");

    // copy bonds, otherwise it will share objects:
    this->_bonds = vec_clone(bonds);
    this->_is_braket_form = this->_update_braket();

    // need to maintain the mapper for contiguous for block_form.
    this->_mapper = utils_internal::range_cpu(this->_bonds.size());
    this->_inv_mapper = this->_mapper;

    // Symmetry, initialize memories for blocks.
    vector<Bond> tot_bonds = this->getTotalQnums();
    vector<cytnx_uint64> degenerates;
    vector<vector<cytnx_int64>> uniq_bonds_row = tot_bonds[0].getUniqueQnums();
    vector<vector<cytnx_int64>> uniq_bonds_col = tot_bonds[1].getUniqueQnums();

    // get common qnum set of row-col (bra-ket) space.
    this->_blockqnums = vec2d_intersect(uniq_bonds_row, uniq_bonds_col, true, true);
    cytnx_error_msg(
      this->_blockqnums.size() == 0,
      "[ERROR][SparseUniTensor] invalid qnums. no common block (qnum) in this setup.%s", "\n");

    // calculate&init the No. of blocks and their sizes.
    this->_blocks.resize(this->_blockqnums.size());
    cytnx_uint64 rowdim, coldim;
    this->_inner2outer_row.resize(this->_blocks.size());
    this->_inner2outer_col.resize(this->_blocks.size());

    for (cytnx_uint64 i = 0; i < this->_blocks.size(); i++) {
      rowdim = tot_bonds[0].getDegeneracy(this->_blockqnums[i], this->_inner2outer_row[i]);
      coldim = tot_bonds[1].getDegeneracy(this->_blockqnums[i], this->_inner2outer_col[i]);
      for (cytnx_uint64 j = 0; j < this->_inner2outer_row[i].size(); j++) {
        this->_outer2inner_row[this->_inner2outer_row[i][j]] =
          pair<cytnx_uint64, cytnx_uint64>(i, j);
      }

      for (cytnx_uint64 j = 0; j < this->_inner2outer_col[i].size(); j++) {
        this->_outer2inner_col[this->_inner2outer_col[i][j]] =
          pair<cytnx_uint64, cytnx_uint64>(i, j);
      }

      this->_blocks[i].Init({rowdim, coldim}, dtype, device);
    }
  }

  vector<Bond> SparseUniTensor::getTotalQnums(const bool &physical) {
    if (physical) {
      cytnx_error_msg(true, "[Developing!]%s", "\n");

    } else {
      vector<Bond> cb_inbonds = vec_clone(this->_bonds, this->_Rowrank);
      cytnx_uint64 N_sym;
      for (cytnx_uint64 i = 0; i < cb_inbonds.size(); i++) {
        N_sym = cb_inbonds[i].Nsym();

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (cytnx_uint64 d = 0; d < N_sym * cb_inbonds[i].dim(); d++) {
          cb_inbonds[i].qnums()[cytnx_uint64(d / N_sym)][d % N_sym] *=
            cb_inbonds[i].type() * bondType::BD_KET;
        }
      }
      if (cb_inbonds.size() > 1) {
        for (cytnx_uint64 i = 1; i < cb_inbonds.size(); i++) {
          cb_inbonds[0].combineBond_(cb_inbonds[i]);
        }
      }

      vector<Bond> cb_outbonds = vec_clone(this->_bonds, this->_Rowrank, this->_bonds.size());
      for (cytnx_uint64 i = 0; i < cb_outbonds.size(); i++) {
        N_sym = cb_outbonds[i].Nsym();

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (cytnx_uint64 d = 0; d < N_sym * cb_outbonds[i].dim(); d++) {
          cb_outbonds[i].qnums()[cytnx_uint64(d / N_sym)][d % N_sym] *=
            cb_outbonds[i].type() * bondType::BD_BRA;
        }
      }
      if (cb_outbonds.size() > 1) {
        for (cytnx_uint64 i = 1; i < cb_outbonds.size(); i++) {
          cb_outbonds[0].combineBond_(cb_outbonds[i]);
        }
      }

      vector<Bond> out(2);
      out[0] = cb_inbonds[0];
      cb_inbonds.clear();
      out[1] = cb_outbonds[0];
      cb_outbonds.clear();
      return out;
    }
  }
  void SparseUniTensor::permute_(const std::vector<cytnx_int64> &mapper, const cytnx_int64 &Rowrank,
                                 const bool &by_label) {
    std::vector<cytnx_uint64> mapper_u64;
    if (by_label) {
      // cytnx_error_msg(true,"[Developing!]%s","\n");
      std::vector<cytnx_int64>::iterator it;
      for (cytnx_uint64 i = 0; i < mapper.size(); i++) {
        it = std::find(this->_labels.begin(), this->_labels.end(), mapper[i]);
        cytnx_error_msg(it == this->_labels.end(),
                        "[ERROR] label %d does not exist in current UniTensor.\n", mapper[i]);
        mapper_u64.push_back(std::distance(this->_labels.begin(), it));
      }

    } else {
      mapper_u64 = std::vector<cytnx_uint64>(mapper.begin(), mapper.end());
    }

    this->_bonds = vec_map(vec_clone(this->bonds()), mapper_u64);  // this will check validity
    this->_labels = vec_map(this->labels(), mapper_u64);

    std::vector<cytnx_uint64> new_fwdmap(this->_mapper.size());
    std::vector<cytnx_uint64> new_shape(this->_mapper.size());
    std::vector<cytnx_uint64> new_idxmap(this->_mapper.size());

    for (cytnx_uint32 i = 0; i < mapper_u64.size(); i++) {
      if (mapper_u64[i] >= mapper_u64.size()) {
        cytnx_error_msg(1, "%s", "invalid rank index.\n");
      }
      // std::cout << this->_mapper[rnks[i]] << " " << i << std::endl;
      new_idxmap[this->_mapper[mapper_u64[i]]] = i;
      new_fwdmap[i] = this->_mapper[mapper_u64[i]];
    }

    this->_inv_mapper = new_idxmap;
    this->_mapper = new_fwdmap;

    /// checking if permute back to contiguous:
    bool iconti = true;
    for (cytnx_uint32 i = 0; i < mapper_u64.size(); i++) {
      if (new_fwdmap[i] != new_idxmap[i]) {
        iconti = false;
        break;
      }
      if (new_fwdmap[i] != i) {
        iconti = false;
        break;
      }
    }
    this->_contiguous = iconti;

    // check rowrank.
    if (Rowrank >= 0) {
      cytnx_error_msg((Rowrank >= this->_bonds.size()) || (Rowrank <= 0),
                      "[ERROR] Rowrank should >=1 and <= UniTensor.rank-1 for SparseUniTensor "
                      "(UniTensor in blockform)(UniTensor with symmetries).%s",
                      "\n");
      this->_Rowrank = Rowrank;
    }

    // update braket form status.
    this->_is_braket_form = this->_update_braket();
  }

}  // namespace cytnx
