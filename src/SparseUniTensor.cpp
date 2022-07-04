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

#ifdef UNI_OMP
  #include <omp.h>
#endif

using namespace std;
namespace cytnx {
  typedef Accessor ac;
  void SparseUniTensor::Init(const std::vector<Bond> &bonds,
                             const std::vector<cytnx_int64> &in_labels, const cytnx_int64 &rowrank,
                             const unsigned int &dtype, const int &device, const bool &is_diag,
                             const bool &no_alloc) {
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

    // check rowrank:
    cytnx_error_msg((N_ket < 1) || (N_ket > bonds.size() - 1),
                    "[ERROR][SparseUniTensor] must have at least one ket-bond and one bra-bond.%s",
                    "\n");

    if (rowrank < 0) {
      this->_rowrank = N_ket;
      this->_inner_rowrank = N_ket;
    } else {
      cytnx_error_msg((rowrank < 1) || (rowrank > bonds.size() - 1),
                      "[ERROR][SparseUniTensor] rowrank must be >=1 and <=rank-1.%s", "\n");
      this->_rowrank = rowrank;
      this->_inner_rowrank = rowrank;
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

    // cytnx_error_msg(is_diag,"[ERROR][SparseUniTensor] Cannot set is_diag=true when the UniTensor
    // is with symmetry.%s","\n");
    this->_is_diag = is_diag;

    // copy bonds, otherwise it will share objects:
    this->_bonds = vec_clone(bonds);
    this->_is_braket_form = this->_update_braket();

    // need to maintain the mapper for contiguous for block_form.
    this->_mapper = utils_internal::range_cpu(this->_bonds.size());
    this->_inv_mapper = this->_mapper;
    this->_contiguous = true;

    // Symmetry, initialize memories for blocks.
    vector<Bond> tot_bonds = this->getTotalQnums();
    vector<cytnx_uint64> degenerates;
    vector<vector<cytnx_int64>> uniq_bonds_row = tot_bonds[0].getUniqueQnums();
    vector<vector<cytnx_int64>> uniq_bonds_col = tot_bonds[1].getUniqueQnums();
    // vec_print(std::cout,uniq_bonds_row);// << endl;
    // vec_print(std::cout,uniq_bonds_col);// << endl;
    // exit(1);
    // vec_print(std::cout,tot_bonds[0].qnums());
    // vec_print(std::cout,tot_bonds[1].qnums());
    //[DDK]

    // get common qnum set of row-col (bra-ket) space.
    this->_blockqnums = vec2d_intersect(uniq_bonds_row, uniq_bonds_col, false, false);

    cytnx_error_msg(
      this->_blockqnums.size() == 0,
      "[ERROR][SparseUniTensor] invalid qnums. no common block (qnum) in this setup.%s", "\n");

    // vec_print(std::cout,this->_blockqnums);

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

      if (is_diag) {
        // checking if each block are square matrix!:
        cytnx_error_msg(rowdim != coldim,
                        "[ERROR][SparseUniTensor] is_diag =True can only apply to UniTensor with "
                        "each block to be a square matrix!\n block[%d] row.dim:[%d] col.dim:[%d]\n",
                        i, rowdim, coldim);
        if (!no_alloc) this->_blocks[i] = zeros({rowdim}, dtype, device);
      } else {
        if (!no_alloc) this->_blocks[i] = zeros({rowdim, coldim}, dtype, device);
      }
    }
  }

  vector<Bond> SparseUniTensor::getTotalQnums(const bool &physical) {
    if (physical) {
      cytnx_error_msg(true, "[Developing!]%s", "\n");
      return vector<Bond>();
    } else {
      vector<Bond> cb_inbonds = vec_clone(this->_bonds, this->_rowrank);
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
        cb_inbonds[i].set_type(bondType::BD_KET);
      }
      if (cb_inbonds.size() > 1) {
        for (cytnx_uint64 i = 1; i < cb_inbonds.size(); i++) {
          cb_inbonds[0].combineBond_(cb_inbonds[i]);
        }
      }

      vector<Bond> cb_outbonds = vec_clone(this->_bonds, this->_rowrank, this->_bonds.size());
      for (cytnx_uint64 i = 0; i < cb_outbonds.size(); i++) {
        N_sym = cb_outbonds[i].Nsym();

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (cytnx_uint64 d = 0; d < N_sym * cb_outbonds[i].dim(); d++) {
          cb_outbonds[i].qnums()[cytnx_uint64(d / N_sym)][d % N_sym] *=
            cb_outbonds[i].type() * bondType::BD_BRA;
        }
        cb_outbonds[i].set_type(bondType::BD_BRA);
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
  boost::intrusive_ptr<UniTensor_base> SparseUniTensor::permute(
    const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank, const bool &by_label) {
    // boost::intrusive_ptr<UniTensor_base> out = this->clone();
    // out->permute_(mapper,rowrank,by_label);
    // return out;

    SparseUniTensor *out_raw = this->clone_meta(true, true);
    out_raw->_blocks = this->_blocks;  // share content!!

    std::vector<cytnx_uint64> mapper_u64;
    if (by_label) {
      // cytnx_error_msg(true,"[Developing!]%s","\n");
      std::vector<cytnx_int64>::iterator it;
      for (cytnx_uint64 i = 0; i < mapper.size(); i++) {
        it = std::find(out_raw->_labels.begin(), out_raw->_labels.end(), mapper[i]);
        cytnx_error_msg(it == out_raw->_labels.end(),
                        "[ERROR] label %d does not exist in current UniTensor.\n", mapper[i]);
        mapper_u64.push_back(std::distance(out_raw->_labels.begin(), it));
      }

    } else {
      mapper_u64 = std::vector<cytnx_uint64>(mapper.begin(), mapper.end());
    }

    out_raw->_bonds = vec_map(vec_clone(out_raw->bonds()), mapper_u64);  // this will check validity
    out_raw->_labels = vec_map(out_raw->labels(), mapper_u64);

    std::vector<cytnx_uint64> new_fwdmap(out_raw->_mapper.size());
    std::vector<cytnx_uint64> new_shape(out_raw->_mapper.size());
    std::vector<cytnx_uint64> new_idxmap(out_raw->_mapper.size());

    for (cytnx_uint32 i = 0; i < mapper_u64.size(); i++) {
      if (mapper_u64[i] >= mapper_u64.size()) {
        cytnx_error_msg(1, "%s", "invalid rank index.\n");
      }
      // std::cout << this->_mapper[rnks[i]] << " " << i << std::endl;
      new_idxmap[out_raw->_mapper[mapper_u64[i]]] = i;
      new_fwdmap[i] = out_raw->_mapper[mapper_u64[i]];
    }

    out_raw->_inv_mapper = new_idxmap;
    out_raw->_mapper = new_fwdmap;

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
    out_raw->_contiguous = iconti;

    // check rowrank.
    if (rowrank >= 0) {
      cytnx_error_msg((rowrank >= out_raw->_bonds.size()) || (rowrank <= 0),
                      "[ERROR] rowrank should >=1 and <= UniTensor.rank-1 for SparseUniTensor "
                      "(UniTensor in blockform)(UniTensor with symmetries).%s",
                      "\n");
      out_raw->set_rowrank(rowrank);
    }

    // update braket form status.
    out_raw->_is_braket_form = out_raw->_update_braket();

    boost::intrusive_ptr<UniTensor_base> out(out_raw);
    return out;
  };
  void SparseUniTensor::permute_(const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank,
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
    if (this->_inner_rowrank != this->_rowrank) iconti = false;
    this->_contiguous = iconti;

    // check rowrank.
    if (rowrank >= 0) {
      cytnx_error_msg((rowrank >= this->_bonds.size()) || (rowrank <= 0),
                      "[ERROR] rowrank should >=1 and <= UniTensor.rank-1 for SparseUniTensor "
                      "(UniTensor in blockform)(UniTensor with symmetries).%s",
                      "\n");
      // this->_rowrank = rowrank; //only update the outer meta.
      this->set_rowrank(rowrank);
    }

    // update braket form status.
    this->_is_braket_form = this->_update_braket();
  }

  boost::intrusive_ptr<UniTensor_base> SparseUniTensor::relabels(
    const std::vector<cytnx_int64> &new_labels) {
    SparseUniTensor *tmp = this->clone_meta(true, true);
    tmp->_blocks = this->_blocks;
    tmp->set_labels(new_labels);
    boost::intrusive_ptr<UniTensor_base> out(tmp);
    return out;
  }

  boost::intrusive_ptr<UniTensor_base> SparseUniTensor::relabel(const cytnx_int64 &inx,
                                                                const cytnx_int64 &new_label,
                                                                const bool &by_label) {
    SparseUniTensor *tmp = this->clone_meta(true, true);
    tmp->_blocks = this->_blocks;
    tmp->set_label(inx, new_label, by_label);
    boost::intrusive_ptr<UniTensor_base> out(tmp);
    return out;
  }

  std::vector<Symmetry> SparseUniTensor::syms() const { return this->_bonds[0].syms(); }

  void SparseUniTensor::print_diagram(const bool &bond_info) {
    char *buffer = (char *)malloc(256 * sizeof(char));

    sprintf(buffer, "-----------------------%s", "\n");
    std::cout << std::string(buffer);
    sprintf(buffer, "tensor Name : %s\n", this->_name.c_str());
    std::cout << std::string(buffer);
    sprintf(buffer, "tensor Rank : %d\n", this->_labels.size());
    std::cout << std::string(buffer);
    sprintf(buffer, "block_form  : true%s", "\n");
    std::cout << std::string(buffer);
    sprintf(buffer, "contiguous  : %s\n", this->is_contiguous() ? "True" : "False");
    std::cout << std::string(buffer);
    sprintf(buffer, "valid bocks : %d\n", this->_blocks.size());
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
    char *l = (char *)malloc(40 * sizeof(char));
    char *llbl = (char *)malloc(40 * sizeof(char));
    char *r = (char *)malloc(40 * sizeof(char));
    char *rlbl = (char *)malloc(40 * sizeof(char));

    sprintf(buffer, "braket_form : %s\n", this->_is_braket_form ? "True" : "False");
    std::cout << std::string(buffer);
    sprintf(buffer, "        row               col %s", "\n");
    std::cout << std::string(buffer);
    sprintf(buffer, "           ---------------      %s", "\n");
    std::cout << std::string(buffer);
    for (cytnx_uint64 i = 0; i < vl; i++) {
      sprintf(buffer, "           |             |     %s", "\n");
      std::cout << std::string(buffer);
      if (i < Nin) {
        if (this->_bonds[i].type() == bondType::BD_KET)
          bks = " -->";
        else
          bks = "*<--";
        memset(l, 0, sizeof(char) * 40);
        memset(llbl, 0, sizeof(char) * 40);
        sprintf(l, "%3d %s", this->_labels[i], bks.c_str());
        sprintf(llbl, "%-3d", this->_bonds[i].dim());
      } else {
        memset(l, 0, sizeof(char) * 40);
        memset(llbl, 0, sizeof(char) * 40);
        sprintf(l, "%s", "        ");
        sprintf(llbl, "%s", "   ");
      }
      if (i < Nout) {
        if (this->_bonds[Nin + i].type() == bondType::BD_KET)
          bks = "<--*";
        else
          bks = "--> ";
        memset(r, 0, sizeof(char) * 40);
        memset(rlbl, 0, sizeof(char) * 40);
        sprintf(r, "%s %-3d", bks.c_str(), this->_labels[Nin + i]);
        sprintf(rlbl, "%3d", this->_bonds[Nin + i].dim());
      } else {
        memset(r, 0, sizeof(char) * 40);
        memset(rlbl, 0, sizeof(char) * 40);
        sprintf(r, "%s", "        ");
        sprintf(rlbl, "%s", "   ");
      }
      sprintf(buffer, "   %s| %s     %s |%s\n", l, llbl, rlbl, r);
      std::cout << std::string(buffer);
    }
    sprintf(buffer, "           |             |     %s", "\n");
    std::cout << std::string(buffer);
    sprintf(buffer, "           ---------------     %s", "\n");
    std::cout << std::string(buffer);
    sprintf(buffer, "%s", "\n");
    std::cout << std::string(buffer);

    if (bond_info) {
      for (cytnx_uint64 i = 0; i < this->_bonds.size(); i++) {
        sprintf(buffer, "lbl:%d ", this->_labels[i]);
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

  boost::intrusive_ptr<UniTensor_base> SparseUniTensor::contiguous() {
    // cout << "[enter contiguous]" << endl;
    if (this->is_contiguous()) {
      boost::intrusive_ptr<UniTensor_base> out(this);
      return out;
    } else {
      // make new instance
      SparseUniTensor *tmp = new SparseUniTensor();
      tmp->Init(this->_bonds, this->_labels, this->_rowrank, this->dtype(), this->device(),
                this->_is_diag);

      // calculate new inner meta, and copy the element from it.
      if (this->_is_diag) {
        for (unsigned int b = 0; b < this->_blocks.size(); b++) tmp->_blocks[b] = this->_blocks[b];
      } else {
        // vector<Storage> dest_storages;
        // vector<Storage> src_storages;
        // vector<vector<cytnx_uint64>> dest_shapes;
        // vector<vector<cytnx_uint64>> src_shapes;
        // for(unsigned int b=0;b<this->_blocks.size();b++){
        //     dest_storages.push_back(tmp->_blocks[b]._impl->storage());
        //     src_storages.push_back(this->_blocks[b]._impl->storage());
        //     dest_shapes.push_back(tmp->_blocks[b].shape());
        //     src_shapes.push_back(this->_blocks[b].shape());
        // }
        // utils_internal::uii.blocks_mvelems_ii[this->dtype()](dest_storages, src_storages,
        //                                                 dest_shapes, src_shapes,
        //                                                 this->shape(), this->_inner2outer_row,
        //                                                 this->_inner2outer_col,
        //                                                 tmp->_outer2inner_row,
        //                                                 tmp->_outer2inner_col, this->_mapper,
        //                                                 this->_inv_mapper, this->_inner_rowrank,
        //                                                 tmp->_rowrank);

        utils_internal::uii.blocks_mvelems_ii[this->dtype()](
          tmp->_blocks, this->_blocks, this->shape(), this->_inner2outer_row,
          this->_inner2outer_col, tmp->_outer2inner_row, tmp->_outer2inner_col, this->_mapper,
          this->_inv_mapper, this->_inner_rowrank, tmp->_rowrank);
      }  // is diag

      // update comm-meta:
      // tmp->_contiguous = true;
      // tmp->_mapper = utils_internal::range_cpu(cytnx_uint64(this->_bonds.size()));
      // tmp->_inv_mapper = tmp->_mapper;
      // transform to a intr_ptr.
      boost::intrusive_ptr<UniTensor_base> out(tmp);
      return out;
    }
  }
  //=======================================================================
  // at_for_sparse;
  //=======================================================================
  // some helper function:
  std::vector<cytnx_uint64> _locator_to_inner_ij(const std::vector<cytnx_uint64> &locator,
                                                 const std::vector<cytnx_uint64> &current_shape,
                                                 const cytnx_uint64 &inner_rowrank,
                                                 const std::vector<cytnx_uint64> &inv_mapper) {
    // 1. map the locator to the memory layout:
    std::vector<cytnx_uint64> mem_locator = vec_map(locator, inv_mapper);

    // 2. dispatch to row and column part:
    std::vector<cytnx_uint64> row_locator(inner_rowrank),
      col_locator(mem_locator.size() - inner_rowrank);
    memcpy(&row_locator[0], &mem_locator[0], sizeof(cytnx_uint64) * row_locator.size());
    memcpy(&col_locator[0], &mem_locator[inner_rowrank], sizeof(cytnx_uint64) * col_locator.size());

    // 3.
    //  build accumulate index with current memory shape.
    vector<cytnx_uint64> oldshape = vec_map(current_shape, inv_mapper);

    vector<cytnx_uint64> acc_in_old(inner_rowrank), acc_out_old(oldshape.size() - inner_rowrank);
    acc_out_old[acc_out_old.size() - 1] = 1;
    acc_in_old[acc_in_old.size() - 1] = 1;
    for (unsigned int s = 0; s < acc_out_old.size() - 1; s++) {
      acc_out_old[acc_out_old.size() - 2 - s] =
        oldshape.back() * acc_out_old[acc_out_old.size() - 1 - s];
      oldshape.pop_back();
    }
    oldshape.pop_back();
    for (unsigned int s = 0; s < acc_in_old.size() - 1; s++) {
      acc_in_old[acc_in_old.size() - 2 - s] =
        oldshape.back() * acc_in_old[acc_in_old.size() - 1 - s];
      oldshape.pop_back();
    }

    // get row/col for whole memory
    return {cartesian2c(row_locator, acc_in_old), cartesian2c(col_locator, acc_out_old)};
  }

  Scalar::Sproxy SparseUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator) {
    // 1. check if out of range:
    cytnx_error_msg(locator.size() != this->_bonds.size(),
                    "[ERROR] len(locator) does not match the rank of tensor.%s", "\n");
    for (int i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(locator[i] >= this->_bonds[i].dim(),
                      "[ERROR][SparseUniTensor][at] locator @index: %d out of range.\n", i);
    }

    // 2. calculate the location in real memory using meta datas.
    std::vector<cytnx_uint64> ij =
      _locator_to_inner_ij(locator, this->shape(), this->_inner_rowrank, this->_inv_mapper);
    cytnx_uint64 &i = ij[0];
    cytnx_uint64 &j = ij[1];

    // 3. check if the item is there:
    //  if they ref to different block, then the element is invalid (zero)
    auto it1 = this->_outer2inner_row.find(i);
    auto it2 = this->_outer2inner_col.find(j);
    if ((it1 == this->_outer2inner_row.end()) || (it2 == this->_outer2inner_col.end())) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }
    if (it1->second.first != it2->second.first) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }

    cytnx_uint64 block_index = it1->second.first;
    return this->_blocks[block_index].at({it1->second.second, it2->second.second});
  }
  const Scalar::Sproxy SparseUniTensor::at_for_sparse(
    const std::vector<cytnx_uint64> &locator) const {
    // 1. check if out of range:
    cytnx_error_msg(locator.size() != this->_bonds.size(),
                    "[ERROR] len(locator) does not match the rank of tensor.%s", "\n");
    for (int i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(locator[i] >= this->_bonds[i].dim(),
                      "[ERROR][SparseUniTensor][at] locator @index: %d out of range.\n", i);
    }

    // 2. calculate the location in real memory using meta datas.
    std::vector<cytnx_uint64> ij =
      _locator_to_inner_ij(locator, this->shape(), this->_inner_rowrank, this->_inv_mapper);
    cytnx_uint64 &i = ij[0];
    cytnx_uint64 &j = ij[1];

    // 3. check if the item is there:
    //  if they ref to different block, then the element is invalid (zero)
    auto it1 = this->_outer2inner_row.find(i);
    auto it2 = this->_outer2inner_col.find(j);
    if ((it1 == this->_outer2inner_row.end()) || (it2 == this->_outer2inner_col.end())) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }
    if (it1->second.first != it2->second.first) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }

    cytnx_uint64 block_index = it1->second.first;
    return this->_blocks[block_index].at({it1->second.second, it2->second.second});
  }

  cytnx_complex128 &SparseUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                   const cytnx_complex128 &aux) {
    // 1. check if out of range:
    cytnx_error_msg(locator.size() != this->_bonds.size(),
                    "[ERROR] len(locator) does not match the rank of tensor.%s", "\n");
    for (int i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(locator[i] >= this->_bonds[i].dim(),
                      "[ERROR][SparseUniTensor][at] locator @index: %d out of range.\n", i);
    }

    // 2. calculate the location in real memory using meta datas.
    std::vector<cytnx_uint64> ij =
      _locator_to_inner_ij(locator, this->shape(), this->_inner_rowrank, this->_inv_mapper);
    cytnx_uint64 &i = ij[0];
    cytnx_uint64 &j = ij[1];

    // 3. check if the item is there:
    //  if they ref to different block, then the element is invalid (zero)
    auto it1 = this->_outer2inner_row.find(i);
    auto it2 = this->_outer2inner_col.find(j);
    if ((it1 == this->_outer2inner_row.end()) || (it2 == this->_outer2inner_col.end())) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }
    if (it1->second.first != it2->second.first) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }

    cytnx_uint64 block_index = it1->second.first;
    return this->_blocks[block_index].at<cytnx_complex128>(
      {it1->second.second, it2->second.second});
  }
  const cytnx_complex128 &SparseUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                         const cytnx_complex128 &aux) const {
    // 1. check if out of range:
    cytnx_error_msg(locator.size() != this->_bonds.size(),
                    "[ERROR] len(locator) does not match the rank of tensor.%s", "\n");
    for (int i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(locator[i] >= this->_bonds[i].dim(),
                      "[ERROR][SparseUniTensor][at] locator @index: %d out of range.\n", i);
    }

    // 2. calculate the location in real memory using meta datas.
    std::vector<cytnx_uint64> ij =
      _locator_to_inner_ij(locator, this->shape(), this->_inner_rowrank, this->_inv_mapper);
    cytnx_uint64 &i = ij[0];
    cytnx_uint64 &j = ij[1];

    // 3. check if the item is there:
    //  if they ref to different block, then the element is invalid (zero)
    auto it1 = this->_outer2inner_row.find(i);
    auto it2 = this->_outer2inner_col.find(j);
    if ((it1 == this->_outer2inner_row.end()) || (it2 == this->_outer2inner_col.end())) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }
    if (it1->second.first != it2->second.first) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }

    cytnx_uint64 block_index = it1->second.first;
    return this->_blocks[block_index].at<cytnx_complex128>(
      {it1->second.second, it2->second.second});
  }

  //-----------------------------------------
  cytnx_complex64 &SparseUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                  const cytnx_complex64 &aux) {
    // 1. check if out of range:
    cytnx_error_msg(locator.size() != this->_bonds.size(),
                    "[ERROR] len(locator) does not match the rank of tensor.%s", "\n");
    for (int i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(locator[i] >= this->_bonds[i].dim(),
                      "[ERROR][SparseUniTensor][at] locator @index: %d out of range.\n", i);
    }

    // 2. calculate the location in real memory using meta datas.
    std::vector<cytnx_uint64> ij =
      _locator_to_inner_ij(locator, this->shape(), this->_inner_rowrank, this->_inv_mapper);
    cytnx_uint64 &i = ij[0];
    cytnx_uint64 &j = ij[1];

    // 3. check if the item is there:
    //  if they ref to different block, then the element is invalid (zero)
    auto it1 = this->_outer2inner_row.find(i);
    auto it2 = this->_outer2inner_col.find(j);
    if ((it1 == this->_outer2inner_row.end()) || (it2 == this->_outer2inner_col.end())) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }
    if (it1->second.first != it2->second.first) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }

    cytnx_uint64 block_index = it1->second.first;
    return this->_blocks[block_index].at<cytnx_complex64>({it1->second.second, it2->second.second});
  }
  const cytnx_complex64 &SparseUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                        const cytnx_complex64 &aux) const {
    // 1. check if out of range:
    cytnx_error_msg(locator.size() != this->_bonds.size(),
                    "[ERROR] len(locator) does not match the rank of tensor.%s", "\n");
    for (int i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(locator[i] >= this->_bonds[i].dim(),
                      "[ERROR][SparseUniTensor][at] locator @index: %d out of range.\n", i);
    }

    // 2. calculate the location in real memory using meta datas.
    std::vector<cytnx_uint64> ij =
      _locator_to_inner_ij(locator, this->shape(), this->_inner_rowrank, this->_inv_mapper);
    cytnx_uint64 &i = ij[0];
    cytnx_uint64 &j = ij[1];

    // 3. check if the item is there:
    //  if they ref to different block, then the element is invalid (zero)
    auto it1 = this->_outer2inner_row.find(i);
    auto it2 = this->_outer2inner_col.find(j);
    if ((it1 == this->_outer2inner_row.end()) || (it2 == this->_outer2inner_col.end())) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }
    if (it1->second.first != it2->second.first) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }

    cytnx_uint64 block_index = it1->second.first;
    return this->_blocks[block_index].at<cytnx_complex64>({it1->second.second, it2->second.second});
  }
  //-------------------------------------
  cytnx_double &SparseUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                               const cytnx_double &aux) {
    // 1. check if out of range:
    cytnx_error_msg(locator.size() != this->_bonds.size(),
                    "[ERROR] len(locator) does not match the rank of tensor.%s", "\n");
    for (int i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(locator[i] >= this->_bonds[i].dim(),
                      "[ERROR][SparseUniTensor][at] locator @index: %d out of range.\n", i);
    }

    // 2. calculate the location in real memory using meta datas.
    std::vector<cytnx_uint64> ij =
      _locator_to_inner_ij(locator, this->shape(), this->_inner_rowrank, this->_inv_mapper);
    cytnx_uint64 &i = ij[0];
    cytnx_uint64 &j = ij[1];

    // 3. check if the item is there:
    //  if they ref to different block, then the element is invalid (zero)
    auto it1 = this->_outer2inner_row.find(i);
    auto it2 = this->_outer2inner_col.find(j);
    if ((it1 == this->_outer2inner_row.end()) || (it2 == this->_outer2inner_col.end())) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }
    if (it1->second.first != it2->second.first) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }

    cytnx_uint64 block_index = it1->second.first;
    return this->_blocks[block_index].at<cytnx_double>({it1->second.second, it2->second.second});
  }
  const cytnx_double &SparseUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                     const cytnx_double &aux) const {
    // 1. check if out of range:
    cytnx_error_msg(locator.size() != this->_bonds.size(),
                    "[ERROR] len(locator) does not match the rank of tensor.%s", "\n");
    for (int i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(locator[i] >= this->_bonds[i].dim(),
                      "[ERROR][SparseUniTensor][at] locator @index: %d out of range.\n", i);
    }

    // 2. calculate the location in real memory using meta datas.
    std::vector<cytnx_uint64> ij =
      _locator_to_inner_ij(locator, this->shape(), this->_inner_rowrank, this->_inv_mapper);
    cytnx_uint64 &i = ij[0];
    cytnx_uint64 &j = ij[1];

    // 3. check if the item is there:
    //  if they ref to different block, then the element is invalid (zero)
    auto it1 = this->_outer2inner_row.find(i);
    auto it2 = this->_outer2inner_col.find(j);
    if ((it1 == this->_outer2inner_row.end()) || (it2 == this->_outer2inner_col.end())) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }
    if (it1->second.first != it2->second.first) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }

    cytnx_uint64 block_index = it1->second.first;
    return this->_blocks[block_index].at<cytnx_double>({it1->second.second, it2->second.second});
  }
  //--------------------------------------
  cytnx_float &SparseUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                              const cytnx_float &aux) {
    // 1. check if out of range:
    cytnx_error_msg(locator.size() != this->_bonds.size(),
                    "[ERROR] len(locator) does not match the rank of tensor.%s", "\n");
    for (int i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(locator[i] >= this->_bonds[i].dim(),
                      "[ERROR][SparseUniTensor][at] locator @index: %d out of range.\n", i);
    }

    // 2. calculate the location in real memory using meta datas.
    std::vector<cytnx_uint64> ij =
      _locator_to_inner_ij(locator, this->shape(), this->_inner_rowrank, this->_inv_mapper);
    cytnx_uint64 &i = ij[0];
    cytnx_uint64 &j = ij[1];

    // 3. check if the item is there:
    //  if they ref to different block, then the element is invalid (zero)
    auto it1 = this->_outer2inner_row.find(i);
    auto it2 = this->_outer2inner_col.find(j);
    if ((it1 == this->_outer2inner_row.end()) || (it2 == this->_outer2inner_col.end())) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }
    if (it1->second.first != it2->second.first) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }

    cytnx_uint64 block_index = it1->second.first;
    return this->_blocks[block_index].at<cytnx_float>({it1->second.second, it2->second.second});
  }
  const cytnx_float &SparseUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                    const cytnx_float &aux) const {
    // 1. check if out of range:
    cytnx_error_msg(locator.size() != this->_bonds.size(),
                    "[ERROR] len(locator) does not match the rank of tensor.%s", "\n");
    for (int i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(locator[i] >= this->_bonds[i].dim(),
                      "[ERROR][SparseUniTensor][at] locator @index: %d out of range.\n", i);
    }

    // 2. calculate the location in real memory using meta datas.
    std::vector<cytnx_uint64> ij =
      _locator_to_inner_ij(locator, this->shape(), this->_inner_rowrank, this->_inv_mapper);
    cytnx_uint64 &i = ij[0];
    cytnx_uint64 &j = ij[1];

    // 3. check if the item is there:
    //  if they ref to different block, then the element is invalid (zero)
    auto it1 = this->_outer2inner_row.find(i);
    auto it2 = this->_outer2inner_col.find(j);
    if ((it1 == this->_outer2inner_row.end()) || (it2 == this->_outer2inner_col.end())) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }
    if (it1->second.first != it2->second.first) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }

    cytnx_uint64 block_index = it1->second.first;
    return this->_blocks[block_index].at<cytnx_float>({it1->second.second, it2->second.second});
  }
  //--------------------------------------
  cytnx_uint64 &SparseUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                               const cytnx_uint64 &aux) {
    // 1. check if out of range:
    cytnx_error_msg(locator.size() != this->_bonds.size(),
                    "[ERROR] len(locator) does not match the rank of tensor.%s", "\n");
    for (int i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(locator[i] >= this->_bonds[i].dim(),
                      "[ERROR][SparseUniTensor][at] locator @index: %d out of range.\n", i);
    }

    // 2. calculate the location in real memory using meta datas.
    std::vector<cytnx_uint64> ij =
      _locator_to_inner_ij(locator, this->shape(), this->_inner_rowrank, this->_inv_mapper);
    cytnx_uint64 &i = ij[0];
    cytnx_uint64 &j = ij[1];

    // 3. check if the item is there:
    //  if they ref to different block, then the element is invalid (zero)
    auto it1 = this->_outer2inner_row.find(i);
    auto it2 = this->_outer2inner_col.find(j);
    if ((it1 == this->_outer2inner_row.end()) || (it2 == this->_outer2inner_col.end())) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }
    if (it1->second.first != it2->second.first) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }

    cytnx_uint64 block_index = it1->second.first;
    return this->_blocks[block_index].at<cytnx_uint64>({it1->second.second, it2->second.second});
  }
  const cytnx_uint64 &SparseUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                     const cytnx_uint64 &aux) const {
    // 1. check if out of range:
    cytnx_error_msg(locator.size() != this->_bonds.size(),
                    "[ERROR] len(locator) does not match the rank of tensor.%s", "\n");
    for (int i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(locator[i] >= this->_bonds[i].dim(),
                      "[ERROR][SparseUniTensor][at] locator @index: %d out of range.\n", i);
    }

    // 2. calculate the location in real memory using meta datas.
    std::vector<cytnx_uint64> ij =
      _locator_to_inner_ij(locator, this->shape(), this->_inner_rowrank, this->_inv_mapper);
    cytnx_uint64 &i = ij[0];
    cytnx_uint64 &j = ij[1];

    // 3. check if the item is there:
    //  if they ref to different block, then the element is invalid (zero)
    auto it1 = this->_outer2inner_row.find(i);
    auto it2 = this->_outer2inner_col.find(j);
    if ((it1 == this->_outer2inner_row.end()) || (it2 == this->_outer2inner_col.end())) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }
    if (it1->second.first != it2->second.first) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }

    cytnx_uint64 block_index = it1->second.first;
    return this->_blocks[block_index].at<cytnx_uint64>({it1->second.second, it2->second.second});
  }
  //--------------------------------------
  cytnx_int64 &SparseUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                              const cytnx_int64 &aux) {
    // 1. check if out of range:
    cytnx_error_msg(locator.size() != this->_bonds.size(),
                    "[ERROR] len(locator) does not match the rank of tensor.%s", "\n");
    for (int i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(locator[i] >= this->_bonds[i].dim(),
                      "[ERROR][SparseUniTensor][at] locator @index: %d out of range.\n", i);
    }

    // 2. calculate the location in real memory using meta datas.
    std::vector<cytnx_uint64> ij =
      _locator_to_inner_ij(locator, this->shape(), this->_inner_rowrank, this->_inv_mapper);
    cytnx_uint64 &i = ij[0];
    cytnx_uint64 &j = ij[1];

    // 3. check if the item is there:
    //  if they ref to different block, then the element is invalid (zero)
    auto it1 = this->_outer2inner_row.find(i);
    auto it2 = this->_outer2inner_col.find(j);
    if ((it1 == this->_outer2inner_row.end()) || (it2 == this->_outer2inner_col.end())) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }
    if (it1->second.first != it2->second.first) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }

    cytnx_uint64 block_index = it1->second.first;
    return this->_blocks[block_index].at<cytnx_int64>({it1->second.second, it2->second.second});
  }
  const cytnx_int64 &SparseUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                    const cytnx_int64 &aux) const {
    // 1. check if out of range:
    cytnx_error_msg(locator.size() != this->_bonds.size(),
                    "[ERROR] len(locator) does not match the rank of tensor.%s", "\n");
    for (int i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(locator[i] >= this->_bonds[i].dim(),
                      "[ERROR][SparseUniTensor][at] locator @index: %d out of range.\n", i);
    }

    // 2. calculate the location in real memory using meta datas.
    std::vector<cytnx_uint64> ij =
      _locator_to_inner_ij(locator, this->shape(), this->_inner_rowrank, this->_inv_mapper);
    cytnx_uint64 &i = ij[0];
    cytnx_uint64 &j = ij[1];

    // 3. check if the item is there:
    //  if they ref to different block, then the element is invalid (zero)
    auto it1 = this->_outer2inner_row.find(i);
    auto it2 = this->_outer2inner_col.find(j);
    if ((it1 == this->_outer2inner_row.end()) || (it2 == this->_outer2inner_col.end())) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }
    if (it1->second.first != it2->second.first) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }

    cytnx_uint64 block_index = it1->second.first;
    return this->_blocks[block_index].at<cytnx_int64>({it1->second.second, it2->second.second});
  }
  //--------------------------------------
  cytnx_uint32 &SparseUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                               const cytnx_uint32 &aux) {
    // 1. check if out of range:
    cytnx_error_msg(locator.size() != this->_bonds.size(),
                    "[ERROR] len(locator) does not match the rank of tensor.%s", "\n");
    for (int i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(locator[i] >= this->_bonds[i].dim(),
                      "[ERROR][SparseUniTensor][at] locator @index: %d out of range.\n", i);
    }

    // 2. calculate the location in real memory using meta datas.
    std::vector<cytnx_uint64> ij =
      _locator_to_inner_ij(locator, this->shape(), this->_inner_rowrank, this->_inv_mapper);
    cytnx_uint64 &i = ij[0];
    cytnx_uint64 &j = ij[1];

    // 3. check if the item is there:
    //  if they ref to different block, then the element is invalid (zero)
    auto it1 = this->_outer2inner_row.find(i);
    auto it2 = this->_outer2inner_col.find(j);
    if ((it1 == this->_outer2inner_row.end()) || (it2 == this->_outer2inner_col.end())) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }
    if (it1->second.first != it2->second.first) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }

    cytnx_uint64 block_index = it1->second.first;
    return this->_blocks[block_index].at<cytnx_uint32>({it1->second.second, it2->second.second});
  }
  const cytnx_uint32 &SparseUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                     const cytnx_uint32 &aux) const {
    // 1. check if out of range:
    cytnx_error_msg(locator.size() != this->_bonds.size(),
                    "[ERROR] len(locator) does not match the rank of tensor.%s", "\n");
    for (int i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(locator[i] >= this->_bonds[i].dim(),
                      "[ERROR][SparseUniTensor][at] locator @index: %d out of range.\n", i);
    }

    // 2. calculate the location in real memory using meta datas.
    std::vector<cytnx_uint64> ij =
      _locator_to_inner_ij(locator, this->shape(), this->_inner_rowrank, this->_inv_mapper);
    cytnx_uint64 &i = ij[0];
    cytnx_uint64 &j = ij[1];

    // 3. check if the item is there:
    //  if they ref to different block, then the element is invalid (zero)
    auto it1 = this->_outer2inner_row.find(i);
    auto it2 = this->_outer2inner_col.find(j);
    if ((it1 == this->_outer2inner_row.end()) || (it2 == this->_outer2inner_col.end())) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }
    if (it1->second.first != it2->second.first) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }

    cytnx_uint64 block_index = it1->second.first;
    return this->_blocks[block_index].at<cytnx_uint32>({it1->second.second, it2->second.second});
  }
  //--------------------------------------
  cytnx_int32 &SparseUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                              const cytnx_int32 &aux) {
    // 1. check if out of range:
    cytnx_error_msg(locator.size() != this->_bonds.size(),
                    "[ERROR] len(locator) does not match the rank of tensor.%s", "\n");
    for (int i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(locator[i] >= this->_bonds[i].dim(),
                      "[ERROR][SparseUniTensor][at] locator @index: %d out of range.\n", i);
    }

    // 2. calculate the location in real memory using meta datas.
    std::vector<cytnx_uint64> ij =
      _locator_to_inner_ij(locator, this->shape(), this->_inner_rowrank, this->_inv_mapper);
    cytnx_uint64 &i = ij[0];
    cytnx_uint64 &j = ij[1];

    // 3. check if the item is there:
    //  if they ref to different block, then the element is invalid (zero)
    auto it1 = this->_outer2inner_row.find(i);
    auto it2 = this->_outer2inner_col.find(j);
    if ((it1 == this->_outer2inner_row.end()) || (it2 == this->_outer2inner_col.end())) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }
    if (it1->second.first != it2->second.first) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }

    cytnx_uint64 block_index = it1->second.first;
    return this->_blocks[block_index].at<cytnx_int32>({it1->second.second, it2->second.second});
  }
  const cytnx_int32 &SparseUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                    const cytnx_int32 &aux) const {
    // 1. check if out of range:
    cytnx_error_msg(locator.size() != this->_bonds.size(),
                    "[ERROR] len(locator) does not match the rank of tensor.%s", "\n");
    for (int i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(locator[i] >= this->_bonds[i].dim(),
                      "[ERROR][SparseUniTensor][at] locator @index: %d out of range.\n", i);
    }

    // 2. calculate the location in real memory using meta datas.
    std::vector<cytnx_uint64> ij =
      _locator_to_inner_ij(locator, this->shape(), this->_inner_rowrank, this->_inv_mapper);
    cytnx_uint64 &i = ij[0];
    cytnx_uint64 &j = ij[1];

    // 3. check if the item is there:
    //  if they ref to different block, then the element is invalid (zero)
    auto it1 = this->_outer2inner_row.find(i);
    auto it2 = this->_outer2inner_col.find(j);
    if ((it1 == this->_outer2inner_row.end()) || (it2 == this->_outer2inner_col.end())) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }
    if (it1->second.first != it2->second.first) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }

    cytnx_uint64 block_index = it1->second.first;
    return this->_blocks[block_index].at<cytnx_int32>({it1->second.second, it2->second.second});
  }
  //--------------------------------------
  cytnx_uint16 &SparseUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                               const cytnx_uint16 &aux) {
    // 1. check if out of range:
    cytnx_error_msg(locator.size() != this->_bonds.size(),
                    "[ERROR] len(locator) does not match the rank of tensor.%s", "\n");
    for (int i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(locator[i] >= this->_bonds[i].dim(),
                      "[ERROR][SparseUniTensor][at] locator @index: %d out of range.\n", i);
    }

    // 2. calculate the location in real memory using meta datas.
    std::vector<cytnx_uint64> ij =
      _locator_to_inner_ij(locator, this->shape(), this->_inner_rowrank, this->_inv_mapper);
    cytnx_uint64 &i = ij[0];
    cytnx_uint64 &j = ij[1];

    // 3. check if the item is there:
    //  if they ref to different block, then the element is invalid (zero)
    auto it1 = this->_outer2inner_row.find(i);
    auto it2 = this->_outer2inner_col.find(j);
    if ((it1 == this->_outer2inner_row.end()) || (it2 == this->_outer2inner_col.end())) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }
    if (it1->second.first != it2->second.first) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }

    cytnx_uint64 block_index = it1->second.first;
    return this->_blocks[block_index].at<cytnx_uint16>({it1->second.second, it2->second.second});
  }
  const cytnx_uint16 &SparseUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                     const cytnx_uint16 &aux) const {
    // 1. check if out of range:
    cytnx_error_msg(locator.size() != this->_bonds.size(),
                    "[ERROR] len(locator) does not match the rank of tensor.%s", "\n");
    for (int i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(locator[i] >= this->_bonds[i].dim(),
                      "[ERROR][SparseUniTensor][at] locator @index: %d out of range.\n", i);
    }

    // 2. calculate the location in real memory using meta datas.
    std::vector<cytnx_uint64> ij =
      _locator_to_inner_ij(locator, this->shape(), this->_inner_rowrank, this->_inv_mapper);
    cytnx_uint64 &i = ij[0];
    cytnx_uint64 &j = ij[1];

    // 3. check if the item is there:
    //  if they ref to different block, then the element is invalid (zero)
    auto it1 = this->_outer2inner_row.find(i);
    auto it2 = this->_outer2inner_col.find(j);
    if ((it1 == this->_outer2inner_row.end()) || (it2 == this->_outer2inner_col.end())) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }
    if (it1->second.first != it2->second.first) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }

    cytnx_uint64 block_index = it1->second.first;
    return this->_blocks[block_index].at<cytnx_uint16>({it1->second.second, it2->second.second});
  }
  //--------------------------------------
  cytnx_int16 &SparseUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                              const cytnx_int16 &aux) {
    // 1. check if out of range:
    cytnx_error_msg(locator.size() != this->_bonds.size(),
                    "[ERROR] len(locator) does not match the rank of tensor.%s", "\n");
    for (int i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(locator[i] >= this->_bonds[i].dim(),
                      "[ERROR][SparseUniTensor][at] locator @index: %d out of range.\n", i);
    }

    // 2. calculate the location in real memory using meta datas.
    std::vector<cytnx_uint64> ij =
      _locator_to_inner_ij(locator, this->shape(), this->_inner_rowrank, this->_inv_mapper);
    cytnx_uint64 &i = ij[0];
    cytnx_uint64 &j = ij[1];

    // 3. check if the item is there:
    //  if they ref to different block, then the element is invalid (zero)
    auto it1 = this->_outer2inner_row.find(i);
    auto it2 = this->_outer2inner_col.find(j);
    if ((it1 == this->_outer2inner_row.end()) || (it2 == this->_outer2inner_col.end())) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }
    if (it1->second.first != it2->second.first) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }

    cytnx_uint64 block_index = it1->second.first;
    return this->_blocks[block_index].at<cytnx_int16>({it1->second.second, it2->second.second});
  }
  const cytnx_int16 &SparseUniTensor::at_for_sparse(const std::vector<cytnx_uint64> &locator,
                                                    const cytnx_int16 &aux) const {
    // 1. check if out of range:
    cytnx_error_msg(locator.size() != this->_bonds.size(),
                    "[ERROR] len(locator) does not match the rank of tensor.%s", "\n");
    for (int i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(locator[i] >= this->_bonds[i].dim(),
                      "[ERROR][SparseUniTensor][at] locator @index: %d out of range.\n", i);
    }

    // 2. calculate the location in real memory using meta datas.
    std::vector<cytnx_uint64> ij =
      _locator_to_inner_ij(locator, this->shape(), this->_inner_rowrank, this->_inv_mapper);
    cytnx_uint64 &i = ij[0];
    cytnx_uint64 &j = ij[1];

    // 3. check if the item is there:
    //  if they ref to different block, then the element is invalid (zero)
    auto it1 = this->_outer2inner_row.find(i);
    auto it2 = this->_outer2inner_col.find(j);
    if ((it1 == this->_outer2inner_row.end()) || (it2 == this->_outer2inner_col.end())) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }
    if (it1->second.first != it2->second.first) {
      cytnx_error_msg(true, "[ERROR] trying to access element that doesn't belong to any block.%s",
                      "\n");
    }

    cytnx_uint64 block_index = it1->second.first;
    return this->_blocks[block_index].at<cytnx_int16>({it1->second.second, it2->second.second});
  }
  //================================
  bool SparseUniTensor::elem_exists(const std::vector<cytnx_uint64> &locator) const {
    // 1. check if out of range:
    cytnx_error_msg(locator.size() != this->_bonds.size(),
                    "[ERROR] len(locator) does not match the rank of tensor.%s", "\n");
    for (int i = 0; i < this->_bonds.size(); i++) {
      cytnx_error_msg(locator[i] >= this->_bonds[i].dim(),
                      "[ERROR][SparseUniTensor][elem_exists] locator @index: %d out of range.\n",
                      i);
    }

    // 2. calculate the location in real memory using meta datas.
    std::vector<cytnx_uint64> ij =
      _locator_to_inner_ij(locator, this->shape(), this->_inner_rowrank, this->_inv_mapper);
    cytnx_uint64 &i = ij[0];
    cytnx_uint64 &j = ij[1];

    //[DEBUG]
    /*
    cout << "o2i_row" << endl;
    for(auto it=this->_outer2inner_row.begin();it!=this->_outer2inner_row.end();it++){
        cout << it->first << " " << it->second.first << " " << it->second.second << endl;
    }
    cout << "o2i_col" << endl;
    for(auto it=this->_outer2inner_col.begin();it!=this->_outer2inner_col.end();it++){
        cout << it->first << " " << it->second.first << " " << it->second.second << endl;
    }
    */
    // exit(1);

    // 3. check if the item is there:
    //  if they ref to different block, then the element is invalid (zero)
    if ((this->_outer2inner_row.find(i) == this->_outer2inner_row.end()) ||
        (this->_outer2inner_col.find(j) == this->_outer2inner_col.end())) {
      return false;
    }

    if (this->_outer2inner_row.find(i)->second.first !=
        this->_outer2inner_col.find(j)->second.first) {
      return false;
    } else {
      // cout <<"[true]"<< i << " " << j << endl;
      return true;
    }
  }

  void SparseUniTensor::Transpose_() {
    // permute as usual:
    vector<cytnx_int64> new_order =
      vec_concatenate(vec_range<cytnx_int64>(this->rowrank(), this->rank()),
                      vec_range<cytnx_int64>(0, this->rowrank()));
    this->permute_(new_order, this->rank() - this->rowrank());

    // modify tag, and reverse qnum:
    for (int i = 0; i < this->bonds().size(); i++) {
      this->bonds()[i].set_type((this->bonds()[i].type() == BD_KET) ? BD_BRA : BD_KET);
    }

    // reverse qnum!:
    Bond redundant =
      Bond(this->_blockqnums.size(), BD_KET, this->_blockqnums, this->bonds()[0].syms());
    this->_blockqnums = redundant.calc_reverse_qnums();
    this->_is_braket_form = this->_update_braket();
  }

  Tensor SparseUniTensor::Norm() const {
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

  boost::intrusive_ptr<UniTensor_base> SparseUniTensor::Trace(const cytnx_int64 &a,
                                                              const cytnx_int64 &b,
                                                              const bool &by_label) {
    cytnx_error_msg(
      this->_bonds.size() != 2,
      "[ERROR] SparseUniTensor currently only support Trace on SparseUniTensor with rank-2!%s",
      "\n");
    // std::cout << "entry" << std::endl;
    //[NOTE] Currently ONLY WORK FOR RANK-2 !!

    // 1) from label to indx.
    cytnx_uint64 ida, idb;

    if (by_label) {
      ida = vec_where(this->_labels, a);
      idb = vec_where(this->_labels, b);
    } else {
      cytnx_error_msg(a < 0 || b < 0, "[ERROR] invalid index a, b%s", "\n");
      cytnx_error_msg(a >= this->rank() || b >= this->rank(), "[ERROR] index out of bound%s", "\n");
      ida = a;
      idb = b;
    }

    // check if indices are the same:
    cytnx_error_msg(
      ida == idb, "[ERROR][SparseUniTensor::Trace_] index a and index b should not be the same.%s",
      "\n");

    // check bra-ket if tagged
    if (this->is_braket_form()) {
      // check if it is the same species:
      if (this->_bonds[ida].type() == this->_bonds[idb].type()) {
        cytnx_error_msg(
          true, "[ERROR][SparseUniTensor::Trace_] BD_BRA can only contract with BD_KET.%s", "\n");
      }
    }

    // check dimension:
    cytnx_error_msg(
      this->_bonds[ida].dim() != this->_bonds[idb].dim(),
      "[ERROR][SparseUniTensor::Trace_] The dimension of two bond for trace does not match!%s",
      "\n");

    // check qnums:
    cytnx_error_msg(
      this->_bonds[ida].qnums() != this->_bonds[idb].qnums(),
      "[ERROR][SparseUniTensor::Trace_] The qnums of two bond for trace does not match!%s", "\n");

    // trace the block:
    Tensor t;
    if (this->_is_diag) {
      // we need linalg.Sum
      for (cytnx_int64 i = 0; i < this->_blocks.size(); i++) {
        if (t.dtype() == Type.Void)
          t = linalg::Pow(linalg::Norm(this->_blocks[i]), 2);
        else
          t += linalg::Pow(linalg::Norm(this->_blocks[i]), 2);
      }
    } else {
      for (cytnx_int64 i = 0; i < this->_blocks.size(); i++) {
        if (t.dtype() == Type.Void)
          t = linalg::Trace(this->_blocks[i]);
        else
          t += linalg::Trace(this->_blocks[i]);
      }
    }

    boost::intrusive_ptr<UniTensor_base> out(new DenseUniTensor());
    out->Init_by_Tensor(t, false, 0);
    return out;
  }

  boost::intrusive_ptr<UniTensor_base> SparseUniTensor::contract(
    const boost::intrusive_ptr<UniTensor_base> &rhs, const bool &mv_elem_self,
    const bool &mv_elem_rhs) {
    // cytnx_error_msg(true,"[ERROR][Developing.]%s","\n");

    // checking type
    cytnx_error_msg(rhs->uten_type() != UTenType.Sparse,
                    "[ERROR] cannot contract symmetry UniTensor with non-symmetry UniTensor%s",
                    "\n");

    // checking symmetry:
    cytnx_error_msg(this->syms() != rhs->syms(),
                    "[ERROR] two UniTensor have different symmetry type cannot contract.%s", "\n");

    // get common labels:
    std::vector<cytnx_int64> comm_labels;
    std::vector<cytnx_uint64> comm_idx1, comm_idx2;
    vec_intersect_(comm_labels, this->labels(), rhs->labels(), comm_idx1, comm_idx2);

    // output instance:
    SparseUniTensor *tmp = new SparseUniTensor();

    bool is_scalar_out = false;
    std::vector<cytnx_int64> out_labels;
    std::vector<Bond> out_bonds;
    cytnx_int64 out_rowrank;

    if (comm_idx1.size() == 0) {
      // no common labels:

      cytnx_error_msg(true, "developing%s", "\n");
      /*
      // output labels:
      out_labels.insert(out_labels.end(), this->_labels.begin()
                                        , this->_labels.begin()+this->rowrank());
      out_labels.insert(out_labels.end(), rhs->_labels.begin()
                                        , rhs->_labels.begin() + rhs->rowrank());
      out_labels.insert(out_labels.end(), this->_labels.begin()+this->rowrank()
                                        , this->_labels.end());
      out_labels.insert(out_labels.end(), rhs->_labels.begin() + rhs->rowrank()
                                        , rhs->_labels.end());

      // output bonds:
      out_bonds.insert(out_bonds.end(), this->_bonds.begin()
                                        , this->_bonds.begin()+this->rowrank());
      out_bonds.insert(out_bonds.end(), rhs->_bonds.begin()
                                        , rhs->_bonds.begin() + rhs->rowrank());
      out_bonds.insert(out_bonds.end(), this->_bonds.begin()+this->rowrank()
                                        , this->_bonds.end());
      out_bonds.insert(out_bonds.end(), rhs->_bonds.begin() + rhs->rowrank()
                                        , rhs->_bonds.end());


      out_rowrank = this->rowrank() + rhs->rowrank();


      //cout << "[IN]" << endl;

      boost::intrusive_ptr<UniTensor_base> t_this = this->contiguous();
      boost::intrusive_ptr<UniTensor_base> t_rhs = rhs->contiguous();



      std::vector< std::vector<Tensor> > out_blocks;
      std::map< std::vector<cytnx_int64>, cytnx_uint64 > qns2idx;
      std::map< std::vector<cytnx_int64>, cytnx_uint64 >::iterator it;

      auto Lqns = t_this->get_blocks_qnums();
      auto Rqns = t_rhs->get_blocks_qnums();
      auto Syms = t_this->syms();

      cout << "L=-==================\n";
      cout << Lqns;
      cout << t_this->get_blocks_(true);
      cout << "R=-==================\n";
      cout << Rqns;
      cout << t_rhs->get_blocks_(true);

      if((t_this->is_diag() == t_rhs->is_diag()) && t_this->is_diag()){
          //diag x diag:
          cytnx_error_msg(true,"[developing]%s","\n");

      }else{

          if(this->is_diag()!=rhs->is_diag()){
              // diag x dense:
              cytnx_error_msg(true,"[develope]%s","\n");


          }else{

              for(int l=0;l<Lqns.size();l++){
                  for(int r=0;r<Rqns.size();r++){
                      //std::cout << l << r << endl;
                      std::vector<cytnx_int64> new_qns;

                      //calculate fusion:
                      for(int s=0;s<Syms.size();s++){
                          new_qns.push_back(Syms[s].combine_rule(Lqns[l][s],Rqns[r][s]));
                      }

                      //kron the sub-blocks and store in the out_blocks:
                      auto BLK = linalg::Kron(t_this->get_blocks_(true)[l],
      t_rhs->get_blocks_(true)[r]);

                      it = qns2idx.find(new_qns);
                      if(it == qns2idx.end()){
                          //not exists:
                          std::vector<Tensor> tmp;
                          tmp.push_back(BLK);
                          out_blocks.push_back(tmp);
                          qns2idx[new_qns] = out_blocks.size()-1;
                      }else{
                          out_blocks[it->second].push_back(BLK);
                      }


                  }//r
              }//l

              //cout << "[OK] kron!" << endl;
              //cout << out_blocks << endl;

              //traverse out_blocks and fuse it into a dense matrix:
              for(int i=0;i<out_blocks.size();i++){
                  //cout << i << endl;
                  if(out_blocks[i].size()>1){
                      //merge them!:
                      cytnx_uint64 Row=0;
                      cytnx_uint64 Col=0;
                      for(int j=0;j<out_blocks[i].size();j++){
                          Row += out_blocks[i][j].shape()[0];
                          Col += out_blocks[i][j].shape()[1];
                      }

                      Tensor M = zeros({Row,Col},t_this->dtype(),t_this->device());
                      //cout << "create M susc" << endl;
                      cytnx_uint64 endR=0; Row=0;
                      cytnx_uint64 endC=0; Col=0;
                      //cout << out_blocks[i] << endl;
                      for(int j=0;j<out_blocks[i].size();j++){
                          //cout << j << endl;
                          //offs:
                          endR += out_blocks[i][j].shape()[0];
                          endC += out_blocks[i][j].shape()[1];

                          M( ac::range(Row,endR), ac::range(Col,endC) ) = out_blocks[i][j];

                          Row = endR;
                          Col = endC;

                      }
                      out_blocks[i].clear();
                      out_blocks[i].push_back(M);
                  }
              }


              //now create instance:
              tmp->Init(out_bonds, out_labels, out_rowrank);
              tmp->print_diagram();
              cout << tmp->get_blocks_qnums();
              cout << tmp->get_blocks_();
              //traversal each qnums, get the block and set it:
              for(it = qns2idx.begin(); it != qns2idx.end(); it++){
                  Tensor &T = tmp->get_block_(it->first,true);
                  cout << T.shape() << endl;
                  cout << out_blocks[it->second][0].shape() << endl;
                  cytnx_error_msg(T.shape() !=
      out_blocks[it->second][0].shape(),"[ERROR][FATAL][SparseUniTensor][contract][No comm idx] blk
      size does not match!%s","\n");

                  T = out_blocks[it->second][0];

              }


          }// check diag x dense

      }// check diag x diag
      */

    } else {
      // has common labels:

      // check qnums & type:
      for (int i = 0; i < comm_labels.size(); i++) {
        // std::cout << this->_bonds[comm_idx1[i]];
        // std::cout << rhs->_bonds[comm_idx2[i]];
        if (User_debug)
          cytnx_error_msg(this->_bonds[comm_idx1[i]].qnums() != rhs->_bonds[comm_idx2[i]].qnums(),
                          "[ERROR] contract bond @ label %d have qnum mismatch.\n", comm_labels[i]);

        cytnx_error_msg(this->_bonds[comm_idx1[i]].type() + rhs->_bonds[comm_idx2[i]].type(),
                        "[ERROR] BRA can only contract with KET. invalid @ label: %d\n",
                        comm_labels[i]);
      }
      // std::cout << "end checking" << std::endl;

      // proc meta, labels:
      std::vector<cytnx_uint64> non_comm_idx1 =
        vec_erase(utils_internal::range_cpu(this->rank()), comm_idx1);
      std::vector<cytnx_uint64> non_comm_idx2 =
        vec_erase(utils_internal::range_cpu(rhs->rank()), comm_idx2);

      // checking exception where no non-comm indices on both side:
      is_scalar_out = false;
      if ((non_comm_idx1.size() == 0) && (non_comm_idx2.size() == 0)) {
        // no-common indices
        is_scalar_out = true;

        // find the largest bond:
        auto s1 = this->shape();

        auto it1 = std::max_element(s1.begin(), s1.end());
        auto idx1 = std::distance(s1.begin(), it1);
        auto idx2 =
          std::distance(rhs->labels().begin(),
                        std::find(rhs->labels().begin(), rhs->labels().end(), this->_labels[idx1]));

        auto minlbl = min(*std::min_element(this->labels().begin(), this->labels().end()),
                          *std::min_element(rhs->labels().begin(), rhs->labels().end()));

        auto new_tlabel = this->labels();
        new_tlabel[idx1] = minlbl - 1;
        auto new_rlabel = rhs->labels();
        new_rlabel[idx2] = minlbl - 2;

        // recalculate:
        comm_labels.clear();
        comm_idx1.clear();
        comm_idx2.clear();
        vec_intersect_(comm_labels, new_tlabel, new_rlabel, comm_idx1, comm_idx2);

        non_comm_idx1 = vec_erase(utils_internal::range_cpu(this->rank()), comm_idx1);
        non_comm_idx2 = vec_erase(utils_internal::range_cpu(rhs->rank()), comm_idx2);
        vec_concatenate_(out_labels, vec_clone(new_tlabel, non_comm_idx1),
                         vec_clone(new_rlabel, non_comm_idx2));

      } else {
        vec_concatenate_(out_labels, vec_clone(this->_labels, non_comm_idx1),
                         vec_clone(rhs->_labels, non_comm_idx2));
      }

      // these two cannot omp parallel, due to intrusive_ptr
      for (cytnx_uint64 i = 0; i < non_comm_idx1.size(); i++)
        out_bonds.push_back(this->_bonds[non_comm_idx1[i]]);
      for (cytnx_uint64 i = 0; i < non_comm_idx2.size(); i++)
        out_bonds.push_back(rhs->_bonds[non_comm_idx2[i]]);

      out_rowrank = this->rowrank() + rhs->rowrank();

      for (cytnx_uint64 i = 0; i < comm_idx1.size(); i++)
        if (comm_idx1[i] < this->_rowrank) out_rowrank--;
      for (cytnx_uint64 i = 0; i < comm_idx2.size(); i++)
        if (comm_idx2[i] < rhs->_rowrank) out_rowrank--;

      // permute input:
      std::vector<cytnx_uint64> _mapperL, _mapperR;
      vec_concatenate_(_mapperL, non_comm_idx1, comm_idx1);
      vec_concatenate_(_mapperR, comm_idx2, non_comm_idx2);

      std::vector<cytnx_int64> mapperL(_mapperL.begin(), _mapperL.end());
      std::vector<cytnx_int64> mapperR(_mapperR.begin(), _mapperR.end());
      // std::cout << "end calc permute meta" << std::endl;
      // std::cout << mapperL << std::endl;
      // std::cout << mapperR << std::endl;

      boost::intrusive_ptr<UniTensor_base> t_this =
        this->permute(mapperL, non_comm_idx1.size(), false)->contiguous();
      boost::intrusive_ptr<UniTensor_base> t_rhs =
        rhs->permute(mapperR, comm_labels.size(), false)->contiguous();

      // std::cout << "[INTERNAL]" << std::endl;
      // t_this->print_diagram();
      // t_rhs->print_diagram();

      // std::cout << t_this->is_contiguous();
      // std::cout << t_rhs->is_contiguous();

      // std::cout << "END [INTERNAL]" << std::endl;
      // std::cout << t_this->get_blocks_(true);

      if ((t_this->is_diag() == t_rhs->is_diag()) && t_this->is_diag()) {
        // diag x diag:
        // cytnx_error_msg(true,"[developing]%s","\n");

        tmp->Init(
          out_bonds, out_labels, non_comm_idx1.size(), Type.Double, Device.cpu,
          true);  // type and dev does not matter here, cauz we are going to overwrite them later

        cytnx_error_msg(t_this->get_blocks_(true).size() != t_rhs->get_blocks_(true).size(),
                        "[ERROR] internal fatal, this.blocks.size() != rhs.blocks.size()%s", "\n");

        auto comm_qnums = t_rhs->get_blocks_qnums();
        for (int i = 0; i < comm_qnums.size(); i++) {
          tmp->get_block_(comm_qnums[i], true) =
            t_this->get_block_(comm_qnums[i], true) * t_rhs->get_block_(comm_qnums[i], true);
        }

        for (int i = 0; i < comm_qnums.size(); i++) {
          Tensor &T = tmp->get_block_(comm_qnums[i], true);
          Tensor otmp =
            t_this->get_block_(comm_qnums[i], true) * t_rhs->get_block_(comm_qnums[i], true);
          cytnx_error_msg(otmp.shape() != T.shape(),
                          "[ERROR][internal] invalid shape when assign contract blks @ %d\n", i);
          T = otmp;
        }

      } else {
        if (this->is_diag() != rhs->is_diag()) {
          // diag x dense:
          // Tensor tmpL,tmpR;
          // cytnx_error_msg(true,"[develope]%s","\n");
          tmp->Init(out_bonds, out_labels, non_comm_idx1.size());
          // cytnx_error_msg(t_this->get_blocks_(true).size() !=
          // t_rhs->get_blocks_(true).size(),"[ERROR] internal fatal, this.blocks.size() !=
          // rhs.blocks.size()%s","\n"); auto comm_qnums = t_rhs->get_blocks_qnums(); cout <<
          // t_rhs->get_blocks_qnums() << endl; cout << t_this->get_blocks_qnums() << endl;
          auto comm_qnums = vec2d_intersect(t_rhs->get_blocks_qnums(), t_this->get_blocks_qnums());
          // cout << comm_qnums << endl;

          for (int i = 0; i < comm_qnums.size(); i++) {
            Tensor &T = tmp->get_block_(comm_qnums[i], true);
            Tensor otmp = linalg::Matmul_dg(t_this->get_block_(comm_qnums[i], true),
                                            t_rhs->get_block_(comm_qnums[i], true));
            cytnx_error_msg(otmp.shape() != T.shape(),
                            "[ERROR][internal] invalid shape when assign contract blks @ %d\n", i);
            T = otmp;
          }

        } else {
          // dense x dense:
          // Tensor tmpL,tmpR;
          // tmpL = this->_block;
          // tmpR =  rhs->get_block_(); // share view!!
          // std::cout << "dkd" << std::endl;
          // std::cout << this->_block.shape() << std::endl;
          // std::cout << rhs->get_block_().shape() << std::endl;

          tmp->Init(out_bonds, out_labels, non_comm_idx1.size());
          // std::cout << non_comm_idx1.size() << " " << out_rowrank << std::endl;
          // tmp->print_diagram();
          // std::cout << tmp->get_blocks_qnums();
          // std::vector<Tensor> &Lblk = t_this->get_blocks_(false);
          // std::vector<Tensor> &Rblk = t_rhs->get_blocks_(false);

          // cytnx_error_msg(t_this->get_blocks_(true).size() !=
          // t_rhs->get_blocks_(true).size(),"[ERROR] internal fatal, this.blocks.size() !=
          // rhs.blocks.size()%s","\n");

          // std::cout << Lblk << std::endl;
          // std::cout << Rblk << std::endl;
          // std::cout << tmp->_blocks.size() << std::endl;
          // std::cout << Rblk.size() << std::endl;
          // std::cout << Lblk.size() << std::endl;

          // note, output can have more blocks than the L / R !
          // => get the contract part QNs:
          // cout << t_rhs->get_blocks_qnums() << endl;
          // cout << t_this->get_blocks_qnums() << endl;
          auto comm_qnums = vec2d_intersect(t_rhs->get_blocks_qnums(), t_this->get_blocks_qnums());
          // cout << comm_qnums << endl;

          for (int i = 0; i < comm_qnums.size(); i++) {
            Tensor &T = tmp->get_block_(comm_qnums[i], true);
            Tensor otmp = linalg::Matmul(t_this->get_block_(comm_qnums[i], true),
                                         t_rhs->get_block_(comm_qnums[i], true));
            cytnx_error_msg(otmp.shape() != T.shape(),
                            "[ERROR][internal] invalid shape when assign contract blks @ %d\n", i);
            T = otmp;
          }
        }
        tmp->_is_diag = false;
      }
      tmp->_is_braket_form = tmp->_update_braket();

    }  // check if no common index

    // std::cout << "[OK]" << std::endl;

    if (is_scalar_out)
      return tmp->Trace(0, 1);
    else {
      boost::intrusive_ptr<UniTensor_base> out(tmp);

      // boost::intrusive_ptr<UniTensor_base> out(new UniTensor_base());
      return out;
    }
  }

  void SparseUniTensor::truncate_(const cytnx_int64 &bond_idx, const cytnx_uint64 &dim,
                                  const bool &by_label) {
    cytnx_error_msg(true, "[ERROR] truncate for SparseUniTensor is under developing!!%s", "\n");
  }

  void SparseUniTensor::_save_dispatch(std::fstream &f) const {
    // cytnx_error_msg(true,"[ERROR] Save for SparseUniTensor is under developing!!%s","\n");

    cytnx_uint64 Nblocks = this->_blocks.size();
    f.write((char *)&Nblocks, sizeof(cytnx_uint64));

    for (unsigned int i = 0; i < this->_blocks.size(); i++) {
      this->_blocks[i]._Save(f);
    }
  }

  void SparseUniTensor::_load_dispatch(std::fstream &f) {
    // cytnx_error_msg(true,"[ERROR] Save for SparseUniTensor is under developing!!%s","\n");

    cytnx_uint64 Nblocks;
    f.read((char *)&Nblocks, sizeof(cytnx_uint64));

    this->Init(this->_bonds, this->_labels, this->_rowrank, Type.Double, Device.cpu, false, true);
    cytnx_error_msg(Nblocks != this->_blocks.size(), "[ERROR] corrupted data. SparseUniTensor%s",
                    "\n");

    for (unsigned int i = 0; i < this->_blocks.size(); i++) {
      this->_blocks[i]._Load(f);
    }
  }

  // arithmetic
  void SparseUniTensor::Add_(const boost::intrusive_ptr<UniTensor_base> &rhs) {
    cytnx_error_msg(true, "[ERROR] cannot perform arithmetic + with one of UniTensor is sparse!.%s",
                    "\n");
  }
  void SparseUniTensor::Add_(const Scalar &rhs) {
    cytnx_error_msg(true, "[ERROR] cannot perform arithmetic + btwn Scalar and SparseUniTensor!.%s",
                    "\n");
  }

  void SparseUniTensor::Sub_(const boost::intrusive_ptr<UniTensor_base> &rhs) {
    cytnx_error_msg(true, "[ERROR] cannot perform arithmetic - with one of UniTensor is sparse!.%s",
                    "\n");
  }
  void SparseUniTensor::Sub_(const Scalar &rhs) {
    cytnx_error_msg(true, "[ERROR] cannot perform arithmetic - btwn Scalar and UniTensor!.%s",
                    "\n");
  }
  void SparseUniTensor::lSub_(const Scalar &lhs) {
    cytnx_error_msg(true, "[ERROR] cannot perform arithmetic - btwn Scalar and UniTensor!.%s",
                    "\n");
  }

  void SparseUniTensor::Mul_(const boost::intrusive_ptr<UniTensor_base> &rhs) {
    cytnx_error_msg(true, "[ERROR] cannot perform arithmetic * with one of UniTensor is sparse!.%s",
                    "\n");
  }

  void SparseUniTensor::Mul_(const Scalar &rhs) {
    // cytnx_error_msg(true,"[ERROR] cannot perform arithmetic on all tagged tensor, @spase
    // unitensor%s","\n");
    for (cytnx_int64 i = 0; i < this->_blocks.size(); i++) {
      this->_blocks[i] *= rhs;
    }
  }

  void SparseUniTensor::Div_(const boost::intrusive_ptr<UniTensor_base> &rhs) {
    cytnx_error_msg(true, "[ERROR] cannot arithmetic SparseUniTensor/SparseUniTensor.%s", "\n");
  }
  void SparseUniTensor::Div_(const Scalar &rhs) {
    // cytnx_error_msg(true,"[ERROR] cannot perform arithmetic on all tagged tensor, @spase
    // unitensor%s","\n");
    for (cytnx_int64 i = 0; i < this->_blocks.size(); i++) {
      this->_blocks[i] /= rhs;
    }
  }

  void SparseUniTensor::lDiv_(const Scalar &lhs) {
    cytnx_error_msg(true, "[ERROR] cannot arithmetic Scalar/SparseUniTensor.%s", "\n");
  }

}  // namespace cytnx
