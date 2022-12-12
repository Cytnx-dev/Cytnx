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

    // need to maintain the mapper for contiguous for block_form.
    //this->_mapper = utils_internal::range_cpu(this->_bonds.size());
    //this->_inv_mapper = this->_mapper;
    //this->_contiguous = true;

    // Symmetry, initialize memories for blocks.
    /*
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
      "[ERROR][BlockUniTensor] invalid qnums. no common block (qnum) in this setup.%s", "\n");

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
                        "[ERROR][BlockUniTensor] is_diag =True can only apply to UniTensor with "
                        "each block to be a square matrix!\n block[%d] row.dim:[%d] col.dim:[%d]\n",
                        i, rowdim, coldim);
        if (!no_alloc) this->_blocks[i] = zeros({rowdim}, dtype, device);
      } else {
        if (!no_alloc) this->_blocks[i] = zeros({rowdim, coldim}, dtype, device);
      }
    }
    */
  }
  void BlockUniTensor::Init(const std::vector<Bond> &bonds,
                             const std::vector<cytnx_int64> &in_labels, const cytnx_int64 &rowrank,
                             const unsigned int &dtype, const int &device, const bool &is_diag,
                             const bool &no_alloc) {
    vector<string> vs;
    transform(in_labels.begin(), in_labels.end(), vs.begin(),
              [](cytnx_int64 x) -> string { return to_string(x); });
    Init(bonds, vs, rowrank, dtype, device, is_diag, no_alloc);
  }



}  // namespace cytnx
