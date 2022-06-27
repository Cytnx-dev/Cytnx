#include "UniTensor.hpp"
#include "utils/utils.hpp"
#include "Generator.hpp"
#include "linalg/linalg.hpp"
#include <vector>
namespace cytnx {

  void DenseUniTensor::Init(const std::vector<Bond> &bonds,
                            const std::vector<cytnx_int64> &in_labels, const cytnx_int64 &Rowrank,
                            const unsigned int &dtype, const int &device, const bool &is_diag) {
    // check for all bonds
    this->_is_tag = false;
    cytnx_uint32 N_ket = 0;
    if (bonds.size() != 0) this->_is_tag = (bonds[0].type() != bondType::BD_REG);
    for (cytnx_uint64 i = 0; i < bonds.size(); i++) {
      // check
      cytnx_error_msg(bonds[i].qnums().size() != 0, "%s",
                      "[ERROR][DenseUniTensor] All bonds must have non symmetries.");
      if (this->_is_tag) {
        cytnx_error_msg(bonds[i].type() == bondType::BD_REG, "%s",
                        "[ERROR][DenseUniTensor] cannot mix tagged bond with un-tagged bond!%s",
                        "\n");
        N_ket += cytnx_uint32(bonds[i].type() == bondType::BD_KET);
      } else {
        cytnx_error_msg(bonds[i].type() != bondType::BD_REG, "%s",
                        "[ERROR][DenseUniTensor] cannot mix tagged bond with un-tagged bond!%s",
                        "\n");
      }
      cytnx_error_msg(bonds[i].dim() == 0, "%s", "[ERROR] All bonds must have dimension >=1");
    }

    // check Rowrank
    if (this->_is_tag) {
      if (Rowrank < 0) {
        this->_Rowrank = N_ket;
      } else {
        cytnx_error_msg(Rowrank >= bonds.size(),
                        "[ERROR] Rowrank cannot exceed total rank of Tensor.%s", "\n");
        this->_Rowrank = Rowrank;
      }
    } else {
      if (bonds.size() == 0)
        this->_Rowrank = 0;
      else {
        cytnx_error_msg(
          Rowrank < 0,
          "[ERROR] initialize a non-symmetry, un-tagged tensor should assign a >=0 Rowrank.%s",
          "\n");
        cytnx_error_msg(Rowrank >= bonds.size(),
                        "[ERROR] Rowrank cannot exceed total rank of Tensor.%s", "\n");
        this->_Rowrank = Rowrank;
      }
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

    if (is_diag) {
      cytnx_error_msg(bonds.size() != 2,
                      "[ERROR] is_diag= ture should have the shape for initializing the UniTensor "
                      "is square, 2-rank tensor.%s",
                      "\n");
      cytnx_error_msg(bonds[0].dim() != bonds[1].dim(),
                      "[ERROR] is_diag= ture should have the shape for initializing the UniTensor "
                      "is square, 2-rank tensor.%s",
                      "\n");
    }

    // copy bonds, otherwise it will share objects:
    this->_bonds = vec_clone(bonds);
    this->_is_braket_form = this->_update_braket();

    // non symmetry, initialize memory.
    if (this->_bonds.size() == 0) {
      // scalar:
      this->_block = zeros({1}, dtype, device);
    } else {
      if (is_diag) {
        this->_block = zeros({_bonds[0].dim()}, dtype, device);
        this->_is_diag = is_diag;
      } else {
        std::vector<cytnx_uint64> _shape(bonds.size());
        for (unsigned int i = 0; i < _shape.size(); i++) _shape[i] = bonds[i].dim();

        this->_block = zeros(_shape, dtype, device);
      }
    }
  }

  boost::intrusive_ptr<UniTensor_base> DenseUniTensor::permute(
    const std::vector<cytnx_int64> &mapper, const cytnx_int64 &Rowrank, const bool &by_label) {
    boost::intrusive_ptr<UniTensor_base> out = this->clone();
    out->permute_(mapper, Rowrank, by_label);
    return out;
  }
  void DenseUniTensor::permute_(const std::vector<cytnx_int64> &mapper, const cytnx_int64 &Rowrank,
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

    if (this->_is_diag) {
      if (Rowrank >= 0) {
        cytnx_error_msg(Rowrank != 1,
                        "[ERROR] Rowrank should be =1 for UniTensor with is_diag=true%s", "\n");
      }
      this->_bonds = vec_map(vec_clone(this->bonds()), mapper_u64);  // this will check validity
      this->_labels = vec_map(this->labels(), mapper_u64);
      this->_is_braket_form = this->_update_braket();

    } else {
      this->_bonds = vec_map(vec_clone(this->bonds()), mapper_u64);  // this will check validity
      this->_labels = vec_map(this->labels(), mapper_u64);
      this->_block.permute_(mapper_u64);
      if (Rowrank >= 0) {
        cytnx_error_msg((Rowrank > this->_bonds.size()) || (Rowrank < 0),
                        "[ERROR] Rowrank cannot exceed the rank of UniTensor, and should be >=0.%s",
                        "\n");
        this->_Rowrank = Rowrank;
      }
      this->_is_braket_form = this->_update_braket();
    }
  };

  void DenseUniTensor::print_diagram(const bool &bond_info) {
    printf("-----------------------%s", "\n");
    printf("tensor Name : %s\n", this->_name.c_str());
    printf("tensor Rank : %d\n", this->_labels.size());
    printf("is_diag     : %s\n", this->_is_diag ? "True" : "False");
    printf("on device   : %s\n", this->device_str().c_str());

    cytnx_uint64 Nin = this->_Rowrank;
    cytnx_uint64 Nout = this->_labels.size() - this->_Rowrank;
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
    if (this->is_tag()) {
      printf("braket_form : %s\n", this->_is_braket_form ? "True" : "False");
      printf("      |ket>               <bra| %s", "\n");
      printf("           ---------------      %s", "\n");
      for (cytnx_uint64 i = 0; i < vl; i++) {
        printf("           |             |     %s", "\n");
        if (i < Nin) {
          if (this->_bonds[i].type() == bondType::BD_KET)
            bks = "> ";
          else
            bks = "<*";
          memset(l, 0, sizeof(char) * 40);
          memset(llbl, 0, sizeof(char) * 40);
          sprintf(l, "%3d %s__", this->_labels[i], bks.c_str());
          sprintf(llbl, "%-3d", this->_bonds[i].dim());
        } else {
          memset(l, 0, sizeof(char) * 40);
          memset(llbl, 0, sizeof(char) * 40);
          sprintf(l, "%s", "        ");
          sprintf(llbl, "%s", "   ");
        }
        if (i < Nout) {
          if (this->_bonds[Nin + i].type() == bondType::BD_KET)
            bks = "*>";
          else
            bks = " <";
          memset(r, 0, sizeof(char) * 40);
          memset(rlbl, 0, sizeof(char) * 40);
          sprintf(r, "__%s %-3d", bks.c_str(), this->_labels[Nin + i]);
          sprintf(rlbl, "%-3d", this->_bonds[Nin + i].dim());
        } else {
          memset(r, 0, sizeof(char) * 40);
          memset(rlbl, 0, sizeof(char) * 40);
          sprintf(r, "%s", "        ");
          sprintf(rlbl, "%s", "   ");
        }
        printf("   %s| %s     %s |%s\n", l, llbl, rlbl, r);
      }
      printf("           |             |     %s", "\n");
      printf("           ---------------     %s", "\n");

    } else {
      printf("            -------------      %s", "\n");
      for (cytnx_uint64 i = 0; i < vl; i++) {
        if (i == 0)
          printf("           /             \\     %s", "\n");
        else
          printf("           |             |     %s", "\n");

        if (i < Nin) {
          bks = "__";
          memset(l, 0, sizeof(char) * 40);
          memset(llbl, 0, sizeof(char) * 40);
          sprintf(l, "%3d %s__", this->_labels[i], bks.c_str());
          sprintf(llbl, "%-3d", this->_bonds[i].dim());
        } else {
          memset(l, 0, sizeof(char) * 40);
          memset(llbl, 0, sizeof(char) * 40);
          sprintf(l, "%s", "        ");
          sprintf(llbl, "%s", "   ");
        }
        if (i < Nout) {
          bks = "__";
          memset(r, 0, sizeof(char) * 40);
          memset(rlbl, 0, sizeof(char) * 40);
          sprintf(r, "__%s %-3d", bks.c_str(), this->_labels[Nin + i]);
          sprintf(rlbl, "%3d", this->_bonds[Nin + i].dim());
        } else {
          memset(r, 0, sizeof(char) * 40);
          memset(rlbl, 0, sizeof(char) * 40);
          sprintf(r, "%s", "        ");
          sprintf(rlbl, "%s", "   ");
        }
        printf("   %s| %s     %s |%s\n", l, llbl, rlbl, r);
      }
      printf("           \\             /     %s", "\n");
      printf("            -------------      %s", "\n");
    }

    if (bond_info) {
      for (cytnx_uint64 i = 0; i < this->_bonds.size(); i++) {
        printf("lbl:%d ", this->_labels[i]);
        std::cout << this->_bonds[i] << std::endl;
      }
    }

    fflush(stdout);
    free(l);
    free(llbl);
    free(r);
    free(rlbl);
  }

  void DenseUniTensor::combineBonds(const std::vector<cytnx_int64> &indicators,
                                    const bool &permute_back, const bool &by_label) {
    cytnx_error_msg(indicators.size() < 2, "[ERROR] the number of bonds to combine must be > 1%s",
                    "\n");
    std::vector<cytnx_int64>::iterator it;
    std::vector<cytnx_uint64> idx_mapper;
    if (by_label) {
      // find the index of label:
      for (cytnx_uint64 i = 0; i < indicators.size(); i++) {
        it = std::find(this->_labels.begin(), this->_labels.end(), indicators[i]);
        cytnx_error_msg(it == this->_labels.end(),
                        "[ERROR] labels not found in current UniTensor%s", "\n");
        idx_mapper.push_back(std::distance(this->_labels.begin(), it));
      }

    } else {
      idx_mapper = std::vector<cytnx_uint64>(indicators.begin(), indicators.end());
    }

    /// first permute the Tensor:
    std::vector<cytnx_uint64> old_shape = this->shape();

    cytnx_error_msg(this->_is_diag,
                    "[ERROR] cannot combineBond on a is_diag=True UniTensor. suggestion: try "
                    "UniTensor.to_dense()/to_dense_() first.%s",
                    "\n");

    if (permute_back) {
      cytnx_uint64 new_Nin = this->_Rowrank;
      //[Fusion tree]>>>
      for (cytnx_uint64 i = 1; i < idx_mapper.size(); i++) {
        if (idx_mapper[i] < this->_Rowrank) new_Nin -= 1;
        this->_bonds[idx_mapper[0]].combineBond_(this->_bonds[idx_mapper[i]]);
      }
      //<<<
      /// create mapper for permute
      std::vector<cytnx_uint64> idx_no_combine = utils_internal::range_cpu(this->_labels.size());
      vec_erase_(idx_no_combine, idx_mapper);

      std::vector<cytnx_uint64> mapper;
      vec_concatenate_(mapper, idx_mapper, idx_no_combine);

      std::vector<cytnx_int64> new_shape;
      new_shape.push_back(-1);
      for (cytnx_uint64 i = 0; i < idx_no_combine.size(); i++)
        new_shape.push_back(this->_bonds[idx_no_combine[i]].dim());

      this->_block.permute_(mapper);

      this->_block.reshape_(new_shape);

      cytnx_int64 f_label = this->_labels[idx_mapper[0]];
      vec_erase_(this->_bonds, std::vector<cytnx_uint64>(idx_mapper.begin() + 1, idx_mapper.end()));
      vec_erase_(this->_labels,
                 std::vector<cytnx_uint64>(idx_mapper.begin() + 1, idx_mapper.end()));
      // permute back>>
      // find index
      cytnx_uint64 x = vec_where(this->_labels, f_label);
      idx_no_combine = utils_internal::range_cpu(1, this->_labels.size());
      idx_no_combine.insert(idx_no_combine.begin() + x, 0);
      this->_block.permute_(idx_no_combine);
      this->_Rowrank = new_Nin;

      if (this->is_tag()) {
        this->_is_braket_form = this->_update_braket();
      }

    } else {
      //[Fusion tree]>>>
      for (cytnx_uint64 i = 1; i < idx_mapper.size(); i++) {
        this->_bonds[idx_mapper[0]].combineBond_(this->_bonds[idx_mapper[i]]);
      }
      //<<<
      std::vector<cytnx_uint64> idx_no_combine = utils_internal::range_cpu(this->_labels.size());
      vec_erase_(idx_no_combine, idx_mapper);

      std::vector<cytnx_uint64> mapper;
      std::vector<cytnx_int64> new_shape;
      if (idx_mapper[0] >= this->_Rowrank) {
        std::vector<Bond> new_bonds;
        std::vector<cytnx_int64> new_labels;
        vec_concatenate_(mapper, idx_no_combine, idx_mapper);

        for (cytnx_uint64 i = 0; i < idx_no_combine.size(); i++) {
          new_shape.push_back(this->_bonds[idx_no_combine[i]].dim());
          new_bonds.push_back(this->_bonds[idx_no_combine[i]]);
          new_labels.push_back(this->_labels[idx_no_combine[i]]);
        }
        new_bonds.push_back(this->_bonds[idx_mapper[0]]);
        new_labels.push_back(this->_labels[idx_mapper[0]]);
        new_shape.push_back(-1);

        this->_block.permute_(mapper);
        this->_block.reshape_(new_shape);

        this->_bonds = new_bonds;
        this->_labels = new_labels;
        this->_Rowrank = this->_labels.size() - 1;

      } else {
        std::vector<Bond> new_bonds;
        std::vector<cytnx_int64> new_labels;
        vec_concatenate_(mapper, idx_mapper, idx_no_combine);

        new_bonds.push_back(this->_bonds[idx_mapper[0]]);
        new_labels.push_back(this->_labels[idx_mapper[0]]);
        new_shape.push_back(-1);
        for (cytnx_uint64 i = 0; i < idx_no_combine.size(); i++) {
          new_shape.push_back(this->_bonds[idx_no_combine[i]].dim());
          new_bonds.push_back(this->_bonds[idx_no_combine[i]]);
          new_labels.push_back(this->_labels[idx_no_combine[i]]);
        }

        this->_block.permute_(mapper);
        this->_block.reshape_(new_shape);

        this->_bonds = new_bonds;
        this->_labels = new_labels;
        this->_Rowrank = 1;
      }

      if (this->is_tag()) {
        this->_is_braket_form = this->_update_braket();
      }
    }  // permute_back
  }

  boost::intrusive_ptr<UniTensor_base> DenseUniTensor::to_dense() {
    cytnx_error_msg(!(this->_is_diag),
                    "[ERROR] to_dense can only operate on UniTensor with is_diag = True.%s", "\n");
    DenseUniTensor *tmp = this->clone_meta();
    tmp->_block = cytnx::linalg::Diag(this->_block);
    tmp->_is_diag = false;
    boost::intrusive_ptr<UniTensor_base> out(tmp);
    return out;
  }

  boost::intrusive_ptr<UniTensor_base> DenseUniTensor::contract(
    const boost::intrusive_ptr<UniTensor_base> &rhs) {
    // checking :
    cytnx_error_msg(rhs->is_blockform(),
                    "[ERROR] cannot contract non-symmetry UniTensor with symmetry UniTensor%s",
                    "\n");
    cytnx_error_msg(this->is_tag() != rhs->is_tag(),
                    "[ERROR] cannot contract tagged UniTensor with untagged UniTensor.%s", "\n");
    // cytnx_error_msg(this->is_diag() != rhs->is_diag(),"[ERROR] cannot contract a diagonal tensor
    // with non-diagonal tensor. [suggestion:] call UniTensor.to_dense/to_dense_ first%s","\n"); get
    // common labels:
    std::vector<cytnx_int64> comm_labels;
    std::vector<cytnx_uint64> comm_idx1, comm_idx2;
    vec_intersect_(comm_labels, this->labels(), rhs->labels(), comm_idx1, comm_idx2);
    // output instance:
    DenseUniTensor *tmp = new DenseUniTensor();

    tmp->_bonds.clear();
    tmp->_labels.clear();

    if (comm_idx1.size() == 0) {
      // process meta
      vec_concatenate_(tmp->_labels, this->labels(), rhs->labels());

      // these two cannot omp parallel, due to intrusive_ptr
      for (cytnx_uint64 i = 0; i < this->_bonds.size(); i++)
        tmp->_bonds.push_back(this->_bonds[i].clone());
      for (cytnx_uint64 i = 0; i < rhs->_bonds.size(); i++)
        tmp->_bonds.push_back(rhs->_bonds[i].clone());

      tmp->_is_tag = this->is_tag();
      tmp->_Rowrank = this->Rowrank() + rhs->Rowrank();

      if ((this->is_diag() == rhs->is_diag()) && this->is_diag()) {
        tmp->_block = linalg::Otimes(this->_block, rhs->get_block_());
        tmp->_block.reshape_({-1});
        tmp->_is_diag = true;
      } else {
        Tensor tmpL, tmpR;
        if (this->is_diag())
          tmpL = linalg::Diag(this->_block);
        else
          tmpL = this->_block;
        if (rhs->is_diag())
          tmpR = linalg::Diag(rhs->get_block_());
        else
          tmpR = rhs->get_block_();  // share view!!

        tmp->_block = linalg::Otimes(tmpL, tmpR);
        tmp->_is_diag = false;
      }
      tmp->_is_braket_form = tmp->_update_braket();

    } else {
      // process meta
      std::vector<cytnx_uint64> non_comm_idx1 =
        vec_erase(utils_internal::range_cpu(this->rank()), comm_idx1);
      std::vector<cytnx_uint64> non_comm_idx2 =
        vec_erase(utils_internal::range_cpu(rhs->rank()), comm_idx2);

      vec_concatenate_(tmp->_labels, vec_clone(this->_labels, non_comm_idx1),
                       vec_clone(rhs->_labels, non_comm_idx2));

      // these two cannot omp parallel, due to intrusive_ptr
      for (cytnx_uint64 i = 0; i < non_comm_idx1.size(); i++)
        tmp->_bonds.push_back(this->_bonds[non_comm_idx1[i]].clone());
      for (cytnx_uint64 i = 0; i < non_comm_idx2.size(); i++)
        tmp->_bonds.push_back(rhs->_bonds[non_comm_idx2[i]].clone());

      tmp->_is_tag = this->is_tag();
      tmp->_Rowrank = this->Rowrank() + rhs->Rowrank();
      for (cytnx_uint64 i = 0; i < comm_idx1.size(); i++)
        if (comm_idx1[i] < this->_Rowrank) tmp->_Rowrank--;
      for (cytnx_uint64 i = 0; i < comm_idx2.size(); i++)
        if (comm_idx2[i] < rhs->_Rowrank) tmp->_Rowrank--;

      if ((this->is_diag() == rhs->is_diag()) && this->is_diag()) {
        // diag x diag:
        tmp->_block = this->_block * rhs->get_block_();
        tmp->_is_diag = true;
      } else {
        // diag x dense:
        Tensor tmpL, tmpR;
        if (this->is_diag())
          tmpL = linalg::Diag(this->_block);
        else
          tmpL = this->_block;
        if (rhs->is_diag())
          tmpR = linalg::Diag(rhs->get_block_());
        else
          tmpR = rhs->get_block_();  // share view!!
        tmp->_block = linalg::Tensordot(tmpL, tmpR, comm_idx1, comm_idx2);
        tmp->_is_diag = false;
      }
      tmp->_is_braket_form = tmp->_update_braket();

    }  // check if no common index

    boost::intrusive_ptr<UniTensor_base> out(tmp);
    return out;
  }

}  // namespace cytnx
