#include "UniTensor.hpp"
#include "utils/utils.hpp"
#include "utils/utils_internal_interface.hpp"
#include "Generator.hpp"
#include "linalg.hpp"
#include <algorithm>
#include <utility>
#include <vector>
typedef cytnx::Accessor ac;
namespace cytnx {

  void DenseUniTensor::Init(const std::vector<Bond> &bonds,
                            const std::vector<cytnx_int64> &in_labels, const cytnx_int64 &rowrank,
                            const unsigned int &dtype, const int &device, const bool &is_diag,
                            const bool &no_alloc) {
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

    // check rowrank
    if (this->_is_tag) {
      if (rowrank < 0) {
        this->_rowrank = N_ket;
      } else {
        cytnx_error_msg(rowrank > bonds.size(),
                        "[ERROR] rowrank cannot exceed total rank of Tensor.%s", "\n");
        this->_rowrank = rowrank;
      }
    } else {
      if (bonds.size() == 0)
        this->_rowrank = 0;
      else {
        cytnx_error_msg(
          rowrank < 0,
          "[ERROR] initialize a non-symmetry, un-tagged tensor should assign a >=0 rowrank.%s",
          "\n");
        cytnx_error_msg(rowrank > bonds.size(),
                        "[ERROR] rowrank cannot exceed total rank of Tensor.%s", "\n");
        this->_rowrank = rowrank;
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
      if (!no_alloc) this->_block = zeros({1}, dtype, device);
    } else {
      if (is_diag) {
        if (!no_alloc) this->_block = zeros({_bonds[0].dim()}, dtype, device);
        this->_is_diag = is_diag;
      } else {
        std::vector<cytnx_uint64> _shape(bonds.size());
        for (unsigned int i = 0; i < _shape.size(); i++) _shape[i] = bonds[i].dim();
        if (!no_alloc) this->_block = zeros(_shape, dtype, device);
      }
    }
  }

  void DenseUniTensor::Init_by_Tensor(const Tensor &in_tensor, const bool &is_diag,
                                      const cytnx_int64 &rowrank) {
    cytnx_error_msg(
      in_tensor.dtype() == Type.Void,
      "[ERROR][Init_by_Tensor] cannot init a UniTensor from an un-initialize Tensor.%s", "\n");

    cytnx_int64 i_rowrank = rowrank;

    if (is_diag) {
      // std::cout << in_tensor.shape() << std::endl;
      cytnx_error_msg(in_tensor.shape().size() != 1,
                      "[ERROR][Init_by_tensor] setting is_diag=True should have input Tensor to be "
                      "rank-1 with diagonal elements.%s",
                      "\n");
      if (rowrank == -1) i_rowrank = 1;
    }

    cytnx_error_msg(rowrank < -1, "[ERROR][Init_by_tensor] rowrank should be >=0%s", "\n");

    // DISABLE SCALAR TENSOR FEATURE [khw can we remove this completely??]
    if (false && in_tensor.storage().size() == 1 && in_tensor.rank() == 1) {
      // scalalr:
      cytnx_error_msg(rowrank != 0,
                      "[ERROR][Init_by_Tensor] detect the input Tensor is a scalar with only one "
                      "element. the rowrank should be =0%s",
                      "\n");
      this->_bonds.clear();
      this->_block = in_tensor;
      this->_labels.clear();
      this->_rowrank = i_rowrank;

    } else {
      if (is_diag) {
        std::vector<Bond> bds(2);
        bds[0] = Bond(in_tensor.shape()[0]);
        bds[1] = bds[0].clone();
        this->_bonds = bds;
        this->_block = in_tensor;
        this->_labels = utils_internal::range_cpu<cytnx_int64>(2);
        cytnx_error_msg(
          i_rowrank != 1,
          "[ERROR][Init_by_tensor] rowrank should be 1 for UniTensor with is_diag=True.%s", "\n");
        this->_rowrank = i_rowrank;
        this->_is_diag = true;

      } else {
        std::vector<Bond> bds;
        for (cytnx_uint64 i = 0; i < in_tensor.shape().size(); i++) {
          bds.push_back(Bond(in_tensor.shape()[i]));
        }
        this->_bonds = bds;
        this->_block = in_tensor;
        this->_labels = utils_internal::range_cpu<cytnx_int64>(in_tensor.shape().size());

        if (i_rowrank == -1) {
          i_rowrank = int(in_tensor.shape().size() / 2);
        }

        cytnx_error_msg(i_rowrank > in_tensor.shape().size(),
                        "[ERROR][Init_by_tensor] rowrank exceed the rank of Tensor.%s", "\n");

        this->_rowrank = i_rowrank;
      }
    }
  }

  boost::intrusive_ptr<UniTensor_base> DenseUniTensor::relabels(
    const std::vector<cytnx_int64> &new_labels) {
    DenseUniTensor *out_raw = this->clone_meta();
    out_raw->_block = this->_block;
    out_raw->set_labels(new_labels);
    boost::intrusive_ptr<UniTensor_base> out(out_raw);
    return out;
  }

  boost::intrusive_ptr<UniTensor_base> DenseUniTensor::relabel(const cytnx_int64 &inx,
                                                               const cytnx_int64 &new_label,
                                                               const bool &by_label) {
    DenseUniTensor *out_raw = this->clone_meta();
    out_raw->_block = this->_block;
    out_raw->set_label(inx, new_label, by_label);
    boost::intrusive_ptr<UniTensor_base> out(out_raw);
    return out;
  }

  boost::intrusive_ptr<UniTensor_base> DenseUniTensor::permute(
    const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank, const bool &by_label) {
    // boost::intrusive_ptr<UniTensor_base> out = this->clone();
    // out->permute_(mapper,rowrank,by_label);
    // return out;
    DenseUniTensor *out_raw = this->clone_meta();
    // boost::intrusive_ptr<UniTensor_base> out(this->clone_meta());

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

    if (out_raw->_is_diag) {
      if (rowrank >= 0) {
        cytnx_error_msg(rowrank != 1,
                        "[ERROR] rowrank should be =1 for UniTensor with is_diag=true%s", "\n");
      }
      out_raw->_bonds =
        vec_map(vec_clone(out_raw->bonds()), mapper_u64);  // this will check validity
      out_raw->_labels = vec_map(out_raw->labels(), mapper_u64);
      out_raw->_is_braket_form = out_raw->_update_braket();
      out_raw->_block = this->_block;  // share mem
    } else {
      out_raw->_bonds =
        vec_map(vec_clone(out_raw->bonds()), mapper_u64);  // this will check validity
      out_raw->_labels = vec_map(out_raw->labels(), mapper_u64);
      out_raw->_block = this->_block.permute(mapper_u64);  // share mem

      if (rowrank >= 0) {
        cytnx_error_msg((rowrank > out_raw->_bonds.size()) || (rowrank < 0),
                        "[ERROR] rowrank cannot exceed the rank of UniTensor, and should be >=0.%s",
                        "\n");
        out_raw->_rowrank = rowrank;
      }
      out_raw->_is_braket_form = out_raw->_update_braket();
    }
    boost::intrusive_ptr<UniTensor_base> out(out_raw);
    return out;
  }
  void DenseUniTensor::permute_(const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank,
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
      if (rowrank >= 0) {
        cytnx_error_msg(rowrank != 1,
                        "[ERROR] rowrank should be =1 for UniTensor with is_diag=true%s", "\n");
      }
      this->_bonds = vec_map(vec_clone(this->bonds()), mapper_u64);  // this will check validity
      this->_labels = vec_map(this->labels(), mapper_u64);
      this->_is_braket_form = this->_update_braket();

    } else {
      this->_bonds = vec_map(vec_clone(this->bonds()), mapper_u64);  // this will check validity
      this->_labels = vec_map(this->labels(), mapper_u64);
      this->_block.permute_(mapper_u64);
      if (rowrank >= 0) {
        cytnx_error_msg((rowrank > this->_bonds.size()) || (rowrank < 0),
                        "[ERROR] rowrank cannot exceed the rank of UniTensor, and should be >=0.%s",
                        "\n");
        this->_rowrank = rowrank;
      }
      this->_is_braket_form = this->_update_braket();
    }
  };

  void DenseUniTensor::print_diagram(const bool &bond_info) {
    char *buffer = (char *)malloc(256 * sizeof(char));

    sprintf(buffer, "-----------------------%s", "\n");
    std::cout << std::string(buffer);
    sprintf(buffer, "tensor Name : %s\n", this->_name.c_str());
    std::cout << std::string(buffer);
    sprintf(buffer, "tensor Rank : %ld\n", this->_labels.size());
    std::cout << std::string(buffer);
    sprintf(buffer, "block_form  : false%s", "\n");
    std::cout << std::string(buffer);
    sprintf(buffer, "is_diag     : %s\n", this->_is_diag ? "True" : "False");
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
    if (this->is_tag()) {
      sprintf(buffer, "braket_form : %s\n", this->_is_braket_form ? "True" : "False");
      std::cout << std::string(buffer);
      sprintf(buffer, "        row               col   %s", "\n");
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
          sprintf(l, "%3ld %s", this->_labels[i], bks.c_str());
          sprintf(llbl, "%-3lu", this->_bonds[i].dim());
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

    } else {
      sprintf(buffer, "            -------------      %s", "\n");
      std::cout << std::string(buffer);
      for (cytnx_uint64 i = 0; i < vl; i++) {
        if (i == 0) {
          sprintf(buffer, "           /             \\     %s", "\n");
          std::cout << std::string(buffer);
        } else {
          sprintf(buffer, "           |             |     %s", "\n");
          std::cout << std::string(buffer);
        }

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
        sprintf(buffer, "   %s| %s     %s |%s\n", l, llbl, rlbl, r);
        std::cout << std::string(buffer);
      }
      sprintf(buffer, "           \\             /     %s", "\n");
      std::cout << std::string(buffer);
      sprintf(buffer, "            -------------      %s", "\n");
      std::cout << std::string(buffer);
    }

    if (bond_info) {
      for (cytnx_uint64 i = 0; i < this->_bonds.size(); i++) {
        sprintf(buffer, "lbl:%ld ", this->_labels[i]);
        std::cout << std::string(buffer);
        std::cout << this->_bonds[i] << std::endl;
      }
    }

    // fflush(stdout);
    free(l);
    free(llbl);
    free(r);
    free(rlbl);
    free(buffer);
  }
  void DenseUniTensor::reshape_(const std::vector<cytnx_int64> &new_shape,
                                const cytnx_uint64 &rowrank) {
    cytnx_error_msg(this->is_tag(),
                    "[ERROR] cannot reshape a tagged UniTensor. suggestion: use untag() first.%s",
                    "\n");
    cytnx_error_msg(rowrank > new_shape.size(),
                    "[ERROR] rowrank cannot larger than the rank of reshaped UniTensor.%s", "\n");
    if (this->is_diag()) {
      // if(new_shape.size()!=2){
      this->_block = cytnx::linalg::Diag(this->_block);
      this->_block.reshape_(new_shape);
      this->Init_by_Tensor(this->_block, false, rowrank);
      //}else{
      //    cytnx_error_msg(new_shape[0]!=new_shape[1],"[ERROR] invalid shape. The total elements
      //    does not match.%s","\n"); cytnx_error_msg(rowrank!=1,"[ERROR] UniTensor with
      //    is_diag=True should have rowrank=1.%s","\n");
      //}
    } else {
      this->_block.reshape_(new_shape);
      this->Init_by_Tensor(this->_block, false, rowrank);
    }
  }
  boost::intrusive_ptr<UniTensor_base> DenseUniTensor::reshape(
    const std::vector<cytnx_int64> &new_shape, const cytnx_uint64 &rowrank) {
    cytnx_error_msg(this->is_tag(),
                    "[ERROR] cannot reshape a tagged UniTensor. suggestion: use untag() first.%s",
                    "\n");
    cytnx_error_msg(rowrank > new_shape.size(),
                    "[ERROR] rowrank cannot larger than the rank of reshaped UniTensor.%s", "\n");

    boost::intrusive_ptr<UniTensor_base> out(new DenseUniTensor());
    if (this->is_diag()) {
      // if(new_shape.size()!=2){
      ((DenseUniTensor *)out.get())->_block = cytnx::linalg::Diag(this->_block);
      ((DenseUniTensor *)out.get())->_block.reshape_(new_shape);
      out->Init_by_Tensor(((DenseUniTensor *)out.get())->_block, false, rowrank);
      //}else{
      //    cytnx_error_msg(new_shape[0]!=new_shape[1],"[ERROR] invalid shape. The total elements
      //    does not match.%s","\n"); cytnx_error_msg(rowrank!=1,"[ERROR] UniTensor with
      //    is_diag=True should have rowrank=1.%s","\n"); out = this->clone();
      //}
    } else {
      out->Init_by_Tensor(this->_block.reshape(new_shape), false, rowrank);
    }
    return out;
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
      cytnx_uint64 new_Nin = this->_rowrank;
      //[Fusion tree]>>>
      for (cytnx_uint64 i = 1; i < idx_mapper.size(); i++) {
        if (idx_mapper[i] < this->_rowrank) new_Nin -= 1;
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
      this->_rowrank = new_Nin;

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
      if (idx_mapper[0] >= this->_rowrank) {
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
        this->_rowrank = this->_labels.size() - 1;

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
        this->_rowrank = 1;
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
  void DenseUniTensor::to_dense_() {
    cytnx_error_msg(!(this->_is_diag),
                    "[ERROR] to_dense_ can only operate on UniTensor with is_diag = True.%s", "\n");
    this->_block = cytnx::linalg::Diag(this->_block);
    this->_is_diag = false;
  }

  boost::intrusive_ptr<UniTensor_base> DenseUniTensor::contract(
    const boost::intrusive_ptr<UniTensor_base> &rhs, const bool &mv_elem_self,
    const bool &mv_elem_rhs) {
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
    // std::cout << comm_idx1 << std::endl;
    // std::cout << comm_idx2 << std::endl;

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
      tmp->_rowrank = this->rowrank() + rhs->rowrank();

      // if((this->is_diag() == rhs->is_diag()) && this->is_diag()){
      //     tmp->_block = linalg::Kron(linalg::Diag(this->_block),
      //     linalg::Diag(rhs->get_block_())); tmp->_block.reshape_({-1}); tmp->_is_diag = true;
      // }else{
      Tensor tmpL, tmpR;
      if (this->is_diag())
        tmpL = linalg::Diag(this->_block);
      else {
        if (this->_block.is_contiguous())
          tmpL = this->_block;
        else
          tmpL = this->_block.contiguous();
      }

      if (rhs->is_diag())
        tmpR = linalg::Diag(rhs->get_block_());
      else {
        if (rhs->get_block_().is_contiguous())
          tmpR = rhs->get_block_();  // share view!!
        else
          tmpR = rhs->get_block_().contiguous();
      }
      std::vector<cytnx_int64> old_shape_L(tmpL.shape().begin(), tmpL.shape().end());
      // vector<cytnx_int64> old_shape_R(tmpR.shape().begin(),tmpR.shape().end());
      std::vector<cytnx_int64> shape_L =
        vec_concatenate(old_shape_L, std::vector<cytnx_int64>(tmpR.shape().size(), 1));
      // vector<cytnx_int64> shape_R =
      // vec_concatenate(vector<cytnx_int64>(old_shape_L.size(),1),old_shape_R);
      tmpL.reshape_(shape_L);
      // tmpR.reshape_(shape_R);
      tmp->_block = linalg::Kron(tmpL, tmpR, false, true);
      tmpL.reshape_(old_shape_L);
      // tmpR.reshape_(old_shapeR);
      tmp->_is_diag = false;

      //}
      tmp->_is_braket_form = tmp->_update_braket();

    } else {
      // if tag, checking bra-ket matching!
      if (this->is_tag()) {
        for (int i = 0; i < comm_idx1.size(); i++)
          cytnx_error_msg(this->_bonds[comm_idx1[i]].type() == rhs->_bonds[comm_idx2[i]].type(),
                          "[ERROR][DenseUniTensor][contract] cannot contract common label: <%d> @ "
                          "self bond#%d & rhs bond#%d, BRA-KET mismatch!%s",
                          this->labels()[comm_idx1[i]], comm_idx1[i], comm_idx2[i], "\n");
      }

      // process meta
      // std::cout << this->rank() << " " << rhs->rank() << std::endl;
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
      tmp->_rowrank = this->rowrank() + rhs->rowrank();
      for (cytnx_uint64 i = 0; i < comm_idx1.size(); i++)
        if (comm_idx1[i] < this->_rowrank) tmp->_rowrank--;
      for (cytnx_uint64 i = 0; i < comm_idx2.size(); i++)
        if (comm_idx2[i] < rhs->_rowrank) tmp->_rowrank--;

      if ((this->is_diag() == rhs->is_diag()) && this->is_diag()) {
        // diag x diag:
        if (tmp->_rowrank != 0) {
          tmp->_block = this->_block * rhs->get_block_();
        } else {
          tmp->_block = linalg::Vectordot(this->_block, rhs->get_block_());
        }
        tmp->_is_diag = true;
      } else {
        if (this->is_diag() != rhs->is_diag()) {
          // diag x dense:
          // Tensor tmpL,tmpR;

          // if(this->is_diag()) tmpL = linalg::Diag(this->_block);
          // else tmpL = this->_block;
          // if(rhs->is_diag()) tmpR = linalg::Diag(rhs->get_block_());
          // else tmpR =  rhs->get_block_(); // share view!!
          tmp->_block = linalg::Tensordot_dg(this->_block, rhs->get_block_(), comm_idx1, comm_idx2,
                                             this->is_diag());

        } else {
          // dense x dense:
          // Tensor tmpL,tmpR;
          // tmpL = this->_block;
          // tmpR =  rhs->get_block_(); // share view!!
          // std::cout << "dkd" << std::endl;
          // std::cout << this->_block.shape() << std::endl;
          // std::cout << rhs->get_block_().shape() << std::endl;
          tmp->_block = linalg::Tensordot(this->_block, rhs->get_block_(), comm_idx1, comm_idx2,
                                          mv_elem_self, mv_elem_rhs);
        }
        tmp->_is_diag = false;
      }
      tmp->_is_braket_form = tmp->_update_braket();

    }  // check if no common index

    boost::intrusive_ptr<UniTensor_base> out(tmp);
    return out;
  }

  Tensor DenseUniTensor::Norm() const { return linalg::Norm(this->_block); }

  void DenseUniTensor::Trace_(const cytnx_int64 &a, const cytnx_int64 &b, const bool &by_label) {
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
    cytnx_error_msg(ida == idb,
                    "[ERROR][DenseUniTensor::Trace_] index a and index b should not be the same.%s",
                    "\n");

    // check dimension:
    cytnx_error_msg(
      this->_bonds[ida].dim() != this->_bonds[idb].dim(),
      "[ERROR][DenseUniTensor::Trace_] The dimension of two bond for trace does not match!%s",
      "\n");

    // check bra-ket if tagged
    if (this->is_braket_form()) {
      // check if it is the same species:
      if (this->_bonds[ida].type() == this->_bonds[idb].type()) {
        cytnx_error_msg(
          true, "[ERROR][DenseUniTensor::Trace_] BD_BRA can only contract with BD_KET.%s", "\n");
      }
    }

    // trace the block:
    if (this->_is_diag) {
      cytnx_error_msg(true, "[Error] We need linalg.Sum!%s", "\n");
    } else {
      this->_block = this->_block.Trace(ida, idb);
    }

    // update rowrank:
    cytnx_int64 tmpRk = this->_rowrank;
    if (ida < tmpRk) this->_rowrank--;
    if (idb < tmpRk) this->_rowrank--;

    // remove the bound, labels:
    if (ida > idb) std::swap(ida, idb);
    this->_bonds.erase(this->_bonds.begin() + idb);
    this->_bonds.erase(this->_bonds.begin() + ida);
    this->_labels.erase(this->_labels.begin() + idb);
    this->_labels.erase(this->_labels.begin() + ida);
  }

  void DenseUniTensor::Transpose_() {
    std::vector<cytnx_int64> new_permute =
      vec_concatenate(vec_range<cytnx_int64>(this->rowrank(), this->rank()),
                      vec_range<cytnx_int64>(0, this->rowrank()));
    this->permute_(new_permute);
    if (this->is_tag()) {
      this->_rowrank = this->rank() - this->_rowrank;
      for (int i = 0; i < this->rank(); i++) {
        this->_bonds[i].set_type((this->_bonds[i].type() == BD_KET) ? BD_BRA : BD_KET);
      }
      this->_is_braket_form = this->_update_braket();
    } else {
      this->_rowrank = this->rank() - this->_rowrank;
    }
  }

  void DenseUniTensor::_save_dispatch(std::fstream &f) const { this->_block._Save(f); }
  void DenseUniTensor::_load_dispatch(std::fstream &f) { this->_block._Load(f); }

  void DenseUniTensor::truncate_(const cytnx_int64 &bond_idx, const cytnx_uint64 &dim,
                                 const bool &by_label) {
    // if it is diagonal tensor, truncate will be done on both index!
    cytnx_error_msg(dim < 1, "[ERROR][DenseUniTensor][truncate] dim should be >0.%s", "\n");
    cytnx_uint64 idx;
    if (by_label) {
      auto it = std::find(this->_labels.begin(), this->_labels.end(), idx);
      cytnx_error_msg(it == this->_labels.end(),
                      "[ERROR][DenseUniTensor][truncate] Error, bond label does not exist in the "
                      "current label list.%s",
                      "\n");
      idx = std::distance(this->_labels.begin(), it);
    } else {
      idx = bond_idx;
    }
    cytnx_error_msg(
      idx >= this->_labels.size(),
      "[ERROR][DenseUniTensor][truncate] Error, index [%d] is out of range. Total rank: %d\n", idx,
      this->_labels.size());
    cytnx_error_msg(dim > this->_bonds[idx].dim(),
                    "[ERROR][DenseUniTensor][truncate] dimension can only be <= the dimension of "
                    "the instance bond.%s",
                    "\n");

    // if dim is the same as the dimension, don't do anything.
    if (dim != this->_bonds[idx].dim()) {
      // if diag type.
      if (this->_is_diag) {
        for (int i = 0; i < this->_bonds.size(); i++) {
          this->_bonds[i]._impl->_dim = dim;
        }
        this->_block = this->_block.get({ac::range(0, dim)});
      } else {
        this->_bonds[idx]._impl->_dim = dim;

        std::vector<ac> accessors(this->_bonds.size(), ac::all());
        accessors[idx] = ac::range(0, dim);
        this->_block = this->_block.get(accessors);
      }
    }
  }

  // Arithmetic:
  void DenseUniTensor::Add_(const boost::intrusive_ptr<UniTensor_base> &rhs) {
    cytnx_error_msg(rhs->is_tag(), "[ERROR] cannot perform arithmetic on tagged unitensor R.%s",
                    "\n");
    cytnx_error_msg(this->is_tag(), "[ERROR] cannot perform arithmetic on tagged unitensor L.%s",
                    "\n");
    this->_block += rhs->get_block_();
  }
  void DenseUniTensor::Add_(const Scalar &rhs) {
    // cytnx_error_msg(this->is_tag(),"[ERROR] cannot perform arithmetic on tagged unitensor
    // L.%s","\n");
    this->_block += rhs;
  }

  void DenseUniTensor::Sub_(const boost::intrusive_ptr<UniTensor_base> &rhs) {
    cytnx_error_msg(rhs->is_tag(), "[ERROR] cannot perform arithmetic on tagged unitensor R.%s",
                    "\n");
    cytnx_error_msg(this->is_tag(), "[ERROR] cannot perform arithmetic on tagged unitensor L.%s",
                    "\n");
    this->_block -= rhs->get_block_();
  }
  void DenseUniTensor::Sub_(const Scalar &rhs) {
    // cytnx_error_msg(this->is_tag(),"[ERROR] cannot perform arithmetic on tagged unitensor
    // L.%s","\n");
    this->_block -= rhs;
  }
  void DenseUniTensor::lSub_(const Scalar &lhs) {
    // cytnx_error_msg(this->is_tag(),"[ERROR] cannot perform arithmetic on tagged unitensor
    // R.%s","\n");
    this->_block = lhs - this->_block;
  }

  void DenseUniTensor::Mul_(const boost::intrusive_ptr<UniTensor_base> &rhs) {
    cytnx_error_msg(rhs->is_tag(), "[ERROR] cannot perform arithmetic on tagged unitensor R.%s",
                    "\n");
    cytnx_error_msg(this->is_tag(), "[ERROR] cannot perform arithmetic on tagged unitensor L.%s",
                    "\n");
    this->_block *= rhs->get_block_();
  }
  void DenseUniTensor::Mul_(const Scalar &rhs) {
    // cytnx_error_msg(this->is_tag(),"[ERROR] cannot perform arithmetic on tagged unitensor
    // L.%s","\n");
    this->_block *= rhs;
  }

  void DenseUniTensor::Div_(const boost::intrusive_ptr<UniTensor_base> &rhs) {
    cytnx_error_msg(rhs->is_tag(), "[ERROR] cannot perform arithmetic on tagged unitensor R.%s",
                    "\n");
    cytnx_error_msg(this->is_tag(), "[ERROR] cannot perform arithmetic on tagged unitensor L.%s",
                    "\n");
    this->_block /= rhs->get_block_();
  }
  void DenseUniTensor::Div_(const Scalar &rhs) {
    // cytnx_error_msg(this->is_tag(),"[ERROR] cannot perform arithmetic on tagged unitensor
    // L.%s","\n");
    this->_block /= rhs;
  }
  void DenseUniTensor::lDiv_(const Scalar &lhs) {
    // cytnx_error_msg(this->is_tag(),"[ERROR] cannot perform arithmetic on tagged unitensor
    // R.%s","\n");
    this->_block = lhs / this->_block;
  }
}  // namespace cytnx
