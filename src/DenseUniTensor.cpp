#include "UniTensor.hpp"
#include "utils/utils.hpp"

#include "Generator.hpp"
#include "linalg.hpp"
#include <algorithm>
#include <utility>
#include <vector>
typedef cytnx::Accessor ac;

using namespace std;

#ifdef BACKEND_TORCH
#else

namespace cytnx {

  void DenseUniTensor::Init(const std::vector<Bond> &bonds,
                            const std::vector<std::string> &in_labels, const cytnx_int64 &rowrank,
                            const unsigned int &dtype, const int &device, const bool &is_diag,
                            const bool &no_alloc, const std::string &name) {
    // check for all bonds
    this->_is_tag = false;
    this->_name = name;
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
    // cout << N_ket << endl;
    // cout << is_diag << endl;
    // cout << this->_is_tag << endl;

    // check rowrank
    if (this->_is_tag) {
      if (is_diag) {
        // cout << "NKET = " << N_ket << endl;
        cytnx_error_msg(N_ket != 1,
                        "[ERROR][DenseUniTensor] is_diag = true with tagged UniTensor must have "
                        "one IN (KET) bond  and one OUT (BRA) bond.%s",
                        "\n");
      }
      if (rowrank == -1) {
        this->_rowrank = N_ket;
      } else {
        if (is_diag) {
          cytnx_error_msg(rowrank != 1,
                          "[ERROR][DenseUniTensor] rowrank must be = 1 when is_diag = true.%s",
                          "\n");
        } else {
          cytnx_error_msg((rowrank < 0) || (rowrank > bonds.size()),
                          "[ERROR] rowrank is invalid or cannot exceed total rank of Tensor.%s",
                          "\n");
        }
        this->_rowrank = rowrank;
      }
    } else {
      if (bonds.size() == 0)
        this->_rowrank = 0;
      else {
        if (rowrank == -1) {
          this->_rowrank = 1;
        } else {
          if (is_diag) {
            cytnx_error_msg(rowrank != 1,
                            "[ERROR][DenseUniTensor] rowrank must be = 1 when is_diag = true.%s",
                            "\n");
          } else {
            cytnx_error_msg(
              rowrank < 0,
              "[ERROR] initialize a non-symmetry, un-tagged tensor should assign a >=0 rowrank.%s",
              "\n");
            cytnx_error_msg(rowrank > bonds.size(),
                            "[ERROR] rowrank cannot exceed total rank of Tensor.%s", "\n");
          }
          this->_rowrank = rowrank;
        }
      }
    }

    // check labels:
    if (in_labels.size() == 0) {
      for (cytnx_int64 i = 0; i < bonds.size(); i++) this->_labels.push_back(std::to_string(i));

    } else {
      // check bonds & labels dim
      cytnx_error_msg(bonds.size() != in_labels.size(), "%s",
                      "[ERROR] labels must have same lenth as # of bonds.");

      std::vector<std::string> tmp = vec_unique(in_labels);
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
                                      const cytnx_int64 &rowrank, const std::string &name) {
    this->_name = name;
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
        std::vector<cytnx_int64> zeroone = vec_range<cytnx_int64>(2);
        std::vector<std::string> vs;
        for (int i = 0; i < (int)zeroone.size(); i++) vs.push_back(std::to_string(zeroone[i]));
        this->_labels = vs;
        cytnx_error_msg(
          i_rowrank != 1,
          "[ERROR][Init_by_tensor] rowrank should be 1 for UniTensor with is_diag=True.%s", "\n");
        this->_rowrank = i_rowrank;
        this->_is_diag = true;

      } else {
        this->_is_diag = false;
        std::vector<Bond> bds;
        for (cytnx_uint64 i = 0; i < in_tensor.shape().size(); i++) {
          bds.push_back(Bond(in_tensor.shape()[i]));
        }
        // std::cout << bds.size() << std::endl;
        this->_bonds = bds;
        this->_block = in_tensor;
        std::vector<cytnx_int64> tmp = vec_range<cytnx_int64>(in_tensor.shape().size());
        std::vector<std::string> vs;
        for (int i = 0; i < (int)tmp.size(); i++) vs.push_back(std::to_string(tmp[i]));
        this->_labels = vs;
        // this->_labels = vec_range<cytnx_int64>(in_tensor.shape().size());

        if (i_rowrank == -1) {
          i_rowrank = int(in_tensor.shape().size() / 2);
        }

        cytnx_error_msg(i_rowrank > in_tensor.shape().size(),
                        "[ERROR][Init_by_tensor] rowrank exceed the rank of Tensor.%s", "\n");

        this->_rowrank = i_rowrank;
      }
    }
  }

  boost::intrusive_ptr<UniTensor_base> DenseUniTensor::relabel(
    const std::vector<std::string> &new_labels) {
    DenseUniTensor *out_raw = this->clone_meta();
    out_raw->_block = this->_block;
    out_raw->set_labels(new_labels);
    boost::intrusive_ptr<UniTensor_base> out(out_raw);
    return out;
  }

  boost::intrusive_ptr<UniTensor_base> DenseUniTensor::relabel(
    const std::vector<std::string> &old_labels, const std::vector<std::string> &new_labels) {
    DenseUniTensor *tmp = this->clone_meta();
    tmp->_block = this->_block;
    tmp->relabel_(old_labels, new_labels);
    boost::intrusive_ptr<UniTensor_base> out(tmp);
    return out;
  }

  boost::intrusive_ptr<UniTensor_base> DenseUniTensor::relabels(
    const std::vector<std::string> &new_labels) {
    DenseUniTensor *out_raw = this->clone_meta();
    out_raw->_block = this->_block;
    out_raw->set_labels(new_labels);
    boost::intrusive_ptr<UniTensor_base> out(out_raw);
    return out;
  }

  boost::intrusive_ptr<UniTensor_base> DenseUniTensor::relabels(
    const std::vector<std::string> &old_labels, const std::vector<std::string> &new_labels) {
    DenseUniTensor *tmp = this->clone_meta();
    tmp->_block = this->_block;
    tmp->relabels_(old_labels, new_labels);
    boost::intrusive_ptr<UniTensor_base> out(tmp);
    return out;
  }

  boost::intrusive_ptr<UniTensor_base> DenseUniTensor::relabel(const cytnx_int64 &inx,
                                                               const std::string &new_label) {
    DenseUniTensor *out_raw = this->clone_meta();
    out_raw->_block = this->_block;
    out_raw->set_label(inx, new_label);
    boost::intrusive_ptr<UniTensor_base> out(out_raw);
    return out;
  }

  boost::intrusive_ptr<UniTensor_base> DenseUniTensor::relabel(const std::string &inx,
                                                               const std::string &new_label) {
    DenseUniTensor *out_raw = this->clone_meta();
    out_raw->_block = this->_block;
    out_raw->set_label(inx, new_label);
    boost::intrusive_ptr<UniTensor_base> out(out_raw);
    return out;
  }

  boost::intrusive_ptr<UniTensor_base> DenseUniTensor::permute(
    const std::vector<cytnx_int64> &mapper, const cytnx_int64 &rowrank) {
    // boost::intrusive_ptr<UniTensor_base> out = this->clone();
    // out->permute_(mapper,rowrank,by_label);
    // return out;
    DenseUniTensor *out_raw = this->clone_meta();
    // boost::intrusive_ptr<UniTensor_base> out(this->clone_meta());

    std::vector<cytnx_uint64> mapper_u64;

    mapper_u64 = std::vector<cytnx_uint64>(mapper.begin(), mapper.end());

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

  boost::intrusive_ptr<UniTensor_base> DenseUniTensor::permute(
    const std::vector<std::string> &mapper, const cytnx_int64 &rowrank) {
    // boost::intrusive_ptr<UniTensor_base> out = this->clone();
    // out->permute_(mapper,rowrank,by_label);
    // return out;
    DenseUniTensor *out_raw = this->clone_meta();
    // boost::intrusive_ptr<UniTensor_base> out(this->clone_meta());

    std::vector<cytnx_uint64> mapper_u64;
    // cytnx_error_msg(true,"[Developing!]%s","\n");
    std::vector<std::string>::iterator it;
    for (cytnx_uint64 i = 0; i < mapper.size(); i++) {
      it = std::find(out_raw->_labels.begin(), out_raw->_labels.end(), mapper[i]);
      cytnx_error_msg(it == out_raw->_labels.end(),
                      "[ERROR] label %s does not exist in current UniTensor.\n", mapper[i].c_str());
      mapper_u64.push_back(std::distance(out_raw->_labels.begin(), it));
    }

    // mapper_u64 = std::vector<cytnx_uint64>(mapper.begin(), mapper.end());

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
  void DenseUniTensor::permute_(const std::vector<cytnx_int64> &mapper,
                                const cytnx_int64 &rowrank) {
    std::vector<cytnx_uint64> mapper_u64;

    mapper_u64 = std::vector<cytnx_uint64>(mapper.begin(), mapper.end());

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
  }
  void DenseUniTensor::permute_(const std::vector<std::string> &mapper,
                                const cytnx_int64 &rowrank) {
    std::vector<cytnx_uint64> mapper_u64;
    // cytnx_error_msg(true,"[Developing!]%s","\n");
    std::vector<std::string>::iterator it;
    for (cytnx_uint64 i = 0; i < mapper.size(); i++) {
      it = std::find(this->_labels.begin(), this->_labels.end(), mapper[i]);
      cytnx_error_msg(it == this->_labels.end(),
                      "[ERROR] label %d does not exist in current UniTensor.\n", mapper[i].c_str());
      mapper_u64.push_back(std::distance(this->_labels.begin(), it));
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
  }

  void DenseUniTensor::print_block(const cytnx_int64 &idx, const bool &full_info) const {
    std::ostream &os = std::cout;
    os << "-------- start of print ---------\n";
    char *buffer = (char *)malloc(sizeof(char) * 10240);
    sprintf(buffer, "Tensor name: %s\n", this->_name.c_str());
    os << std::string(buffer);
    if (this->_is_tag) {
      sprintf(buffer, "braket_form : %s\n", this->_is_braket_form ? "True" : "False");
      os << std::string(buffer);
    }
    sprintf(buffer, "is_diag    : %s\n", this->_is_diag ? "True" : "False");
    os << std::string(buffer);
    sprintf(buffer, "contiguous : %s\n", this->is_contiguous() ? "True" : "False");
    os << std::string(buffer);

    if (full_info)
      os << this->_block << std::endl;
    else {
      os << "dtype: " << Type.getname(this->_block.dtype()) << endl;
      os << "device: " << Device.getname(this->_block.device()) << endl;
      os << "shape: ";
      vec_print_simple(os, this->_block.shape());
    }
    free(buffer);
  }
  void DenseUniTensor::print_blocks(const bool &full_info) const {
    this->print_block(0, full_info);
  }

  void DenseUniTensor::print_diagram(const bool &bond_info) {
    char *buffer = (char *)malloc(10240 * sizeof(char));
    unsigned int BUFFsize = 100;

    sprintf(buffer, "-----------------------%s", "\n");
    std::cout << std::string(buffer);
    sprintf(buffer, "tensor Name : %s\n", this->_name.c_str());
    std::cout << std::string(buffer);
    sprintf(buffer, "tensor Rank : %ld\n", this->_labels.size());
    std::cout << std::string(buffer);
    sprintf(buffer, "block_form  : False%s", "\n");
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

    if (this->is_tag()) {
      sprintf(buffer, "braket_form : %s\n", this->_is_braket_form ? "True" : "False");
      std::cout << std::string(buffer);

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
          tmpss =
            this->_labels[i] + std::string(" ") * (Space_Llabel_max - this->_labels[i].size());
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

          tmpss =
            to_string(this->_bonds[Nin + i].dim()) +
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
      // sprintf(buffer, "%s", "\n");
      // std::cout << std::string(buffer);

    } else {
      sprintf(buffer, "%s     %s     %s", LallSpace.c_str(), M_dashes.c_str(), "\n");
      std::cout << std::string(buffer);
      for (cytnx_uint64 i = 0; i < vl; i++) {
        if (i == 0) {
          sprintf(buffer, "%s    /%s\\    %s", LallSpace.c_str(), MallSpace.c_str(), "\n");
          std::cout << std::string(buffer);
        } else {
          sprintf(buffer, "%s    |%s|    %s", LallSpace.c_str(), MallSpace.c_str(), "\n");
          std::cout << std::string(buffer);
        }

        if (i < Nin) {
          bks = "____";
          memset(l, 0, sizeof(char) * BUFFsize);
          memset(llbl, 0, sizeof(char) * BUFFsize);
          tmpss =
            this->_labels[i] + std::string(" ") * (Space_Llabel_max - this->_labels[i].size());
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
          bks = "____";
          memset(r, 0, sizeof(char) * BUFFsize);
          memset(rlbl, 0, sizeof(char) * BUFFsize);

          sprintf(r, "%s %s", bks.c_str(), this->_labels[Nin + i].c_str());

          tmpss =
            to_string(this->_bonds[Nin + i].dim()) +
            std::string(" ") * (Space_Rdim_max - to_string(this->_bonds[Nin + i].dim()).size());
          sprintf(rlbl, "%s", tmpss.c_str());

        } else {
          memset(r, 0, sizeof(char) * 40);
          memset(rlbl, 0, sizeof(char) * 40);
          sprintf(r, "%s", "        ");
          tmpss = std::string(" ") * Space_Rdim_max;
          sprintf(rlbl, "%s", tmpss.c_str());
        }
        sprintf(buffer, "   %s| %s     %s |%s\n", l, llbl, rlbl, r);
        std::cout << std::string(buffer);
      }
      sprintf(buffer, "%s    \\%s/    %s", LallSpace.c_str(), MallSpace.c_str(), "\n");
      std::cout << std::string(buffer);
      sprintf(buffer, "%s     %s     %s", LallSpace.c_str(), M_dashes.c_str(), "\n");
      std::cout << std::string(buffer);
    }

    if (bond_info) {
      for (cytnx_uint64 i = 0; i < this->_bonds.size(); i++) {
        // sprintf(buffer, "lbl:%ld ", this->_labels[i]);
        sprintf(buffer, "lbl:%s ", this->_labels[i].c_str());
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
      // std::cout << this->_block << std::endl;
      this->_block.reshape_(new_shape);
      // std::cout << this->_block << std::endl;
      this->Init_by_Tensor(this->_block, false, rowrank, this->_name);
      //}else{
      //    cytnx_error_msg(new_shape[0]!=new_shape[1],"[ERROR] invalid shape. The total elements
      //    does not match.%s","\n"); cytnx_error_msg(rowrank!=1,"[ERROR] UniTensor with
      //    is_diag=True should have rowrank=1.%s","\n");
      //}
    } else {
      this->_block.reshape_(new_shape);
      this->Init_by_Tensor(this->_block, false, rowrank, this->_name);
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
  void DenseUniTensor::combineBond(const std::vector<std::string> &indicators, const bool &force) {
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
  // Deprecated
  void DenseUniTensor::combineBonds(const std::vector<std::string> &indicators, const bool &force) {
    this->combineBond(indicators, force);
  }
  // Deprecated
  void DenseUniTensor::combineBonds(const std::vector<cytnx_int64> &indicators, const bool &force) {
    cytnx_error_msg(indicators.size() < 2, "[ERROR] the number of bonds to combine must be > 1%s",
                    "\n");
    std::vector<cytnx_int64>::iterator it;
    std::vector<cytnx_int64> idx_mapper;
    idx_mapper.reserve(this->rank());

    cytnx_error_msg(this->_is_diag,
                    "[ERROR] cannot combineBond on a is_diag=True UniTensor. suggestion: try "
                    "UniTensor.to_dense()/to_dense_() first.%s",
                    "\n");

    // get the mapper:
    int cnt = 0;
    int idor;
    for (int i = 0; i < this->rank(); i++) {
      if (cnt == indicators.size()) {
        idx_mapper.push_back(i);
      } else {
        if (std::find(indicators.begin(), indicators.end(), i) == indicators.end()) {
          idx_mapper.push_back(i);
        } else {
          if (i == indicators[0]) {
            idor = idx_mapper.size();  // new_shape_aft_perm.size();
            for (int j = 0; j < indicators.size(); j++) idx_mapper.push_back(indicators[j]);
          }
          cnt += 1;
        }
      }
    }
    this->permute_(idx_mapper);

    // group bonds:
    std::vector<Bond> new_bonds;
    // std::cout << "idor" << idor << std::endl;
    // std::cout << "rank" << this->rank() << std::endl;
    for (int i = 0; i < this->rank(); i++) {
      if (i == idor) {
        Bond tmp = this->_bonds[i];
        for (int j = 1; j < indicators.size(); j++) {
          if (force)
            tmp._impl->force_combineBond_(this->_bonds[i + j]._impl, false);
          else
            tmp.combineBond_(this->_bonds[i + j]);
        }
        new_bonds.push_back(tmp);
        i += indicators.size() - 1;

      } else {
        new_bonds.push_back(this->_bonds[i]);
      }
    }

    // remove labels, update bonds:
    this->_labels.erase(this->_labels.begin() + idor + 1,
                        this->_labels.begin() + idor + 1 + indicators.size() - 1);
    this->_bonds = new_bonds;

    // new shape:
    std::vector<cytnx_int64> new_shape(this->_block.shape().begin(), this->_block.shape().end());
    new_shape[idor] = -1;
    if (idor + indicators.size() < new_shape.size()) {
      memcpy(&new_shape[idor + 1], &new_shape[idor + indicators.size()],
             sizeof(cytnx_int64) * (new_shape.size() - idor - indicators.size()));
    }
    new_shape.resize(this->rank());  // rank follows this->_labels.size()!

    this->_block.reshape_(new_shape);

    if (this->_rowrank >= this->rank()) this->_rowrank = this->rank();

    if (this->is_tag()) {
      this->_is_braket_form = this->_update_braket();
    }
  }

  /*
  void DenseUniTensor::combineBonds(const std::vector<cytnx_int64> &indicators,
                                    const bool &force) {
    cytnx_error_msg(indicators.size() < 2, "[ERROR] the number of bonds to combine must be > 1%s",
                    "\n");
    std::vector<cytnx_int64>::iterator it;
    std::vector<cytnx_uint64> idx_mapper;
    idx_mapper = std::vector<cytnx_uint64>(indicators.begin(), indicators.end());

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

      std::string f_label = this->_labels[idx_mapper[0]];
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
        std::vector<std::string> new_labels;
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
        std::vector<std::string> new_labels;
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
  */

  // Deprecated
  void DenseUniTensor::combineBonds(const std::vector<cytnx_int64> &indicators, const bool &force,
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
    // cytnx_error_msg(this->is_diag() != rhs->is_diag(),"[ERROR] cannot contract a diagonal
    // tensor with non-diagonal tensor. [suggestion:] call UniTensor.to_dense/to_dense_
    // first%s","\n"); get common labels:
    std::vector<std::string> comm_labels;
    std::vector<cytnx_uint64> comm_idx1, comm_idx2;
    vec_intersect_<std::string>(comm_labels, this->labels(), rhs->labels(), comm_idx1, comm_idx2);
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
                          "[ERROR][DenseUniTensor][contract] cannot contract common label: <%s> @ "
                          "self bond#%d & rhs bond#%d, BRA-KET mismatch!%s",
                          this->labels()[comm_idx1[i]].c_str(), comm_idx1[i], comm_idx2[i], "\n");
      }

      // process meta
      // std::cout << this->rank() << " " << rhs->rank() << std::endl;
      std::vector<cytnx_uint64> non_comm_idx1 = vec_erase(vec_range(this->rank()), comm_idx1);
      std::vector<cytnx_uint64> non_comm_idx2 = vec_erase(vec_range(rhs->rank()), comm_idx2);

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
  void DenseUniTensor::Trace_(const std::string &a, const std::string &b) {
    // 1) from label to indx.
    cytnx_uint64 ida, idb;

    ida = vec_where(this->_labels, a);
    idb = vec_where(this->_labels, b);

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
      // cytnx_error_msg(true, "[Error] We need linalg.Sum!%s", "\n");
      this->_block = linalg::Sum(this->_block);
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
  void DenseUniTensor::Trace_(const cytnx_int64 &a, const cytnx_int64 &b) {
    // 1) from label to indx.
    cytnx_uint64 ida, idb;

    cytnx_error_msg(a < 0 || b < 0, "[ERROR] invalid index a, b%s", "\n");
    cytnx_error_msg(a >= this->rank() || b >= this->rank(), "[ERROR] index out of bound%s", "\n");
    ida = a;
    idb = b;

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
      // cytnx_error_msg(true, "[Error] We need linalg.Sum!%s", "\n");
      this->_block = linalg::Sum(this->_block);
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
    if (this->is_tag()) {
      // this->_rowrank = this->rank() - this->_rowrank;
      for (int i = 0; i < this->rank(); i++) {
        this->_bonds[i].set_type((this->_bonds[i].type() == BD_KET) ? BD_BRA : BD_KET);
      }
      this->_is_braket_form = this->_update_braket();

    } else {
      std::vector<cytnx_int64> new_permute =
        vec_concatenate(vec_range<cytnx_int64>(this->rowrank(), this->rank()),
                        vec_range<cytnx_int64>(0, this->rowrank()));
      this->permute_(new_permute);
      this->_rowrank = this->rank() - this->_rowrank;
    }
  }
  void DenseUniTensor::normalize_() { this->_block /= linalg::Norm(this->_block); }

  void DenseUniTensor::_save_dispatch(std::fstream &f) const { this->_block._Save(f); }
  void DenseUniTensor::_load_dispatch(std::fstream &f) { this->_block._Load(f); }

  void DenseUniTensor::truncate_(const std::string &bond_label, const cytnx_uint64 &dim) {
    // if it is diagonal tensor, truncate will be done on both index!
    cytnx_error_msg(dim < 1, "[ERROR][DenseUniTensor][truncate] dim should be >0.%s", "\n");
    cytnx_uint64 idx;
    auto it = std::find(this->_labels.begin(), this->_labels.end(), bond_label);
    cytnx_error_msg(it == this->_labels.end(),
                    "[ERROR][DenseUniTensor][truncate] Error, bond label does not exist in the "
                    "current label list.%s",
                    "\n");
    idx = std::distance(this->_labels.begin(), it);
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
  void DenseUniTensor::truncate_(const cytnx_int64 &bond_idx, const cytnx_uint64 &dim) {
    // if it is diagonal tensor, truncate will be done on both index!
    cytnx_error_msg(dim < 1, "[ERROR][DenseUniTensor][truncate] dim should be >0.%s", "\n");
    cytnx_uint64 idx;
    idx = bond_idx;

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
    // checking if Bond have same direction:
    if (this->is_tag()) {
      cytnx_error_msg(
        rhs->uten_type() != UTenType.Dense,
        "[ERROR][DenseUniTensor] cannot perform arithmetic with different type of UniTensor!%s",
        "\n");
      cytnx_error_msg(!rhs->is_tag(),
                      "[ERROR][DenseUniTensor] cannot perform arithmetic between tag and un-tag "
                      "DenseUniTensor!%s",
                      "\n");
      cytnx_error_msg(rhs->rank() != this->rank(),
                      "[ERROR] the rank of two UniTensor does not match!%s", "\n");

      for (cytnx_int64 i = 0; i < this->rank(); i++) {
        cytnx_error_msg(this->bonds()[i] != rhs->bonds()[i],
                        "[ERROR] Bond @ %d does not match, therefore cannot perform arithmetic!\n",
                        i);
      }
    }

    this->_block += rhs->get_block_();
  }
  void DenseUniTensor::Add_(const Scalar &rhs) {
    // cout << rhs << endl;
    //  cytnx_error_msg(this->is_tag(),"[ERROR] cannot perform arithmetic on tagged unitensor
    //  L.%s","\n");
    this->_block += rhs;
  }

  void DenseUniTensor::Sub_(const boost::intrusive_ptr<UniTensor_base> &rhs) {
    // checking if Bond have same direction:
    if (this->is_tag()) {
      cytnx_error_msg(
        rhs->uten_type() != UTenType.Dense,
        "[ERROR][DenseUniTensor] cannot perform arithmetic with different type of UniTensor!%s",
        "\n");
      cytnx_error_msg(!rhs->is_tag(),
                      "[ERROR][DenseUniTensor] cannot perform arithmetic between tag and un-tag "
                      "DenseUniTensor!%s",
                      "\n");
      cytnx_error_msg(rhs->rank() != this->rank(),
                      "[ERROR] the rank of two UniTensor does not match!%s", "\n");

      for (cytnx_int64 i = 0; i < this->rank(); i++) {
        cytnx_error_msg(this->bonds()[i] != rhs->bonds()[i],
                        "[ERROR] Bond @ %d does not match, therefore cannot perform arithmetic!\n",
                        i);
      }
    }
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
    // checking if Bond have same direction:
    if (this->is_tag()) {
      cytnx_error_msg(
        rhs->uten_type() != UTenType.Dense,
        "[ERROR][DenseUniTensor] cannot perform arithmetic with different type of UniTensor!%s",
        "\n");
      cytnx_error_msg(!rhs->is_tag(),
                      "[ERROR][DenseUniTensor] cannot perform arithmetic between tag and un-tag "
                      "DenseUniTensor!%s",
                      "\n");
      cytnx_error_msg(rhs->rank() != this->rank(),
                      "[ERROR] the rank of two UniTensor does not match!%s", "\n");

      for (cytnx_int64 i = 0; i < this->rank(); i++) {
        cytnx_error_msg(this->bonds()[i] != rhs->bonds()[i],
                        "[ERROR] Bond @ %d does not match, therefore cannot perform arithmetic!\n",
                        i);
      }
    }
    this->_block *= rhs->get_block_();
  }
  void DenseUniTensor::Mul_(const Scalar &rhs) {
    // cytnx_error_msg(this->is_tag(),"[ERROR] cannot perform arithmetic on tagged unitensor
    // L.%s","\n");
    this->_block *= rhs;
  }

  void DenseUniTensor::Div_(const boost::intrusive_ptr<UniTensor_base> &rhs) {
    // checking if Bond have same direction:
    if (this->is_tag()) {
      cytnx_error_msg(
        rhs->uten_type() != UTenType.Dense,
        "[ERROR][DenseUniTensor] cannot perform arithmetic with different type of UniTensor!%s",
        "\n");
      cytnx_error_msg(!rhs->is_tag(),
                      "[ERROR][DenseUniTensor] cannot perform arithmetic between tag and un-tag "
                      "DenseUniTensor!%s",
                      "\n");
      cytnx_error_msg(rhs->rank() != this->rank(),
                      "[ERROR] the rank of two UniTensor does not match!%s", "\n");

      for (cytnx_int64 i = 0; i < this->rank(); i++) {
        cytnx_error_msg(this->bonds()[i] != rhs->bonds()[i],
                        "[ERROR] Bond @ %d does not match, therefore cannot perform arithmetic!\n",
                        i);
      }
    }
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

  void _DN_from_DN(DenseUniTensor *ths, DenseUniTensor *rhs, const bool &force) {
    if (!force) {
      // more checking:
      if ((int(ths->bond_(0).type()) != bondType::BD_NONE) &&
          (int(rhs->bond_(0).type()) != bondType::BD_NONE)) {
        cytnx_error_msg(ths->bonds() != rhs->bonds(),
                        "[ERROR] conversion DenseUT -> DenseUT cannot be made, because "
                        "force=false, BOTH have directional Bond, and direction mismatch.%s",
                        "\n");
      }
    }

    ths->_block = rhs->_block.clone();
  }

  void _DN_from_BK(DenseUniTensor *ths, BlockUniTensor *rhs, const bool &force) {
    if (!force) {
      // more checking:
      if (int(ths->bond_(0).type()) != bondType::BD_NONE) {
        for (int i = 0; i < ths->bonds().size(); i++) {
          cytnx_error_msg(ths->bond_(i).type() != rhs->bond_(i).type(),
                          "[ERROR] conversion BlockUT -> DenseUT cannot be made, because "
                          "force=false, BOTH have directional Bond, and direction mismatch.%s",
                          "\n");
        }
      }
    }

    cytnx_uint64 total_elem = ths->_block.storage().size();

    std::vector<cytnx_uint64> stride_rhs(rhs->shape().size(), 1);
    for (int i = (rhs->rank() - 2); i >= 0; i--) {
      stride_rhs[i] = stride_rhs[i + 1] * rhs->shape()[i + 1];
    }

    // moving element:
    for (cytnx_uint64 i = 0; i < total_elem; i++) {
      auto cart = c2cartesian(i, stride_rhs);
      auto elem = rhs->at_for_sparse(cart);
      if (elem.exists()) {
        ths->_block.at(cart) = Scalar(elem);
      }
    }
  }

  void DenseUniTensor::from_(const boost::intrusive_ptr<UniTensor_base> &rhs, const bool &force) {
    // checking shape:
    cytnx_error_msg(this->shape() != rhs->shape(), "[ERROR][from_] shape does not match.%s", "\n");

    if (rhs->uten_type() == UTenType.Dense) {
      _DN_from_DN(this, (DenseUniTensor *)(rhs.get()), force);
    } else if (rhs->uten_type() == UTenType.Block) {
      _DN_from_BK(this, (BlockUniTensor *)(rhs.get()), force);
    } else {
      cytnx_error_msg(true, "[ERROR] unsupport conversion of UniTensor from %s => DenseUniTensor\n",
                      UTenType.getname(rhs->uten_type()).c_str());
    }
  }

}  // namespace cytnx
#endif
