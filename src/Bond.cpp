
#include "Bond.hpp"

#include <algorithm>
#include <filesystem>

#include "H5Cpp.h"
#include "utils/utils.hpp"

using namespace std;

namespace cytnx {
  void Bond_impl::Init(const cytnx_uint64 &dim, const bondType &bd_type) {
    cytnx_error_msg(dim == 0, "%s", "[ERROR] Bond_impl cannot have 0 or negative dimension.");

    this->_type = bd_type;
    this->_dim = dim;
  }

  void Bond_impl::Init(const bondType &bd_type,
                       const std::vector<std::vector<cytnx_int64>> &in_qnums,
                       const std::vector<cytnx_uint64> &degs,
                       const std::vector<Symmetry> &in_syms) {
    cytnx_error_msg(degs.size() != in_qnums.size(), "%s",
                    "[ERROR] [degs] list must have same size as [qnum] list");
    cytnx_error_msg(degs.size() == 0, "%s", "[ERROR] [degs & qnums] lists cannot be empty!");

    /// calc total dim:
    this->_dim = 0;
    for (auto i : degs) {
      cytnx_error_msg(i == 0, "[ERROR] [degs] degs of Bond cannot contain zero%s", "\n");
      this->_dim += i;
    }

    // checking qnum size are the same for each entry.
    cytnx_uint64 N_syms = in_qnums[0].size();
    for (cytnx_uint64 i = 0; i < in_qnums.size(); i++) {
      cytnx_error_msg(in_qnums[i].size() != N_syms,
                      "[ERROR] invalid syms number on qnum @ index %d. the [# of elements for each "
                      "qnum list] should be the same as [# of symmetries]\n",
                      i);
    }

    // checking qnum size from in_qnums with in_syms!
    if (in_syms.size() == 0) {
      // default U1!
      this->_syms.clear();
      this->_syms.resize(N_syms);
      for (cytnx_uint64 i = 0; i < N_syms; i++) {
        this->_syms[i] = Symmetry::U1();
      }

    } else {
      cytnx_error_msg(
        in_syms.size() != N_syms, "%s",
        "[ERROR] the number of symmetry should match the # of cols of passed-in qnums.");
      this->_syms = vec_clone(in_syms);
    }

    this->_qnums = in_qnums;
    this->_degs = degs;
    this->_type = bd_type;

    // using this constor can only be symmetric!!
    cytnx_error_msg(bd_type == BD_REG, "%s",
                    "[ERROR] bond with qnums (symmetry) can only have bond_type=BD_BRA or BD_KET");

    // check qnums match the rule of each symmetry type
    for (cytnx_uint64 d = 0; d < in_qnums.size(); d++) {
      for (cytnx_uint64 i = 0; i < N_syms; i++)
        cytnx_error_msg(!this->_syms[i].check_qnum(this->_qnums[d][i]),
                        "[ERROR] invalid qnums @ index %d with Symmetry: %s\n", d,
                        this->_syms[i].stype_str().c_str());
    }
  }

  void Bond_impl::force_combineBond_(const boost::intrusive_ptr<Bond_impl> &bd_in,
                                     const bool &is_grp) {
    // check:
    cytnx_error_msg(this->Nsym() != bd_in->Nsym(), "%s\n",
                    "[ERROR] Cannot combine two Bonds with different symmetry.");

    this->_dim *= bd_in->dim();  // update to new total dimension

    if (this->Nsym() != 0) {
      cytnx_error_msg(this->syms() != bd_in->syms(), "%s\n",
                      "[ERROR] Cannot combine two Bonds with different symmetry.");

      // checking the qnum format:
      cytnx_error_msg((this->_degs.size() != 0) ^ (bd_in->_degs.size() != 0), "%s\n",
                      "[ERROR] Cannot combine two symmetry bond with different format!");

      std::vector<std::vector<cytnx_int64>> new_qnums;
      if (this->_degs.size()) {
        // new format:
        cytnx_uint64 Dnew_qnums = this->_qnums.size() * bd_in->_qnums.size();
        std::vector<cytnx_uint64> new_degs(Dnew_qnums);

        if (is_grp) {
          std::map<std::vector<cytnx_int64>, std::vector<cytnx_int32>> QNpool;
          std::vector<cytnx_int64> tmpqnum(this->_syms.size());

          for (cytnx_uint64 d = 0; d < Dnew_qnums; d++) {
            for (cytnx_uint32 i = 0; i < this->Nsym(); i++) {
              if (bd_in->type() != this->type()) {
                this->_syms[i].combine_rule_(
                  tmpqnum[i], this->_qnums[cytnx_uint64(d / bd_in->_qnums.size())][i],
                  this->_syms[i].reverse_rule(bd_in->_qnums[d % bd_in->_qnums.size()][i]));
              } else {
                this->_syms[i].combine_rule_(
                  tmpqnum[i], this->_qnums[cytnx_uint64(d / bd_in->_qnums.size())][i],
                  bd_in->_qnums[d % bd_in->_qnums.size()][i]);
              }
            }
            if (QNpool.find(tmpqnum) == QNpool.end()) {
              QNpool.insert(make_pair(tmpqnum, std::vector<cytnx_int32>({static_cast<int>(d)})));
            } else {
              QNpool[tmpqnum].push_back(d);
            }
            new_degs[d] += this->_degs[cytnx_uint64(d / bd_in->_qnums.size())] *
                           bd_in->_degs[d % bd_in->_qnums.size()];
          }

          // realloc & assign!
          this->_qnums.resize(QNpool.size());
          this->_degs.resize(QNpool.size());

          cytnx_uint64 cnt = 0;
          for (auto elem : QNpool) {
            this->_qnums[cnt] = elem.first;
            cytnx_uint64 DD = 0;
            for (auto i : elem.second) {
              DD += new_degs[i];
            }
            this->_degs[cnt] = DD;
            cnt++;
          }

        } else {
          new_qnums = std::vector<std::vector<cytnx_int64>>(Dnew_qnums,
                                                            std::vector<cytnx_int64>(this->Nsym()));

          for (cytnx_uint64 d = 0; d < new_qnums.size(); d++) {
            for (cytnx_uint32 i = 0; i < this->Nsym(); i++) {
              if (bd_in->type() != this->type()) {
                this->_syms[i].combine_rule_(
                  new_qnums[d][i], this->_qnums[cytnx_uint64(d / bd_in->_qnums.size())][i],
                  this->_syms[i].reverse_rule(bd_in->_qnums[d % bd_in->_qnums.size()][i]));
              } else {
                this->_syms[i].combine_rule_(
                  new_qnums[d][i], this->_qnums[cytnx_uint64(d / bd_in->_qnums.size())][i],
                  bd_in->_qnums[d % bd_in->_qnums.size()][i]);
              }
            }
            new_degs[d] = this->_degs[cytnx_uint64(d / bd_in->_qnums.size())] *
                          bd_in->_degs[d % bd_in->_qnums.size()];
          }

          // cytnx_warning_msg(true, "[WARNING] duplicated qnums might appears!%s", "\n");
          this->_degs = new_degs;
          this->_qnums = new_qnums;
        }

      } else {
        // old format:

        new_qnums =
          std::vector<std::vector<cytnx_int64>>(this->_dim, std::vector<cytnx_int64>(this->Nsym()));

        for (cytnx_uint64 d = 0; d < this->_dim; d++) {
          for (cytnx_uint32 i = 0; i < this->Nsym(); i++) {
            if (bd_in->type() != this->type()) {
              this->_syms[i].combine_rule_(
                new_qnums[d][i], this->_qnums[cytnx_uint64(d / bd_in->dim())][i],
                this->_syms[i].reverse_rule(bd_in->_qnums[d % bd_in->dim()][i]));
            } else {
              this->_syms[i].combine_rule_(new_qnums[d][i],
                                           this->_qnums[cytnx_uint64(d / bd_in->dim())][i],
                                           bd_in->_qnums[d % bd_in->dim()][i]);
            }
          }
        }
        this->_qnums = new_qnums;
      }

    }  // check if symmetry.
  }

  void Bond_impl::combineBond_(const boost::intrusive_ptr<Bond_impl> &bd_in, const bool &is_grp) {
    // check:
    cytnx_error_msg(this->type() != bd_in->type(), "%s\n",
                    "[ERROR] Cannot combine two Bonds with different types.");
    cytnx_error_msg(this->Nsym() != bd_in->Nsym(), "%s\n",
                    "[ERROR] Cannot combine two Bonds with different symmetry.");

    this->_dim *= bd_in->dim();  // update to new total dimension

    if (this->Nsym() != 0) {
      cytnx_error_msg(this->syms() != bd_in->syms(), "%s\n",
                      "[ERROR] Cannot combine two Bonds with different symmetry.");

      // checking the qnum format:
      cytnx_error_msg((this->_degs.size() != 0) ^ (bd_in->_degs.size() != 0), "%s\n",
                      "[ERROR] Cannot combine two symmetry bond with different format!");

      std::vector<std::vector<cytnx_int64>> new_qnums;
      if (this->_degs.size()) {
        // new format:
        cytnx_uint64 Dnew_qnums = this->_qnums.size() * bd_in->_qnums.size();
        std::vector<cytnx_uint64> new_degs(Dnew_qnums);

        if (is_grp) {
          std::map<std::vector<cytnx_int64>, std::vector<cytnx_int32>> QNpool;
          std::vector<cytnx_int64> tmpqnum(this->_syms.size());

          for (cytnx_uint64 d = 0; d < Dnew_qnums; d++) {
            for (cytnx_uint32 i = 0; i < this->Nsym(); i++) {
              this->_syms[i].combine_rule_(tmpqnum[i],
                                           this->_qnums[cytnx_uint64(d / bd_in->_qnums.size())][i],
                                           bd_in->_qnums[d % bd_in->_qnums.size()][i]);
            }
            if (QNpool.find(tmpqnum) == QNpool.end()) {
              QNpool.insert(make_pair(tmpqnum, std::vector<cytnx_int32>({static_cast<int>(d)})));
            } else {
              QNpool[tmpqnum].push_back(d);
            }
            new_degs[d] += this->_degs[cytnx_uint64(d / bd_in->_qnums.size())] *
                           bd_in->_degs[d % bd_in->_qnums.size()];
          }

          // realloc & assign!
          this->_qnums.resize(QNpool.size());
          this->_degs.resize(QNpool.size());

          cytnx_uint64 cnt = 0;
          for (auto elem : QNpool) {
            this->_qnums[cnt] = elem.first;
            cytnx_uint64 DD = 0;
            for (auto i : elem.second) {
              DD += new_degs[i];
            }
            this->_degs[cnt] = DD;
            cnt++;
          }

        } else {
          new_qnums = std::vector<std::vector<cytnx_int64>>(Dnew_qnums,
                                                            std::vector<cytnx_int64>(this->Nsym()));

          for (cytnx_uint64 d = 0; d < new_qnums.size(); d++) {
            for (cytnx_uint32 i = 0; i < this->Nsym(); i++) {
              this->_syms[i].combine_rule_(new_qnums[d][i],
                                           this->_qnums[cytnx_uint64(d / bd_in->_qnums.size())][i],
                                           bd_in->_qnums[d % bd_in->_qnums.size()][i]);
            }
            new_degs[d] = this->_degs[cytnx_uint64(d / bd_in->_qnums.size())] *
                          bd_in->_degs[d % bd_in->_qnums.size()];
          }

          cytnx_warning_msg(true, "[WARNING] duplicated qnums might appears!%s", "\n");
          this->_degs = new_degs;
          this->_qnums = new_qnums;
        }

      } else {
        // old format:

        new_qnums =
          std::vector<std::vector<cytnx_int64>>(this->_dim, std::vector<cytnx_int64>(this->Nsym()));

        for (cytnx_uint64 d = 0; d < this->_dim; d++) {
          for (cytnx_uint32 i = 0; i < this->Nsym(); i++) {
            this->_syms[i].combine_rule_(new_qnums[d][i],
                                         this->_qnums[cytnx_uint64(d / bd_in->dim())][i],
                                         bd_in->_qnums[d % bd_in->dim()][i]);
          }
        }
        this->_qnums = new_qnums;
      }

    }  // check if symmetry.
  }

  //-------------
  std::vector<std::vector<cytnx_int64>> Bond_impl::getUniqueQnums(std::vector<cytnx_uint64> &counts,
                                                                  const bool &return_counts) {
    cytnx_error_msg(this->_qnums.size() == 0,
                    "[ERROR][getUniqueQnums] Cannot call this on a non-symmetry bond!%s", "\n");

    vector<vector<cytnx_int64>> tmp_qnums = this->_qnums;
    if (this->_degs.size()) {
      // new format
      counts = this->_degs;
    } else {
      // old format
      std::sort(tmp_qnums.begin(), tmp_qnums.end(), _fx_compare_vec_dec);
      tmp_qnums.resize(
        std::distance(tmp_qnums.begin(), std::unique(tmp_qnums.begin(), tmp_qnums.end())));
      if (return_counts) {
        counts.resize(tmp_qnums.size());
        for (cytnx_uint64 i = 0; i < tmp_qnums.size(); i++) {
          counts[i] = std::count(this->_qnums.begin(), this->_qnums.end(), tmp_qnums[i]);
        }
      }
    }

    return tmp_qnums;
  }

  std::vector<cytnx_uint64> Bond_impl::group_duplicates_() {
    // the map returns the new index from old index via
    // new_index = return<cytnx_uint64>[old_index]
    // [Note] this will sort QN from small to large

    if (this->_degs.size()) {
      auto mapper = vec_sort(this->_qnums, true);
      auto tmp_degs = vec_map(this->_degs, mapper);

      cytnx_uint64 cnt = 0;
      cytnx_uint64 loc = 0;
      std::vector<cytnx_uint64> return_order(this->_qnums.size());
      std::vector<cytnx_uint64> idx_erase;
      std::vector<cytnx_int64> *last = &this->_qnums[0];
      return_order[mapper[0]] = cnt;
      for (int q = 1; q < this->_qnums.size(); q++) {
        if (this->_qnums[q] != *last) {
          last = &this->_qnums[q];
          cnt++;
          loc = q;
        } else {
          idx_erase.push_back(q);
          tmp_degs[loc] += tmp_degs[q];
        }
        return_order[mapper[q]] = cnt;
      }
      this->_degs = tmp_degs;
      // now, remove:
      this->_qnums = vec_erase(this->_qnums, idx_erase);
      vec_erase_(this->_degs, idx_erase);

      return return_order;
    } else {
      cytnx_error_msg(
        true, "[ERROR] group_duplicates() only work for new Bond format and with symmetry!%s",
        "\n");
      return std::vector<cytnx_uint64>();
    }
  }

  cytnx_uint64 Bond_impl::getDegeneracy(const std::vector<cytnx_int64> &qnum,
                                        const bool &return_indices,
                                        std::vector<cytnx_uint64> &indices) {
    if (this->_qnums.size() == 0) {
      return 0;
    } else {
      cytnx_error_msg(
        qnum.size() != this->Nsym(),
        "[ERROR][getDegeneracy] the qnum specify does not match the number of symmetries.%s", "\n");

      // check if the format is new or old
      if (this->_degs.size()) {
        // new format

        auto itt = std::find(this->_qnums.begin(), this->_qnums.end(), qnum);

        // cytnx_error_msg(itt == this->_qnums.end(),
        //                 "[ERROR] qnums does not exists in the current Bond!%s", "\n");
        indices.clear();
        if (itt == this->_qnums.end()) return 0;

        cytnx_uint64 loc = itt - this->_qnums.begin();

        if (return_indices) {
          indices = {loc};
        }

        return this->_degs[loc];

      } else {
        // old format
        if (return_indices) {
          cytnx_uint64 cnt = 0;
          indices.clear();
          for (cytnx_uint64 i = 0; i < this->_qnums.size(); i++) {
            if (this->_qnums[i] == qnum) {
              cnt++;
              indices.push_back(i);
            }
          }
          return cnt;

        } else {
          return std::count(this->_qnums.begin(), this->_qnums.end(), qnum);
        }  // if return indices
      }  // check format.

    }  // check if there is symmetry.
  }

  std::vector<std::vector<cytnx_int64>> Bond_impl::calc_reverse_qnums() {
    std::vector<std::vector<cytnx_int64>> out(this->_qnums.size(),
                                              std::vector<cytnx_int64>(this->_syms.size()));
    for (cytnx_uint64 i = 0; i < out.size(); i++) {
      for (int j = 0; j < out[i].size(); j++) {
        this->_syms[j].reverse_rule_(out[i][j], this->_qnums[i][j]);
      }
    }
    return out;
  }

  /*
  void Bond_impl::Init(const cytnx_uint64 &dim, const
  std::initializer_list<std::initializer_list<cytnx_int64> > &in_qnums,const
  std::initializer_list<Symmetry> &in_syms,const bondType &bd_type){


      std::vector< std::vector<cytnx_int64> > in_vec_qnums(in_qnums.size());
      for(cytnx_uint64 i=0;i<in_qnums.size();i++){
          //cytnx_error_msg(in_qnums.begin()[i].size() != Nsym,"%s","[ERROR] invalid qnums. the # of
  column of qnums list should be identify across each row. "); in_vec_qnums[i] =
  in_qnums.begin()[i];
      }

      std::vector<Symmetry> in_vec_syms = in_syms;

      this->Init(dim,in_vec_qnums,in_vec_syms,bd_type);

  }
  */

  fermionParity Bond_impl::get_fermion_parity(const std::vector<cytnx_int64> &qnum) {
    if (this->_qnums.size() == 0) {
      return EVEN;
    } else {
      cytnx_error_msg(
        qnum.size() != this->Nsym(),
        "[ERROR][get_fermion_parity] the qnum specify does not match the number of symmetries.%s",
        "\n");

      fermionParity out = EVEN;
      fermionParity curr = EVEN;
      for (cytnx_uint64 i = 0; i < qnum.size(); i++) {
        out = static_cast<fermionParity>(
          out != this->_syms[i].get_fermion_parity(
                   qnum[i]));  // false (ODD) if the symmetries are not equal
      }

      return out;
    }
  }

  //[done]
  bool Bond::operator==(const Bond &rhs) const {
    if (this->dim() != rhs.dim()) return false;
    if (this->type() != rhs.type()) return false;
    if (this->Nsym() != rhs.Nsym()) return false;
    if (this->_impl->_degs.size() != rhs._impl->_degs.size()) return false;
    if (this->Nsym() != 0) {
      if (this->syms() != rhs.syms()) return false;
    }

    if (this->_impl->_degs.size()) {
      if (this->_impl->_degs != rhs._impl->_degs) return false;
      if (this->_impl->_qnums != rhs._impl->_qnums) return false;
    } else {
      if (User_debug)
        if (this->_impl->_qnums != rhs._impl->_qnums) return false;
    }
    return true;
  }

  bool Bond::operator!=(const Bond &rhs) const { return !(*this == rhs); }

  void Bond::Save(const std::filesystem::path &fname, const std::string &path,
                  const char mode) const {
    fstream f;  // only for binary saving, not used for HDF5
    if (fname.has_extension()) {
      // filename extension is given
      std::string ext = fname.extension().string();
      if (ext == ".h5" || ext == ".hdf5" || ext == ".H5" || ext == ".HDF5" || ext == ".hdf" ||
          ext == ".HDF") {
        // save as HDF5
        H5::H5File h5file;
        bool overwrite = false;
        // open file
        if (mode == 'w') {  // Write new file
          h5file = H5::H5File(fname, H5F_ACC_TRUNC);
        } else if (mode == 'x') {  // eXclusive create
          h5file = H5::H5File(fname, H5F_ACC_EXCL);
        } else if (mode == 'a') {  // Append data
          if (std::filesystem::exists(fname))
            h5file = H5::H5File(fname, H5F_ACC_RDWR);
          else
            h5file = H5::H5File(fname, H5F_ACC_EXCL);
        } else if (mode == 'u') {  // Update data
          if (std::filesystem::exists(fname)) {
            h5file = H5::H5File(fname, H5F_ACC_RDWR);
            overwrite = true;
          } else {
            h5file = H5::H5File(fname, H5F_ACC_EXCL);
          }
        } else {
          cytnx_error_msg(true, "[ERROR] Unknown mode '%c' for writing to HDF5 file.", mode);
        }
        // create group
        H5::Group location = h5file;
        try {
          H5::Exception::dontPrint();
          location = h5file.openGroup(path);
        } catch (const H5::Exception &e) {
          H5::LinkCreatPropList lcpl;
          lcpl.setCreateIntermediateGroup(1);
          location = h5file.createGroup(path, lcpl);
        }
        this->to_hdf5(location, overwrite);
        h5file.close();
        return;
      } else {  // create binary file
        if (mode == 'x') {
          cytnx_error_msg(std::filesystem::exists(fname),
                          "[ERROR] File %s already exists. Use mode 'w' to overwrite.", fname);
        } else {
          cytnx_error_msg(mode != 'w', "[ERROR] Unknown mode '%c' for writing to binary file.",
                          mode);
        }
        f.open(fname, std::ios::out | std::ios::trunc | std::ios::binary);
      }
    } else {  // create binary file with standard extension
      std::filesystem::path fnameext = fname;
      fnameext += ".cybd";
      cytnx_warning_msg(true,
                        "Missing file extension in fname '%s'. I am adding the extension '.cybd'. "
                        "This is deprecated, please provide the file extension in the future.\n",
                        fname.c_str());
      if (mode == 'x') {
        cytnx_error_msg(std::filesystem::exists(fnameext),
                        "[ERROR] File %s already exists. Use mode 'w' to overwrite.",
                        fnameext.c_str());
      } else {
        cytnx_error_msg(mode != 'w', "[ERROR] Unknown mode '%c' for writing to binary file.", mode);
      }
      f.open(fnameext, std::ios::out | std::ios::trunc | std::ios::binary);
    }
    // write binary
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for save.%s", "\n");
    }
    this->to_binary(f);
    f.close();
  }
  void Bond::Save(const char *fname, const std::string &path, const char mode) const {
    this->Save(std::filesystem::path(fname), path, mode);
  }

  Bond Bond::Load(const std::filesystem::path &fname, const std::string &path) {
    Bond out;
    out.Load_(fname, path);
    return out;
  }
  Bond Bond::Load(const char *fname, const std::string &path) {
    return Bond::Load(std::filesystem::path(fname), path);
  }

  void Bond::Load_(const std::filesystem::path &fname, const std::string &path) {
    std::string ext = fname.extension().string();
    if (ext == ".h5" || ext == ".hdf5" || ext == ".H5" || ext == ".HDF5" || ext == ".hdf" ||
        ext == ".HDF") {
      // load HDF5
      H5::H5File h5file(fname, H5F_ACC_RDONLY);
      H5::Group location;
      try {
        H5::Exception::dontPrint();
        location = h5file.openGroup(path);
      } catch (const H5::Exception &e) {
        std::cerr << e.getDetailMsg() << std::endl;
        cytnx_error_msg(true, "[ERROR] HDF5 path '%s' not found or is not a group in file '%s'.",
                        path.c_str(), fname.c_str());
      }
      this->from_hdf5(location);
      h5file.close();
    } else {  // load binary
      fstream f;
      f.open(fname, std::ios::in | std::ios::binary);
      if (!f.is_open()) {
        cytnx_error_msg(true, "[ERROR] Cannot open file '%s'.\n", fname.c_str());
      }
      this->from_binary(f);
      f.close();
    }
  }
  void Bond::Load_(const char *fname, const std::string &path) {
    this->Load_(std::filesystem::path(fname), path);
  }

  void Bond::to_hdf5(H5::Group &location, const bool overwrite, const bool save_symmetries) const {
    if (overwrite) {  // delete previous data
      // remove attributes
      if (location.attrExists("dimension")) location.removeAttr("dimension");
      if (location.attrExists("type")) location.removeAttr("type");
      // remove datasets
      if (location.nameExists("degeneracies")) location.unlink("degeneracies");
      if (location.nameExists("quantum_numbers")) location.unlink("quantum_numbers");
      // remove groups and its contents recursively
      if (location.nameExists("Symmetries")) location.unlink("Symmetries");
    }

    H5::DataType datatype;
    H5::Attribute attr;
    H5::DataSet dataset;
    H5::DataSpace dataspace;
    H5::StrType str_type;

    // dimension, write as attribute
    auto dim = this->_impl->_dim;
    datatype = Type.get_hdf5_type(dim);
    attr = location.createAttribute("dimension", datatype, H5::DataSpace(H5S_SCALAR));
    attr.write(H5::PredType::NATIVE_INT, &dim);

    // type, write as string
    std::string typestr = bondtype_to_string.at(this->_impl->_type);
    str_type = H5::StrType(H5::PredType::C_S1, typestr.length() + 1);
    dataspace = H5::DataSpace(H5S_SCALAR);
    attr = location.createAttribute("type", str_type, dataspace);
    attr.write(str_type, typestr);

    // degs; write vector
    hsize_t sectordim = this->_impl->_qnums.size();
    if (sectordim > 0) {  // only with symmetries
      hsize_t vecdims[1] = {sectordim};
      dataspace = H5::DataSpace(1, vecdims);
      datatype = Type.get_hdf5_type(this->_impl->_degs[0]);
      dataset = location.createDataSet("degeneracies", Type.get_hdf5_type(this->_impl->_degs[0]),
                                       dataspace);
      dataset.write(this->_impl->_degs.data(), datatype);

      // qnums; write matrix (dim x qnumdim)
      hsize_t qnumdim = this->_impl->_syms.size();
      std::vector<cytnx_int64> flat(sectordim * qnumdim);  // flatten vector<vector>
      for (hsize_t i = 0; i < sectordim; ++i) {
        std::copy(this->_impl->_qnums[i].begin(), this->_impl->_qnums[i].end(),
                  flat.begin() + i * qnumdim);
      }
      hsize_t matdims[2] = {sectordim, qnumdim};
      dataspace = H5::DataSpace(2, matdims);
      datatype = Type.get_hdf5_type(flat[0]);
      dataset = location.createDataSet("quantum_numbers", datatype, dataspace);
      dataset.write(flat.data(), datatype);
      // label axes
      char labels[2][9] = {"sector", "symmetry"};
      str_type = H5::StrType(H5::PredType::C_S1, 9);
      hsize_t attr_dims[1] = {2};
      H5::DataSpace attr_space(1, attr_dims);
      attr = dataset.createAttribute("axis_labels", str_type, attr_space);
      attr.write(str_type, labels);

      // Symmetries
      if (save_symmetries) {
        // vecdims[1] = { this->_impl->_syms.size() };
        // dataspace = H5::DataSpace(1, vecdims);
        // str_type = H5::StrType(H5::PredType::C_S1, H5T_VARIABLE);
        // dataset = location.createDataSet("symmetries", str_type, dataspace);

        // std::vector<const char*> c_strings;
        // std::string symstring;
        // for (const auto& s : this->_impl->_syms) {
        //   symstring = s.name();
        //   c_strings.push_back(symstring.c_str());
        // }
        // dataset.write(c_strings.data(), str_type);
        H5::Group symloc = location.createGroup("Symmetries");
        for (int sidx = 0; sidx < this->_impl->_syms.size(); sidx++) {
          this->_impl->_syms[sidx].to_hdf5(symloc, overwrite, "Symmetry" + std::to_string(sidx));
        }
      }
    }
  }
  void Bond::from_hdf5(H5::Group &location, const std::vector<Symmetry> &syms) {
    H5::DataType datatype;
    H5::Attribute attr;
    H5::DataSet dataset;
    H5::DataSpace dataspace;
    H5::StrType str_type;

    // type from string
    attr = location.openAttribute("type");
    str_type = attr.getStrType();
    size_t size = str_type.getSize() - 1;  // not including the null terminator
    std::string typestr;
    typestr.resize(size);
    attr.read(str_type, &typestr[0]);
    this->_impl->_type = string_to_bondtype.at(typestr);

    // degs; read vector
    bool symmetric = false;
    hssize_t sectordim;
    if (location.exists("degeneracies")) {
      symmetric = true;
      dataset = location.openDataSet("degeneracies");
      dataspace = dataset.getSpace();
      sectordim = dataspace.getSimpleExtentNpoints();
      this->_impl->_degs.resize(sectordim);
      datatype = dataset.getDataType();
      cytnx_error_msg(
        datatype.getSize() != sizeof(std::size_t),
        "[ERROR] 'degeneracies' bit-length mismatch. File: %zu bytes, expected: %zu bytes.\n",
        datatype.getSize(), sizeof(std::size_t));
      dataset.read(this->_impl->_degs.data(), datatype);
      this->_impl->_dim =
        std::accumulate(this->_impl->_degs.begin(), this->_impl->_degs.end(), cytnx_uint64(0));
    } else {
      this->_impl->_degs = {};
    }

    // qnums; read matrix (dim x qnumdim)
    hsize_t qnumdim;
    if (location.exists("quantum_numbers")) {
      cytnx_error_msg(!symmetric,
                      "[ERROR] 'degeneracies' were not found in HDF5 location, but "
                      "'quantum_numbers' existn. The HDF5 data seems corrupt!%s",
                      "\n");
      dataset = location.openDataSet("quantum_numbers");
      dataspace = dataset.getSpace();
      cytnx_error_msg(dataspace.getSimpleExtentNdims() != 2,
                      "[ERROR] 'quantum_numbers' should be a two-dimensional array. The HDF5 data "
                      "seems corrupt!%s",
                      "\n");
      hsize_t dims[2];
      dataspace.getSimpleExtentDims(dims);
      cytnx_error_msg(dims[0] != sectordim,
                      "[ERROR] Length of 'degeneracies' = %d, but first dimension of "
                      "'quantum_numbers' is %d. The HDF5 data seems corrupt!\n",
                      sectordim, dims[0]);
      qnumdim = dims[1];
      // Read HDF5 data into a flattened temporary vector
      std::vector<cytnx_int64> flat(sectordim * qnumdim);
      datatype = dataset.getDataType();
      cytnx_error_msg(
        datatype.getSize() != sizeof(cytnx_int64),
        "[ERROR] 'quantum_numbers' bit-length mismatch. File: %zu bytes, expected: %zu bytes.\n",
        datatype.getSize(), sizeof(cytnx_int64));
      dataset.read(flat.data(), datatype);
      // Reconstruct the vector of vectors
      this->_impl->_qnums.assign(sectordim, std::vector<cytnx_int64>(qnumdim));
      for (hsize_t i = 0; i < sectordim; ++i) {
        std::copy(flat.begin() + i * qnumdim, flat.begin() + (i + 1) * qnumdim,
                  this->_impl->_qnums[i].begin());
      }
    } else {
      this->_impl->_qnums = {};
    }

    // Symmetries
    if (syms.empty()) {
      if (location.exists("Symmetries")) {
        H5::Group symloc = location.openGroup("Symmetries");
        this->_impl->_syms.clear();
        hsize_t sidx = 0;
        while (true) {
          std::string name = "Symmetry" + std::to_string(sidx);
          if (!symloc.attrExists(name) && !symloc.exists(name)) {
            break;
          }
          Symmetry sym;
          sym.from_hdf5(symloc, name);
          this->_impl->_syms.push_back(sym);
          sidx++;
        }
        symloc.close();
        if (symmetric) {
          cytnx_error_msg(sidx != qnumdim,
                          "[ERROR] %d Symmetries were found, but second dimension of "
                          "'quantum_numbers' is %d. The HDF5 data seems corrupt!\n",
                          sidx, qnumdim);
        } else {
          cytnx_error_msg(sidx > 0,
                          "[ERROR] 'degeneracies' and 'quantum_numbers' were not found in HDF5 "
                          "location, but 'Symmetries' exist. The HDF5 data seems corrupt!%s",
                          "\n");
          this->_impl->_syms = {};
        }
      } else {
        cytnx_error_msg(symmetric,
                        "[ERROR] 'degeneracies' and 'quantum_numbers' exist in HDF5 location, but "
                        "'Symmetries' are missing. The HDF5 data seems corrupt!%s",
                        "\n");
      }
    } else {
      cytnx_error_msg(!symmetric,
                      "[ERROR] 'degeneracies' and 'quantum_numbers' not found in HDF5 location, "
                      "but Symmetries are passed as arguments.%s",
                      "\n");
      cytnx_error_msg(syms.size() != qnumdim,
                      "[ERROR] %d Symmetries are passed, but second dimension of 'quantum_numbers' "
                      "is %d. The HDF5 data seems corrupt!\n",
                      syms.size(), qnumdim);
      this->_impl->_syms = syms;
    }

    // dim; from attribute
    if (location.attrExists("dimension")) {
      attr = location.openAttribute("dimension");
      cytnx_uint64 dimension;
      datatype = attr.getDataType();
      cytnx_error_msg(
        datatype.getSize() != sizeof(cytnx_uint64),
        "[ERROR] 'dimension' bit-length mismatch. File: %zu bytes, expected: %zu bytes.\n",
        datatype.getSize(), sizeof(cytnx_uint64));
      attr.read(datatype, &dimension);
      if (symmetric) {
        cytnx_error_msg(dimension != this->_impl->_dim,
                        "[ERROR] 'dimension' read from HDF5 file is %d, but the sum of all "
                        "degeneracies is %d. The HDF5 data seems corrupt!\n",
                        dimension, this->_impl->_dim);
      } else {
        this->_impl->_dim = dimension;
      }
    } else {
      cytnx_error_msg(!symmetric,
                      "[ERROR] Could not find 'dimension' or 'degeneracies' in HDF5 file. The HDF5 "
                      "data seems corrupt!%s",
                      "\n");
    }
  }

  void Bond::to_binary(std::ostream &f) const {
    unsigned int IDDs = 666;
    f.write((char *)&IDDs, sizeof(unsigned int));

    // write format version!
    int ver = 0;
    if (this->_impl->_degs.size()) {
      ver = 1;
    }
    f.write((char *)&ver, sizeof(int));

    cytnx_uint64 tmp;
    if (ver == 0)
      f.write((char *)&this->_impl->_dim, sizeof(cytnx_uint64));
    else {
      // if it is new version, store the size of qnums list instead!
      tmp = this->_impl->_qnums.size();
      f.write((char *)&tmp, sizeof(cytnx_uint64));
    }
    f.write((char *)&this->_impl->_type, sizeof(int));

    // write Nsyms:
    cytnx_uint64 Nsym = this->_impl->_syms.size();
    f.write((char *)&Nsym, sizeof(cytnx_uint64));

    if (ver == 0) {  // old version:
      if (Nsym != 0) {
        // writing qnums:
        for (cytnx_uint64 i = 0; i < this->_impl->_dim; i++) {
          f.write((char *)&(this->_impl->_qnums[i][0]), sizeof(cytnx_int64) * Nsym);
        }
        //
      }
    } else {
      // new version, can only be symmetric!
      // writing qnums:
      for (cytnx_uint64 i = 0; i < this->_impl->_qnums.size(); i++) {
        f.write((char *)&(this->_impl->_qnums[i][0]), sizeof(cytnx_int64) * Nsym);
      }
      f.write((char *)&(this->_impl->_degs[0]), sizeof(cytnx_uint64) * this->_impl->_degs.size());
    }

    if (Nsym != 0) {
      for (int j = 0; j < Nsym; j++) {
        this->_impl->_syms[j].to_binary(f);
      }
    }
  }

  void Bond::from_binary(std::istream &f) {
    unsigned int tmpIDDs;
    f.read((char *)&tmpIDDs, sizeof(unsigned int));
    cytnx_error_msg(tmpIDDs != 666, "[ERROR] the object is not a cytnx Bond!%s", "\n");

    int ver;
    f.read((char *)&ver, sizeof(int));

    cytnx_uint64 tmpD;
    if (ver == 0)
      f.read((char *)&this->_impl->_dim, sizeof(cytnx_uint64));
    else {
      f.read((char *)&tmpD, sizeof(cytnx_uint64));
    }
    f.read((char *)&this->_impl->_type, sizeof(int));

    // read Nsyms:
    cytnx_uint64 Nsym_in;
    f.read((char *)&Nsym_in, sizeof(cytnx_uint64));

    if (ver == 0) {
      if (Nsym_in != 0) {
        this->_impl->_qnums = std::vector<std::vector<cytnx_int64>>(
          this->_impl->_dim, std::vector<cytnx_int64>(Nsym_in));
        // reading qnums:
        for (cytnx_uint64 i = 0; i < this->_impl->_dim; i++) {
          f.read((char *)&(this->_impl->_qnums[i][0]), sizeof(cytnx_int64) * Nsym_in);
        }
        //
      }
    } else {
      // new version can only be qnums
      // recalc _dim!
      this->_impl->_qnums =
        std::vector<std::vector<cytnx_int64>>(tmpD, std::vector<cytnx_int64>(Nsym_in));

      this->_impl->_degs.resize(tmpD);
      // reading qnums:
      for (cytnx_uint64 i = 0; i < tmpD; i++) {
        f.read((char *)&(this->_impl->_qnums[i][0]), sizeof(cytnx_int64) * Nsym_in);
      }

      f.read((char *)&(this->_impl->_degs[0]), sizeof(cytnx_uint64) * tmpD);

      this->_impl->_dim = 0;
      for (auto c : this->_impl->_degs) {
        this->_impl->_dim += c;
      }
    }

    if (Nsym_in != 0) {
      this->_impl->_syms.resize(Nsym_in);
      for (int j = 0; j < Nsym_in; j++) {
        this->_impl->_syms[j].from_binary(f);
      }
    }
  }

  std::ostream &operator<<(std::ostream &os, const Bond &bin) {
    char *buffer = (char *)malloc(sizeof(char) * 256);
    os << "Dim = " << bin.dim() << " | ";
    if (bin.type() == bondType::BD_REG) {
      os << "type: REGULAR " << std::endl;
    } else if (bin.type() == bondType::BD_BRA) {
      os << "type: <OUT (BRA)| " << std::endl;
    } else if (bin.type() == bondType::BD_KET) {
      os << "type: | IN (KET)> " << std::endl;
    } else {
      cytnx_error_msg(1, "%s", "[ERROR] internal error.");
    }
    // os << bin.get_syms().size() << endl;

    for (cytnx_int32 i = 0; i < bin.Nsym(); i++) {
      os << " " << bin.syms()[i].stype_str() << ":: ";
      for (cytnx_int32 j = 0; j < bin.qnums().size(); j++) {
        sprintf(buffer, " %+3d", bin.qnums()[j][i]);
        os << string(buffer);
      }
      os << std::endl;
    }
    if (bin._impl->_degs.size()) {
      os << "Deg>> ";
      for (cytnx_int32 i = 0; i < bin.qnums().size(); i++) {
        sprintf(buffer, " %3d", bin._impl->_degs[i]);
        os << string(buffer);
      }
      os << std::endl;
    }

    free(buffer);
    return os;
  }

}  // namespace cytnx
