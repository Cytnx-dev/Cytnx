
#include "Bond.hpp"
#include <algorithm>
#include "utils/utils.hpp"
using namespace std;

#ifdef UNI_OMP
  #include <omp.h>
#endif

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
                    "[ERROR] cannot combine two Bonds with differnet symmetry.");

    this->_dim *= bd_in->dim();  // update to new total dimension

    if (this->Nsym() != 0) {
      cytnx_error_msg(this->syms() != bd_in->syms(), "%s\n",
                      "[ERROR] cannot combine two Bonds with differnet symmetry.");

      // checking the qnum format:
      cytnx_error_msg((this->_degs.size() != 0) ^ (bd_in->_degs.size() != 0), "%s\n",
                      "[ERROR] cannot combine two symmetry bond with differet format!");

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

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
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

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
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
                    "[ERROR] cannot combine two Bonds with different types.");
    cytnx_error_msg(this->Nsym() != bd_in->Nsym(), "%s\n",
                    "[ERROR] cannot combine two Bonds with differnet symmetry.");

    this->_dim *= bd_in->dim();  // update to new total dimension

    if (this->Nsym() != 0) {
      cytnx_error_msg(this->syms() != bd_in->syms(), "%s\n",
                      "[ERROR] cannot combine two Bonds with differnet symmetry.");

      // checking the qnum format:
      cytnx_error_msg((this->_degs.size() != 0) ^ (bd_in->_degs.size() != 0), "%s\n",
                      "[ERROR] cannot combine two symmetry bond with differet format!");

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

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
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

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
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
                    "[ERROR][getUniqueQnums] cannot call this on a non-symmetry bond!%s", "\n");

    vector<vector<cytnx_int64>> tmp_qnums = this->_qnums;
    if (this->_degs.size()) {
      // new format
      counts = vec_clone(this->_degs);

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
          // std::cout << "add from loc" << q << " to " << cnt << std::endl;
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
#ifdef UNI_OMP
          std::vector<cytnx_uint64> tmp_cnts;
          std::vector<std::vector<cytnx_uint64>> tmp_indices;
  #pragma omp parallel
          {
            if (omp_get_thread_num() == 0) {
              tmp_cnts.resize(omp_get_num_threads(), 0);
              tmp_indices.resize(omp_get_num_threads());
            }
          }

  #pragma omp parallel for schedule(dynamic)
          for (cytnx_uint64 i = 0; i < this->_qnums.size(); i++) {
            if (this->_qnums[i] == qnum) {
              tmp_cnts[omp_get_thread_num()]++;
              tmp_indices[omp_get_thread_num()].push_back(i);
            }
          }

          // sum over all counts from each thread.
          cytnx_uint64 cnt = tmp_cnts[0];
          for (cytnx_uint64 i = 1; i < tmp_cnts.size(); i++) {
            cnt += tmp_cnts[i];
            tmp_cnts[i] += tmp_cnts[i - 1];  // tmp_cnts[i-1];
          }
          indices.resize(cnt);

          // concate the indices from each thread
          memcpy(&indices[0], &tmp_indices[0][0], sizeof(cytnx_uint64) * tmp_indices[0].size());
          for (cytnx_uint64 i = 1; i < tmp_cnts.size(); i++) {
            memcpy(&indices[tmp_cnts[i - 1]], &tmp_indices[i][0],
                   sizeof(cytnx_uint64) * tmp_indices[i].size());
          }
          std::sort(indices.begin(), indices.end());

          return cnt;
#else
          cytnx_uint64 cnt = 0;
          indices.clear();
          for (cytnx_uint64 i = 0; i < this->_qnums.size(); i++) {
            if (this->_qnums[i] == qnum) {
              cnt++;
              indices.push_back(i);
            }
          }
          return cnt;

#endif
        } else {
#ifdef UNI_OMP
          std::vector<cytnx_uint64> tmp_cnts;
  #pragma omp parallel
          {
            if (omp_get_thread_num() == 0) tmp_cnts.resize(omp_get_num_threads(), 0);
          }

  #pragma omp parallel for schedule(dynamic)
          for (cytnx_uint64 i = 0; i < this->_qnums.size(); i++) {
            if (this->_qnums[i] == qnum) {
              tmp_cnts[omp_get_thread_num()]++;
            }
          }

          for (cytnx_uint64 i = 1; i < tmp_cnts.size(); i++) tmp_cnts[0] += tmp_cnts[i];
          return tmp_cnts[0];
#else
          return std::count(this->_qnums.begin(), this->_qnums.end(), qnum);

#endif
        }  // if return indices
      }  // check format.

    }  // check if there is symmetry.
  }

  std::vector<std::vector<cytnx_int64>> Bond_impl::calc_reverse_qnums() {
    std::vector<std::vector<cytnx_int64>> out(this->_qnums.size(),
                                              std::vector<cytnx_int64>(this->_syms.size()));
#ifdef UNI_OMP
  #pragma omp parallel for
#endif
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

  void Bond::Save(const std::string &fname) const {
    fstream f;
    f.open((fname + ".cybd"), ios::out | ios::trunc | ios::binary);
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for save.%s", "\n");
    }
    this->_Save(f);
    f.close();
  }
  void Bond::Save(const char *fname) const {
    fstream f;
    string ffname = string(fname) + ".cybd";
    f.open((ffname), ios::out | ios::trunc | ios::binary);
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for save.%s", "\n");
    }
    this->_Save(f);
    f.close();
  }

  Bond Bond::Load(const std::string &fname) {
    Bond out;
    fstream f;
    f.open(fname, ios::in | ios::binary);
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for load.%s", "\n");
    }
    out._Load(f);
    f.close();
    return out;
  }

  Bond Bond::Load(const char *fname) {
    Bond out;
    fstream f;
    f.open(fname, ios::in | ios::binary);
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for load.%s", "\n");
    }
    out._Load(f);
    f.close();
    return out;
  }

  void Bond::_Save(fstream &f) const {
    cytnx_error_msg(!f.is_open(), "[ERROR][Bond] invalid fstream%s", "\n");
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
        this->_impl->_syms[j]._Save(f);
      }
    }
  }

  void Bond::_Load(fstream &f) {
    cytnx_error_msg(!f.is_open(), "[ERROR][Bond] invalid fstream%s", "\n");
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
        this->_impl->_syms[j]._Load(f);
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
