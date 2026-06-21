#include "tn_algo/MPS.hpp"
#include "Generator.hpp"
#include "random.hpp"
#include <cmath>
#include <algorithm>
#include "linalg.hpp"
#include "utils/vec_print.hpp"
using namespace std;

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  namespace tn_algo {

    std::ostream &RegularMPS::Print(std::ostream &os) {
      os << "MPS type : "
         << "[Regular]" << endl;
      os << "Size : " << this->_TNs.size() << endl;
      os << "Sloc : " << this->S_loc << endl;
      os << "physBD dim :\n";

      // print Sloc indicator:
      if (this->S_loc == -1) {
        os << ".[";
      } else {
        os << " [";
      }
      for (int i = 0; i < this->_TNs.size(); i++) {
        os << " ";
        if (this->S_loc == i)
          os << "'" << this->_TNs[i].shape()[1] << "'";
        else
          os << this->_TNs[i].shape()[1];
      }
      if (this->S_loc == this->_TNs.size()) {
        os << " ].\n";
      } else {
        os << "] \n";
      }

      os << "virtBD dim : " << this->virt_dim << endl;
      os << endl;
      return os;
    }

    Scalar RegularMPS::norm() const {
      // Accumulate the <psi|psi> left environment site by site. L carries two
      // open legs, "a" on the ket and "b" on the bra, living on the current
      // virtual bond; it is seeded as the 1x1 identity on the trivial left
      // boundary. For each rank-3 site tensor [left, phys, right] the ket leg
      // contracts onto "a", the bra leg onto "b", and the physical legs onto
      // each other, leaving the next virtual bond open as the new {a, b}.
      UniTensor L = UniTensor(ones({1, 1}), false, 1).relabel({"a", "b"});
      for (auto Ai : this->_TNs) {
        auto tA = Ai.relabel({"a", "p", "r"});
        auto tAc = Ai.Conj().relabel({"b", "p", "rc"});
        L = Contract(Contract(L, tA), tAc);  // {a, b} -> {r, rc}
        L.relabel_({"a", "b"});
      }
      return L.Trace().item();
    }

    void RegularMPS::Init(const cytnx_uint64 &N, const std::vector<cytnx_uint64> &vphys_dim,
                          const cytnx_uint64 &virt_dim, const cytnx_int64 &dtype) {
      // checking:
      cytnx_error_msg(N == 0, "[ERROR][RegularMPS] number of site N cannot be ZERO.%s", "\n");
      cytnx_error_msg(N != vphys_dim.size(),
                      "[ERROR] RegularMPS vphys_dim.size() should be equal to N.%s", "\n");
      cytnx_error_msg(dtype != Type.Double,
                      "[ERROR][RegularMPS] currently only Double dtype is support.%s", "\n");

      this->virt_dim = virt_dim;

      const cytnx_uint64 &chi = virt_dim;

      this->_TNs.resize(N);
      this->_TNs[0] = UniTensor(
        cytnx::random::normal({1, vphys_dim[0], min(chi, vphys_dim[0])}, 0., 1.), false, 2);
      cytnx_uint64 dim1, dim2, dim3;

      cytnx_uint64 DR = 1;
      cytnx_int64 k_ov = -1;
      for (cytnx_int64 k = N - 1; k >= 0; k--) {
        if (std::numeric_limits<cytnx_uint64>::max() / vphys_dim[k] <= DR) {
          k_ov = k;
          break;
        } else {
          DR *= vphys_dim[k];
        }
      }

      if (k_ov == -1) {
        DR /= vphys_dim[0];
        k_ov = 0;
      }

      for (cytnx_int64 k = 1; k < N; k++) {
        dim1 = this->_TNs[k - 1].shape()[2];
        dim2 = vphys_dim[k];

        if (k <= k_ov) {
          dim3 = std::min(chi, cytnx_uint64(dim1 * dim2));
        } else {
          DR /= vphys_dim[k];
          dim3 = std::min(std::min(chi, cytnx_uint64(dim1 * dim2)), DR);
        }
        this->_TNs[k] = UniTensor(random::normal({dim1, dim2, dim3}, 0., 1., -1), false, 2);
        this->_TNs[k].relabel_({to_string(2 * k), to_string(2 * k + 1), to_string(2 * k + 2)});
      }
      this->S_loc = -1;
      this->Into_Lortho();
    }

    void RegularMPS::Init_Msector(const cytnx_uint64 &N, const std::vector<cytnx_uint64> &vphys_dim,
                                  const cytnx_uint64 &virt_dim,
                                  const std::vector<cytnx_int64> &select,
                                  const cytnx_int64 &dtype) {
      // checking:
      cytnx_error_msg(N == 0, "[ERROR][RegularMPS] number of site N cannot be ZERO.%s", "\n");
      cytnx_error_msg(N != vphys_dim.size(),
                      "[ERROR] RegularMPS vphys_dim.size() should be equal to N.%s", "\n");
      cytnx_error_msg(dtype != Type.Double,
                      "[ERROR][RegularMPS] currently only Double dtype is support.%s", "\n");
      cytnx_error_msg(select.size() != vphys_dim.size(),
                      "[ERROR][RegularMPS] select.size() should equal to N.%s", "\n");

      this->virt_dim = virt_dim;

      const cytnx_uint64 &chi = virt_dim;

      this->_TNs.resize(N);
      // this->_TNs[0] = UniTensor(cytnx::random::normal({1, vphys_dim[0], min(chi, vphys_dim[0])},
      // 0., 1.,-1),2);
      this->_TNs[0] = UniTensor(cytnx::zeros({1, vphys_dim[0], min(chi, vphys_dim[0])}), false, 2);
      this->_TNs[0].get_block_()(":", select[0], ":") =
        random::normal({1, this->_TNs[0].shape()[2]}, 0., 1.);

      cytnx_uint64 dim1, dim2, dim3;

      cytnx_uint64 DR = 1;
      cytnx_int64 k_ov = -1;
      for (cytnx_int64 k = N - 1; k >= 0; k--) {
        if (std::numeric_limits<cytnx_uint64>::max() / vphys_dim[k] <= DR) {
          k_ov = k;
          break;
        } else {
          DR *= vphys_dim[k];
        }
      }

      if (k_ov == -1) {
        DR /= vphys_dim[0];
        k_ov = 0;
      }

      for (cytnx_int64 k = 1; k < N; k++) {
        dim1 = this->_TNs[k - 1].shape()[2];
        dim2 = vphys_dim[k];
        if (k <= k_ov) {
          dim3 = std::min(chi, cytnx_uint64(dim1 * dim2));
        } else {
          DR /= vphys_dim[k];
          dim3 = std::min(std::min(chi, cytnx_uint64(dim1 * dim2)), DR);
        }
        this->_TNs[k] = UniTensor(zeros({dim1, dim2, dim3}), false, 2);
        this->_TNs[k].get_block_()(":", select[k]) = random::normal({dim1, dim3}, 0., 1.);

        // this->_TNs[k] = UniTensor(random::normal({dim1, dim2, dim3},0.,1.,-1,99),2);
        this->_TNs[k].relabel_({to_string(2 * k), to_string(2 * k + 1), to_string(2 * k + 2)});
      }
      this->S_loc = -1;
      this->Into_Lortho();
    }
    /*
     void Init_prodstate(const std::vector<cytnx_uint64> &phys_dim, const std::vector<cytnx_uint64>
     cstate, const cytnx_int64 &dtype){

         //checking:
         cytnx_error_msg(dtype!=Type.Double,"[ERROR][RegularMPS] currently only Double dtype is
     support.%s","\n"); cytnx_error_msg(cstates.size()!vphys_dim.size(), "[ERROR][RegularMPS]
     len(cstates) should equal to len(phys_dim).%s","\n");

         this->virt_dim = 1;


         //this->_TNs.resize(phys_dim.size());

         for(cytnx_int64 k=0;k<cstates.size();k++){
             cytnx_error_msg(cstate[k]>=phys_dim[k],"[ERROR][RegularMPS][prodstate] site %d index %d
     is larger than the physical dimension %d",k,cstate[k],phys_dim[k]); auto tmp =
     zeros(phys_dim[k],dtype); tmp[cstate[k]] = 1; this->_TNs.append(cytnx.UniTensor(tmp,1));

         }

         this->S_loc = -1;



     }
     */

    // Canonical per-site labels assigned by RegularMPS::Init: site k carries
    // {2k, 2k+1, 2k+2}, so neighbouring sites share their virtual-bond label
    // (site k's right label 2k+2 equals site k+1's left label). Every routine
    // that rebuilds a site tensor through linalg::Svd restores these labels
    // afterwards. This keeps the MPS in a predictable labelling and, crucially,
    // stops Svd's fixed "_aux_L"/"_aux_R" bond labels from surviving into the
    // next decomposition, where the reused generic label would collide and trip
    // set_labels' duplicate check (issue #920).
    static std::vector<std::string> _site_labels(const cytnx_int64 &k) {
      return {to_string(2 * k), to_string(2 * k + 1), to_string(2 * k + 2)};
    }

    void RegularMPS::Into_Lortho() {
      if (this->S_loc == this->_TNs.size()) return;

      if (this->S_loc == -1) {
        this->S_loc = 0;
      } else if (this->S_loc == this->_TNs.size()) {
        return;
      }

      for (cytnx_int64 p = 0; p < this->size() - 1 - this->S_loc; p++) {
        auto out = linalg::Svd(this->_TNs[p]);
        auto s = out[0];
        this->_TNs[p] = out[1];
        auto vt = out[2];
        this->_TNs[p + 1] = Contract(Contract(s, vt), this->_TNs[p + 1]);
        this->_TNs[p].relabel_(_site_labels(p));
        this->_TNs[p + 1].relabel_(_site_labels(p + 1));
      }
      auto out = linalg::Svd(this->_TNs.back(), true);
      this->_TNs.back() = out[1];
      this->_TNs.back().relabel_(_site_labels(this->_TNs.size() - 1));
      this->S_loc = this->_TNs.size();
    }

    void RegularMPS::S_mvright() {
      if (this->S_loc == this->_TNs.size()) {
        return;
      } else if (this->S_loc == -1) {
        this->S_loc += 1;
        return;
      } else {
        this->_TNs[this->S_loc].set_rowrank_(2);
        auto out = linalg::Svd(this->_TNs[this->S_loc]);
        auto s = out[0];
        this->_TNs[this->S_loc] = out[1];
        auto vt = out[2];
        // boundary:
        if (this->S_loc != this->_TNs.size() - 1) {
          this->_TNs[this->S_loc + 1] = Contract(Contract(s, vt), this->_TNs[this->S_loc + 1]);
          this->_TNs[this->S_loc + 1].relabel_(_site_labels(this->S_loc + 1));
        }
        this->_TNs[this->S_loc].relabel_(_site_labels(this->S_loc));
        this->S_loc += 1;
      }
    }

    void RegularMPS::S_mvleft() {
      if (this->S_loc == -1) {
        return;
      } else if (this->S_loc == this->_TNs.size()) {
        this->S_loc -= 1;
        return;
      } else {
        this->_TNs[this->S_loc].set_rowrank_(1);
        auto out = linalg::Svd(this->_TNs[this->S_loc]);
        auto s = out[0];
        auto u = out[1];
        this->_TNs[this->S_loc] = out[2];

        // boundary:
        if (this->S_loc != 0) {
          this->_TNs[this->S_loc - 1] = Contract(this->_TNs[this->S_loc - 1], Contract(u, s));
          this->_TNs[this->S_loc - 1].relabel_(_site_labels(this->S_loc - 1));
        }
        this->_TNs[this->S_loc].relabel_(_site_labels(this->S_loc));
        this->S_loc -= 1;
      }
    }

    void RegularMPS::_save_dispatch(fstream &f) {
      cytnx_uint64 N = this->_TNs.size();
      f.write((char *)&N, sizeof(cytnx_uint64));

      // save UniTensor one by one:
      for (cytnx_uint64 i = 0; i < N; i++) {
        this->_TNs[i]._Save(f);
      }
    }
    void RegularMPS::_load_dispatch(fstream &f) {
      cytnx_uint64 N;

      f.read((char *)&N, sizeof(cytnx_uint64));
      this->_TNs.resize(N);

      // Load UniTensor one by one:
      for (cytnx_uint64 i = 0; i < N; i++) {
        this->_TNs[i]._Load(f);
      }
    }

  }  // namespace tn_algo

}  // namespace cytnx
#endif
