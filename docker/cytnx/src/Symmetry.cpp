#include "Symmetry.hpp"
#include <string>
#ifdef UNI_OMP
  #include <omp.h>
#endif
using namespace std;

namespace cytnx {

  bool cytnx::Symmetry::operator==(const cytnx::Symmetry &rhs) const {
    return (this->stype() == rhs.stype()) && (this->n() == rhs.n());
  }
  bool cytnx::Symmetry::operator!=(const cytnx::Symmetry &rhs) const { return !(*this == rhs); }
  //=============================

  std::string cytnx::SymmetryType_class::getname(const int &stype_id) {
    if (stype_id == this->U) {
      return string("U");
    } else if (stype_id == this->Z) {
      return string("Z");
    } else {
      cytnx_error_msg(true, "%s", "[ERROR] invalid stype_id");
    }
    // extend more in here!!
  }

  //==============================

  std::vector<cytnx_int64> Symmetry_base::combine_rule(const std::vector<cytnx_int64> &inL,
                                                       const std::vector<cytnx_int64> &inR) {
    std::vector<cytnx_int64> out;
    this->combine_rule_(out, inL, inR);
    return out;
  }

  cytnx_int64 Symmetry_base::combine_rule(const cytnx_int64 &inL, const cytnx_int64 &inR) {
    cytnx_int64 out;
    this->combine_rule_(out, inL, inR);
    return out;
  }

  bool cytnx::Symmetry_base::check_qnum(const cytnx_int64 &qnum) {
    cytnx_error_msg(1, "%s", "[ERROR][Internal] should not call Symmerty base!");
  }
  bool cytnx::Symmetry_base::check_qnums(const std::vector<cytnx_int64> &qnums) {
    cytnx_error_msg(1, "%s", "[ERROR][Internal] should not call Symmerty base!");
  }
  void cytnx::Symmetry_base::combine_rule_(std::vector<cytnx_int64> &out,
                                           const std::vector<cytnx_int64> &inL,
                                           const std::vector<cytnx_int64> &inR) {
    cytnx_error_msg(1, "%s", "[ERROR][Internal] should not call Symmerty base!");
  }
  void cytnx::Symmetry_base::combine_rule_(cytnx_int64 &out, const cytnx_int64 &inL,
                                           const cytnx_int64 &inR) {
    cytnx_error_msg(1, "%s", "[ERROR][Internal] should not call Symmerty base!");
  }
  ///=========================
  bool cytnx::U1Symmetry::check_qnum(const cytnx_int64 &qnum) { return true; }
  bool cytnx::U1Symmetry::check_qnums(const std::vector<cytnx_int64> &qnums) { return true; }

  void cytnx::U1Symmetry::combine_rule_(std::vector<cytnx_int64> &out,
                                        const std::vector<cytnx_int64> &inL,
                                        const std::vector<cytnx_int64> &inR) {
    out.resize(inL.size() * inR.size());
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
    for (cytnx_uint64 i = 0; i < out.size(); i++) {
      out[i] = inL[cytnx_uint64(i / inR.size())] + inR[i % inR.size()];
    }
  }
  void cytnx::U1Symmetry::combine_rule_(cytnx_int64 &out, const cytnx_int64 &inL,
                                        const cytnx_int64 &inR) {
    out = inL + inR;
  }

  ///========================
  bool cytnx::ZnSymmetry::check_qnum(const cytnx_int64 &qnum) {
    return (qnum >= 0) && (qnum < this->n);
  }

  bool cytnx::ZnSymmetry::check_qnums(const std::vector<cytnx_int64> &qnums) {
#ifdef UNI_OMP
    std::vector<bool> buf(1, true);
  #pragma omp parallel
    {
      if (omp_get_thread_num() == 0) buf.assign(omp_get_num_threads(), true);
    }

  #pragma omp parallel for schedule(dynamic)
    for (cytnx_uint64 i = 0; i < qnums.size(); i++) {
      if (buf[omp_get_thread_num()] == false)
        continue;
      else
        buf[omp_get_thread_num()] = ((qnums[i] >= 0) && (qnums[i] < this->n));
    }

    for (cytnx_uint64 i = 0; i < buf.size(); i++) {
      buf[0] = (buf[0] && buf[i]);
    }
    return buf[0];
#else
    bool out = true;
    for (cytnx_uint64 i = 0; i < qnums.size(); i++) {
      out = ((qnums[i] >= 0) && (qnums[i] < this->n));
      if (out == false) break;
    }
    return out;
#endif
  }

  void cytnx::ZnSymmetry::combine_rule_(std::vector<cytnx_int64> &out,
                                        const std::vector<cytnx_int64> &inL,
                                        const std::vector<cytnx_int64> &inR) {
    out.resize(inL.size() * inR.size());
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
    for (cytnx_uint64 i = 0; i < out.size(); i++) {
      out[i] = (inL[cytnx_uint64(i / inR.size())] + inR[i % inR.size()]) % (this->n);
    }
  }
  void cytnx::ZnSymmetry::combine_rule_(cytnx_int64 &out, const cytnx_int64 &inL,
                                        const cytnx_int64 &inR) {
    out = (inL + inR) % (this->n);
  }

  //++++++++++++++++++++++++
  SymmetryType_class SymType;
};  // namespace cytnx
