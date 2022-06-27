#include "Symmetry.hpp"
#include <string>
#include <fstream>

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
      return string("");
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
  cytnx_int64 Symmetry_base::reverse_rule(const cytnx_int64 &in) {
    cytnx_int64 out;
    this->reverse_rule_(out, in);
    return out;
  }

  bool cytnx::Symmetry_base::check_qnum(const cytnx_int64 &qnum) {
    cytnx_error_msg(1, "%s", "[ERROR][Internal] should not call Symmerty base!");
    return false;
  }
  bool cytnx::Symmetry_base::check_qnums(const std::vector<cytnx_int64> &qnums) {
    cytnx_error_msg(1, "%s", "[ERROR][Internal] should not call Symmerty base!");
    return false;
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
  void cytnx::Symmetry_base::reverse_rule_(cytnx_int64 &out, const cytnx_int64 &in) {
    cytnx_error_msg(1, "%s", "[ERROR][Internal] should not call Symmerty base!");
  }

  void cytnx::Symmetry_base::print_info() const {
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
  void cytnx::U1Symmetry::reverse_rule_(cytnx_int64 &out, const cytnx_int64 &in) { out = in * -1; }

  void cytnx::U1Symmetry::print_info() const {
    cout << "--------------------\n";
    cout << "[Symmetry]" << endl;
    cout << "type : Abelian, U1" << endl;
    cout << "combine rule : Q1 + Q2" << endl;
    cout << "reverse rule : Q*(-1) " << endl;
    cout << "--------------------\n";
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
  void cytnx::ZnSymmetry::reverse_rule_(cytnx_int64 &out, const cytnx_int64 &in) { out = in * -1; }

  void cytnx::ZnSymmetry::print_info() const {
    cout << "--------------------\n";
    cout << "[Symmetry]" << endl;
    cout << "type : Abelian, Z(" << this->n << ")" << endl;
    cout << "combine rule : (Q1 + Q2)\%" << this->n << endl;
    cout << "reverse rule : Q*(-1) " << endl;
    cout << "--------------------\n";
  }

  //==================================================

  void cytnx::Symmetry::Save(const std::string &fname) const {
    fstream f;
    f.open((fname + ".cysym"), ios::out | ios::trunc | ios::binary);
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for save.%s", "\n");
    }
    this->_Save(f);
    f.close();
  }
  void cytnx::Symmetry::Save(const char *fname) const {
    fstream f;
    string ffname = string(fname) + ".cysym";
    f.open((ffname), ios::out | ios::trunc | ios::binary);
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for save.%s", "\n");
    }
    this->_Save(f);
    f.close();
  }

  cytnx::Symmetry cytnx::Symmetry::Load(const std::string &fname) {
    Symmetry out;
    fstream f;
    f.open(fname, ios::in | ios::binary);
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for load.%s", "\n");
    }
    out._Load(f);
    f.close();
    return out;
  }

  cytnx::Symmetry cytnx::Symmetry::Load(const char *fname) {
    Symmetry out;
    fstream f;
    f.open(fname, ios::in | ios::binary);
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for load.%s", "\n");
    }
    out._Load(f);
    f.close();
    return out;
  }

  //==================
  void cytnx::Symmetry::_Save(fstream &f) const {
    cytnx_error_msg(!f.is_open(), "[ERROR][Symmetry] invalid fstream%s", "\n");
    unsigned int IDDs = 777;
    f.write((char *)&IDDs, sizeof(unsigned int));
    f.write((char *)&this->_impl->stype_id, sizeof(int));
    f.write((char *)&this->_impl->n, sizeof(int));
  }
  void cytnx::Symmetry::_Load(fstream &f) {
    cytnx_error_msg(!f.is_open(), "[ERROR][Symmetry] invalid fstream%s", "\n");
    unsigned int tmpIDDs;
    f.read((char *)&tmpIDDs, sizeof(unsigned int));
    cytnx_error_msg(tmpIDDs != 777, "[ERROR] the object is not a cytnx symmetry!%s", "\n");
    int stype_in;
    int n_in;
    f.read((char *)&stype_in, sizeof(int));
    f.read((char *)&n_in, sizeof(int));
    this->Init(stype_in, n_in);
  }

  ostream &operator<<(ostream &os, const Symmetry &in) {
    in.print_info();
    return os;
  }

  //++++++++++++++++++++++++
  SymmetryType_class SymType;
};  // namespace cytnx
