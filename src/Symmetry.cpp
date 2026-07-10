#include "Symmetry.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace cytnx {

  namespace {
    void ValidateZnQnum(const cytnx_int64 qnum, const cytnx_int64 n) {
      cytnx_error_msg(
        (qnum < 0) || (qnum >= n),
        "[ERROR][ZnSymmetry] qnum %lld is out of the valid range [0, %lld) for Z%lld.\n",
        static_cast<long long>(qnum), static_cast<long long>(n), static_cast<long long>(n));
    }

    void ValidateZnQnums(const std::vector<cytnx_int64> &qnums, const cytnx_int64 n) {
      for (const auto &q : qnums) ValidateZnQnum(q, n);
    }
  }  // namespace

  bool cytnx::Symmetry::operator==(const cytnx::Symmetry &rhs) const {
    return (this->stype() == rhs.stype()) && (this->n() == rhs.n());
  }
  bool cytnx::Symmetry::operator!=(const cytnx::Symmetry &rhs) const { return !(*this == rhs); }

  ///=========================
  bool cytnx::U1Symmetry::check_qnum(const cytnx_int64 &qnum) const { return true; }
  bool cytnx::U1Symmetry::check_qnums(const std::vector<cytnx_int64> &qnums) const { return true; }

  void cytnx::U1Symmetry::combine_rule_(std::vector<cytnx_int64> &out,
                                        const std::vector<cytnx_int64> &inL,
                                        const std::vector<cytnx_int64> &inR) const {
    out.resize(inL.size() * inR.size());
    for (cytnx_uint64 i = 0; i < out.size(); i++) {
      out[i] = inL[cytnx_uint64(i / inR.size())] + inR[i % inR.size()];
    }
  }
  void cytnx::U1Symmetry::combine_rule_(cytnx_int64 &out, const cytnx_int64 &inL,
                                        const cytnx_int64 &inR, const bool &is_reverse) const {
    if (is_reverse)
      this->reverse_rule_(out, inL + inR);
    else
      out = inL + inR;
  }
  void cytnx::U1Symmetry::reverse_rule_(cytnx_int64 &out, const cytnx_int64 &in) const {
    out = in * -1;
  }

  fermionParity cytnx::U1Symmetry::get_fermion_parity(const cytnx_int64 &in_qnum) const {
    return EVEN;
  }

  void cytnx::U1Symmetry::print_info() const {
    std::cout << "--------------------\n";
    std::cout << "[Symmetry]" << std::endl;
    std::cout << "type : Abelian, U1" << std::endl;
    std::cout << "combine rule : Q1 + Q2" << std::endl;
    std::cout << "reverse rule : Q*(-1) " << std::endl;
    std::cout << "--------------------\n";
  }

  ///========================
  bool cytnx::ZnSymmetry::check_qnum(const cytnx_int64 &qnum) const {
    return (qnum >= 0) && (qnum < this->n);
  }

  bool cytnx::ZnSymmetry::check_qnums(const std::vector<cytnx_int64> &qnums) const {
    bool out = true;
    for (cytnx_uint64 i = 0; i < qnums.size(); i++) {
      out = ((qnums[i] >= 0) && (qnums[i] < this->n));
      if (out == false) break;
    }
    return out;
  }

  void cytnx::ZnSymmetry::combine_rule_(std::vector<cytnx_int64> &out,
                                        const std::vector<cytnx_int64> &inL,
                                        const std::vector<cytnx_int64> &inR) const {
    ValidateZnQnums(inL, this->n);
    ValidateZnQnums(inR, this->n);
    out.resize(inL.size() * inR.size());
    for (cytnx_uint64 i = 0; i < inL.size(); i++) {
      for (cytnx_uint64 j = 0; j < inR.size(); j++) {
        out[i * inR.size() + j] = (inL[i] + inR[j]) % this->n;
      }
    }
  }
  void cytnx::ZnSymmetry::combine_rule_(cytnx_int64 &out, const cytnx_int64 &inL,
                                        const cytnx_int64 &inR, const bool &is_reverse) const {
    ValidateZnQnum(inL, this->n);
    ValidateZnQnum(inR, this->n);
    const cytnx_int64 combined = (inL + inR) % this->n;
    if (is_reverse)
      this->reverse_rule_(out, combined);
    else
      out = combined;
  }
  void cytnx::ZnSymmetry::reverse_rule_(cytnx_int64 &out, const cytnx_int64 &in) const {
    ValidateZnQnum(in, this->n);
    out = (this->n - in) % this->n;
  }

  fermionParity cytnx::ZnSymmetry::get_fermion_parity(const cytnx_int64 &in_qnum) const {
    return EVEN;
  }

  void cytnx::ZnSymmetry::print_info() const {
    std::cout << "--------------------" << std::endl;
    std::cout << "[Symmetry]" << std::endl;
    std::cout << "type : Abelian, Z(" << this->n << ")" << std::endl;
    std::cout << "combine rule : (Q1 + Q2)%" << this->n << std::endl;
    std::cout << "reverse rule : Q*(-1) " << std::endl;
    std::cout << "--------------------" << std::endl;
  }

  ///========================
  bool cytnx::FermionParitySymmetry::check_qnum(const cytnx_int64 &qnum) const {
    return (qnum >= 0) && (qnum < 2);
  }

  bool cytnx::FermionParitySymmetry::check_qnums(const std::vector<cytnx_int64> &qnums) const {
    // Delegates to check_qnum so the two forms cannot diverge (#1012): the old
    // hand-written comparison tested against this->n, which for fPar is the -2
    // stype sentinel rather than a modulus, rejecting every non-empty input.
    for (const auto &qnum : qnums) {
      if (!this->check_qnum(qnum)) return false;
    }
    return true;
  }

  void cytnx::FermionParitySymmetry::combine_rule_(std::vector<cytnx_int64> &out,
                                                   const std::vector<cytnx_int64> &inL,
                                                   const std::vector<cytnx_int64> &inR) const {
    out.resize(inL.size() * inR.size());
    for (cytnx_uint64 i = 0; i < out.size(); i++) {
      out[i] = (inL[cytnx_uint64(i / inR.size())] + inR[i % inR.size()]) % 2;
    }
  }
  void cytnx::FermionParitySymmetry::combine_rule_(cytnx_int64 &out, const cytnx_int64 &inL,
                                                   const cytnx_int64 &inR,
                                                   const bool &is_reverse) const {
    if (is_reverse)
      this->reverse_rule_(out, (inL + inR) % 2);
    else
      out = (inL + inR) % 2;
  }
  void cytnx::FermionParitySymmetry::reverse_rule_(cytnx_int64 &out, const cytnx_int64 &in) const {
    // out = -in<0?-in+this->n:-in;
    out = -in + 2;
  }

  fermionParity cytnx::FermionParitySymmetry::get_fermion_parity(const cytnx_int64 &in_qnum) const {
    // std::vector<fermionParity> out(this->n);
    if (in_qnum == 0)
      return EVEN;
    else if (in_qnum == 1)
      return ODD;
    else
      cytnx_error_msg(true, "[ERROR] invalid qunatum number for fermionic parity symmetry.%s",
                      "\n");
  }

  void cytnx::FermionParitySymmetry::print_info() const {
    std::cout << "--------------------" << std::endl;
    std::cout << "[Symmetry]" << std::endl;
    std::cout << "type : fermionic, FermionParity" << std::endl;
    std::cout << "combine rule : (Q1 + Q2)%2" << std::endl;
    std::cout << "reverse rule : Q*(-1) " << std::endl;
    std::cout << "--------------------" << std::endl;
  }

  ///=========================
  bool cytnx::FermionNumberSymmetry::check_qnum(const cytnx_int64 &qnum) const { return true; }
  bool cytnx::FermionNumberSymmetry::check_qnums(const std::vector<cytnx_int64> &qnums) const {
    return true;
  }

  void cytnx::FermionNumberSymmetry::combine_rule_(std::vector<cytnx_int64> &out,
                                                   const std::vector<cytnx_int64> &inL,
                                                   const std::vector<cytnx_int64> &inR) const {
    out.resize(inL.size() * inR.size());
    for (cytnx_uint64 i = 0; i < out.size(); i++) {
      out[i] = inL[cytnx_uint64(i / inR.size())] + inR[i % inR.size()];
    }
  }
  void cytnx::FermionNumberSymmetry::combine_rule_(cytnx_int64 &out, const cytnx_int64 &inL,
                                                   const cytnx_int64 &inR,
                                                   const bool &is_reverse) const {
    if (is_reverse)
      this->reverse_rule_(out, inL + inR);
    else
      out = inL + inR;
  }
  void cytnx::FermionNumberSymmetry::reverse_rule_(cytnx_int64 &out, const cytnx_int64 &in) const {
    out = in * -1;
  }

  fermionParity cytnx::FermionNumberSymmetry::get_fermion_parity(const cytnx_int64 &in_qnum) const {
    // std::vector<fermionParity> out(this->n);
    if (in_qnum % 2 == 0)
      return EVEN;
    else
      return ODD;
  }

  void cytnx::FermionNumberSymmetry::print_info() const {
    std::cout << "--------------------\n";
    std::cout << "[Symmetry]" << std::endl;
    std::cout << "type : fermionic, FermionNumber" << std::endl;
    std::cout << "combine rule : Q1 + Q2" << std::endl;
    std::cout << "reverse rule : Q*(-1) " << std::endl;
    std::cout << "--------------------\n";
  }

  //==================================================

  void cytnx::Symmetry::Save(const std::string &fname) const {
    std::fstream f;
    if (std::filesystem::path(fname).has_extension()) {
      // filename extension is given
      f.open(fname, std::ios::out | std::ios::trunc | std::ios::binary);
    } else {
      // add filename extension
      cytnx_warning_msg(true,
                        "Missing file extension in fname '%s'. I am adding the extension '.cysym'. "
                        "This is deprecated, please provide the file extension in the future.\n",
                        fname.c_str());
      f.open((fname + ".cysym"), std::ios::out | std::ios::trunc | std::ios::binary);
    }
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for save.%s", "\n");
    }
    this->_Save(f);
    f.close();
  }
  void cytnx::Symmetry::Save(const char *fname) const { this->Save(std::string(fname)); }

  cytnx::Symmetry cytnx::Symmetry::Load(const std::string &fname) {
    Symmetry out;
    std::fstream f;
    f.open(fname, std::ios::in | std::ios::binary);
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] Cannot open file '%s'.\n", fname.c_str());
    }
    out._Load(f);
    f.close();
    return out;
  }
  cytnx::Symmetry cytnx::Symmetry::Load(const char *fname) {
    return cytnx::Symmetry::Load(std::string(fname));
  }

  //==================
  void cytnx::Symmetry::_Save(std::fstream &f) const {
    cytnx_error_msg(!f.is_open(), "[ERROR][Symmetry] invalid std::fstream%s", "\n");
    unsigned int IDDs = 777;
    f.write((char *)&IDDs, sizeof(unsigned int));
    const int stype_id = this->stype();
    const int n = this->n();
    f.write((char *)&stype_id, sizeof(int));
    f.write((char *)&n, sizeof(int));
  }
  void cytnx::Symmetry::_Load(std::fstream &f) {
    cytnx_error_msg(!f.is_open(), "[ERROR][Symmetry] invalid std::fstream%s", "\n");
    unsigned int tmpIDDs;
    f.read((char *)&tmpIDDs, sizeof(unsigned int));
    cytnx_error_msg(tmpIDDs != 777, "[ERROR] the object is not a cytnx symmetry!%s", "\n");
    int stype_in;
    int n_in;
    f.read((char *)&stype_in, sizeof(int));
    f.read((char *)&n_in, sizeof(int));
    this->Init(stype_in, n_in);
  }

  std::ostream &operator<<(std::ostream &os, const Symmetry &in) {
    in.print_info();
    return os;
  }
};  // namespace cytnx
