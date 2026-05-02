#include "Symmetry.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "H5Cpp.h"

using namespace std;

namespace cytnx {

  bool cytnx::Symmetry::operator==(const cytnx::Symmetry &rhs) const {
    return (this->stype() == rhs.stype()) && (this->n() == rhs.n());
  }
  bool cytnx::Symmetry::operator!=(const cytnx::Symmetry &rhs) const { return !(*this == rhs); }
  //=============================

  std::vector<cytnx_int64> Symmetry_base::combine_rule(const std::vector<cytnx_int64> &inL,
                                                       const std::vector<cytnx_int64> &inR) {
    std::vector<cytnx_int64> out;
    this->combine_rule_(out, inL, inR);
    return out;
  }

  cytnx_int64 Symmetry_base::combine_rule(const cytnx_int64 &inL, const cytnx_int64 &inR,
                                          const bool &is_reverse) {
    cytnx_int64 out;
    this->combine_rule_(out, inL, inR, is_reverse);
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
                                           const cytnx_int64 &inR, const bool &is_reverse) {
    cytnx_error_msg(1, "%s", "[ERROR][Internal] should not call Symmerty base!");
  }
  void cytnx::Symmetry_base::reverse_rule_(cytnx_int64 &out, const cytnx_int64 &in) {
    cytnx_error_msg(1, "%s", "[ERROR][Internal] should not call Symmerty base!");
  }

  fermionParity cytnx::Symmetry_base::get_fermion_parity(const cytnx_int64 &in_qnum) const {
    return EVEN;
  }

  void cytnx::Symmetry_base::print_info() const {
    cytnx_error_msg(1, "%s", "[ERROR][Internal] should not call Symmerty base!");
  }

  std::string cytnx::Symmetry_base::getname() const {
    cytnx_error_msg(1, "%s", "[ERROR][Internal] should not call Symmerty base!");
  }

  std::string cytnx::Symmetry_base::stype_str() const {
    cytnx_error_msg(1, "%s", "[ERROR][Internal] should not call Symmerty base!");
  }

  ///=========================
  bool cytnx::U1Symmetry::check_qnum(const cytnx_int64 &qnum) { return true; }
  bool cytnx::U1Symmetry::check_qnums(const std::vector<cytnx_int64> &qnums) { return true; }

  void cytnx::U1Symmetry::combine_rule_(std::vector<cytnx_int64> &out,
                                        const std::vector<cytnx_int64> &inL,
                                        const std::vector<cytnx_int64> &inR) {
    out.resize(inL.size() * inR.size());
    for (cytnx_uint64 i = 0; i < out.size(); i++) {
      out[i] = inL[cytnx_uint64(i / inR.size())] + inR[i % inR.size()];
    }
  }
  void cytnx::U1Symmetry::combine_rule_(cytnx_int64 &out, const cytnx_int64 &inL,
                                        const cytnx_int64 &inR, const bool &is_reverse) {
    if (is_reverse)
      this->reverse_rule_(out, inL + inR);
    else
      out = inL + inR;
  }
  void cytnx::U1Symmetry::reverse_rule_(cytnx_int64 &out, const cytnx_int64 &in) { out = in * -1; }

  void cytnx::U1Symmetry::print_info() const {
    cout << "--------------------\n";
    cout << "[Symmetry]" << endl;
    cout << "type : Abelian, " << this->getname() << endl;
    cout << "combine rule : Q1 + Q2" << endl;
    cout << "reverse rule : Q*(-1) " << endl;
    cout << "--------------------\n";
  }

  ///========================
  bool cytnx::ZnSymmetry::check_qnum(const cytnx_int64 &qnum) {
    return (qnum >= 0) && (qnum < this->n);
  }

  bool cytnx::ZnSymmetry::check_qnums(const std::vector<cytnx_int64> &qnums) {
    bool out = true;
    for (cytnx_uint64 i = 0; i < qnums.size(); i++) {
      out = ((qnums[i] >= 0) && (qnums[i] < this->n));
      if (out == false) break;
    }
    return out;
  }

  void cytnx::ZnSymmetry::combine_rule_(std::vector<cytnx_int64> &out,
                                        const std::vector<cytnx_int64> &inL,
                                        const std::vector<cytnx_int64> &inR) {
    out.resize(inL.size() * inR.size());
    for (cytnx_uint64 i = 0; i < out.size(); i++) {
      out[i] = (inL[cytnx_uint64(i / inR.size())] + inR[i % inR.size()]) % (this->n);
    }
  }
  void cytnx::ZnSymmetry::combine_rule_(cytnx_int64 &out, const cytnx_int64 &inL,
                                        const cytnx_int64 &inR, const bool &is_reverse) {
    if (is_reverse)
      this->reverse_rule_(out, (inL + inR) % (this->n));
    else
      out = (inL + inR) % (this->n);
  }
  void cytnx::ZnSymmetry::reverse_rule_(cytnx_int64 &out, const cytnx_int64 &in) {
    // out = -in<0?-in+this->n:-in;
    out = -in + this->n;
  }

  void cytnx::ZnSymmetry::print_info() const {
    cout << "--------------------\n";
    cout << "[Symmetry]" << endl;
    cout << "type : Abelian, " << this->getname() << endl;
    cout << "combine rule : (Q1 + Q2)\%" << this->n << endl;
    cout << "reverse rule : Q*(-1) " << endl;
    cout << "--------------------\n";
  }

  ///========================
  bool cytnx::FermionParitySymmetry::check_qnum(const cytnx_int64 &qnum) {
    return (qnum >= 0) && (qnum < 2);
  }

  bool cytnx::FermionParitySymmetry::check_qnums(const std::vector<cytnx_int64> &qnums) {
    bool out = true;
    for (cytnx_uint64 i = 0; i < qnums.size(); i++) {
      out = ((qnums[i] >= 0) && (qnums[i] < this->n));
      if (out == false) break;
    }
    return out;
  }

  void cytnx::FermionParitySymmetry::combine_rule_(std::vector<cytnx_int64> &out,
                                                   const std::vector<cytnx_int64> &inL,
                                                   const std::vector<cytnx_int64> &inR) {
    out.resize(inL.size() * inR.size());
    for (cytnx_uint64 i = 0; i < out.size(); i++) {
      out[i] = (inL[cytnx_uint64(i / inR.size())] + inR[i % inR.size()]) % 2;
    }
  }
  void cytnx::FermionParitySymmetry::combine_rule_(cytnx_int64 &out, const cytnx_int64 &inL,
                                                   const cytnx_int64 &inR, const bool &is_reverse) {
    if (is_reverse)
      this->reverse_rule_(out, (inL + inR) % 2);
    else
      out = (inL + inR) % 2;
  }
  void cytnx::FermionParitySymmetry::reverse_rule_(cytnx_int64 &out, const cytnx_int64 &in) {
    // out = -in<0?-in+this->n:-in;
    out = -in + 2;
  }

  fermionParity cytnx::FermionParitySymmetry::get_fermion_parity(const cytnx_int64 &in_qnum) const {
    // vector<fermionParity> out(this->n);
    if (in_qnum == 0)
      return EVEN;
    else if (in_qnum == 1)
      return ODD;
    else
      cytnx_error_msg(true, "[ERROR] invalid qunatum number for fermionic parity symmetry.%s",
                      "\n");
  }

  void cytnx::FermionParitySymmetry::print_info() const {
    cout << "--------------------\n";
    cout << "[Symmetry]" << endl;
    cout << "type : Fermionic, " << this->getname() << endl;
    cout << "combine rule : (Q1 + Q2)\%2" << endl;
    cout << "reverse rule : Q*(-1) " << endl;
    cout << "--------------------\n";
  }

  ///=========================
  bool cytnx::FermionNumberSymmetry::check_qnum(const cytnx_int64 &qnum) { return true; }
  bool cytnx::FermionNumberSymmetry::check_qnums(const std::vector<cytnx_int64> &qnums) {
    return true;
  }

  void cytnx::FermionNumberSymmetry::combine_rule_(std::vector<cytnx_int64> &out,
                                                   const std::vector<cytnx_int64> &inL,
                                                   const std::vector<cytnx_int64> &inR) {
    out.resize(inL.size() * inR.size());
    for (cytnx_uint64 i = 0; i < out.size(); i++) {
      out[i] = inL[cytnx_uint64(i / inR.size())] + inR[i % inR.size()];
    }
  }
  void cytnx::FermionNumberSymmetry::combine_rule_(cytnx_int64 &out, const cytnx_int64 &inL,
                                                   const cytnx_int64 &inR, const bool &is_reverse) {
    if (is_reverse)
      this->reverse_rule_(out, inL + inR);
    else
      out = inL + inR;
  }
  void cytnx::FermionNumberSymmetry::reverse_rule_(cytnx_int64 &out, const cytnx_int64 &in) {
    out = in * -1;
  }

  fermionParity cytnx::FermionNumberSymmetry::get_fermion_parity(const cytnx_int64 &in_qnum) const {
    // vector<fermionParity> out(this->n);
    if (in_qnum % 2 == 0)
      return EVEN;
    else
      return ODD;
  }

  void cytnx::FermionNumberSymmetry::print_info() const {
    cout << "--------------------\n";
    cout << "[Symmetry]" << endl;
    cout << "type : fermionic, " << this->getname() << endl;
    cout << "combine rule : Q1 + Q2" << endl;
    cout << "reverse rule : Q*(-1) " << endl;
    cout << "--------------------\n";
  }

  //==================================================

  void cytnx::Symmetry::Save(const std::filesystem::path &fname, const std::string &path,
                             const char mode) const {
    fstream f;  // only for binary saving, not used for HDF5
    if (fname.has_extension()) {
      // filename extension is given
      std::string ext = fname.extension().string();
      if (ext == ".h5" || ext == ".hdf5" || ext == ".H5" || ext == ".HDF5" || ext == ".hdf" ||
          ext == ".HDF") {
        // save as HDF5
        H5::H5File h5file;
        // Enable reuse of space after data is deleted;
        // Set the strategy: FSM_AGGR is standard for free-space management
        // Parameters: strategy, persist (true), threshold (default 1: track all free-space
        // sections)
        H5::FileCreatPropList fcpl;
        fcpl.setFileSpaceStrategy(H5F_FSPACE_STRATEGY_FSM_AGGR, true, 1);
        // Persistent free space requires HDF5 1.10.x format or later
        H5::FileAccPropList fapl;
        fapl.setLibverBounds(H5F_LIBVER_V200, H5F_LIBVER_LATEST);
        // open file
        bool overwrite = false;
        if (mode == 'w') {  // Write new file
          h5file = H5::H5File(fname, H5F_ACC_TRUNC, fcpl, fapl);
        } else if (mode == 'x') {  // eXclusive create
          h5file = H5::H5File(fname, H5F_ACC_EXCL, fcpl, fapl);
        } else if (mode == 'a') {  // Append data
          if (std::filesystem::exists(fname))
            h5file = H5::H5File(fname, H5F_ACC_RDWR, H5::FileCreatPropList::DEFAULT, fapl);
          else
            h5file = H5::H5File(fname, H5F_ACC_EXCL, fcpl, fapl);
        } else if (mode == 'u') {  // Update data
          if (std::filesystem::exists(fname)) {
            h5file = H5::H5File(fname, H5F_ACC_RDWR, H5::FileCreatPropList::DEFAULT, fapl);
            overwrite = true;
          } else {
            h5file = H5::H5File(fname, H5F_ACC_EXCL, fcpl, fapl);
          }
        } else {
          cytnx_error_msg(true, "[ERROR] Unknown mode '%c' for writing to HDF5 file.", mode);
        }
        // split path into group and name
        std::filesystem::path p(path);
        std::filesystem::path grouppath = p.parent_path();
        std::string datasetname = p.filename().string();
        if (datasetname.empty()) datasetname = "Symmetry";
        // create group
        std::filesystem::path subpath;
        std::string groupfolder = "/";
        for (const auto &part : grouppath) {
          if (part.empty()) continue;
          subpath /= part;
          groupfolder = subpath.generic_string();
          if (!h5file.exists(groupfolder)) h5file.createGroup(groupfolder);
        }
        H5::Group location = h5file.openGroup(groupfolder);
        // write data
        this->to_hdf5(location, overwrite, datasetname);
        h5file.close();
        return;
      } else {  // create binary file
        if (mode == 'x') {
          cytnx_error_msg(std::filesystem::exists(fname),
                          "[ERROR] File %s already exists. Use mode 'w' to overwrite.",
                          fname.string().c_str());
        } else {
          cytnx_error_msg(mode != 'w', "[ERROR] Unknown mode '%c' for writing to binary file.",
                          mode);
        }
        f.open(fname, std::ios::out | std::ios::trunc | std::ios::binary);
      }
    } else {  // create binary file with standard extension
      std::filesystem::path fnameext = fname;
      fnameext += ".cysym";
      cytnx_warning_msg(true,
                        "Missing file extension in fname '%s'. I am adding the extension '.cysym'. "
                        "This is deprecated, please provide the file extension in the future.\n",
                        fname.string().c_str());
      if (mode == 'x') {
        cytnx_error_msg(std::filesystem::exists(fnameext),
                        "[ERROR] File %s already exists. Use mode 'w' to overwrite.",
                        fnameext.string().c_str());
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
  void cytnx::Symmetry::Save(const char *fname, const std::string &path, const char mode) const {
    this->Save(std::filesystem::path(fname), path, mode);
  }

  cytnx::Symmetry cytnx::Symmetry::Load(const std::filesystem::path &fname,
                                        const std::string &path) {
    Symmetry out;
    out.Load_(fname, path);
    return out;
  }
  cytnx::Symmetry cytnx::Symmetry::Load(const char *fname, const std::string &path) {
    return cytnx::Symmetry::Load(std::filesystem::path(fname), path);
  }

  void cytnx::Symmetry::Load_(const std::filesystem::path &fname, const std::string &path) {
    std::string ext = fname.extension().string();
    if (ext == ".h5" || ext == ".hdf5" || ext == ".H5" || ext == ".HDF5" || ext == ".hdf" ||
        ext == ".HDF") {  // load HDF5
      H5::H5File h5file(fname, H5F_ACC_RDONLY);
      // split path into group and name
      std::filesystem::path p(path);
      std::string grouppath = p.parent_path().generic_string();
      std::string datasetname = p.filename().string();
      if (datasetname.empty()) datasetname = "Symmetry";
      // open group
      H5::Group location;
      try {
        location = h5file.openGroup(grouppath.empty() ? "/" : grouppath);
      } catch (const H5::Exception &e) {
        std::cerr << e.getDetailMsg() << std::endl;
        cytnx_error_msg(true, "[ERROR] HDF5 path '%s' not found or is not a group in file '%s'.",
                        grouppath.c_str(), fname.string().c_str());
      }
      // read data
      this->from_hdf5(location, datasetname);
      h5file.close();
    } else {  // load binary
      fstream f;
      f.open(fname, ios::in | ios::binary);
      if (!f.is_open()) {
        cytnx_error_msg(true, "[ERROR] Cannot open file '%s'.\n", fname.string().c_str());
      }
      this->from_binary(f);
      f.close();
    }
  }
  void cytnx::Symmetry::Load_(const char *fname, const std::string &path) {
    this->Load_(std::filesystem::path(fname), path);
  }

  void cytnx::Symmetry::to_hdf5(H5::Group &location, const bool overwrite,
                                const std::string &name) const {
    if (overwrite) {  // delete previous data
      if (location.attrExists(name)) location.removeAttr(name);
    }

    std::string symname = this->getname();
    H5::StrType str_type(H5::PredType::C_S1, symname.length() + 1);
    H5::DataSpace dataspace = H5::DataSpace(H5S_SCALAR);
    H5::Attribute attr = location.createAttribute(name, str_type, dataspace);
    attr.write(str_type, symname);
  }
  void cytnx::Symmetry::from_hdf5(H5::Group &location, const std::string &name) {
    H5::Attribute attr = location.openAttribute(name);
    H5::StrType str_type = attr.getStrType();
    size_t size = str_type.getSize() - 1;  // remove the null terminator
    std::string symname;
    symname.resize(size);
    attr.read(str_type, &symname[0]);
    this->Init(symname);
  }

  void cytnx::Symmetry::to_binary(std::ostream &f) const {
    unsigned int IDDs = 777;
    f.write((char *)&IDDs, sizeof(unsigned int));
    f.write((char *)&this->_impl->stype_id, sizeof(int));
    f.write((char *)&this->_impl->n, sizeof(int));
  }
  void cytnx::Symmetry::from_binary(std::istream &f) {
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
};  // namespace cytnx
