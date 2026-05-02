#include "UniTensor.hpp"

#include <filesystem>
#include <typeinfo>

#include "H5Cpp.h"

#include "linalg.hpp"
#include "random.hpp"
#include "utils/utils.hpp"

using namespace std;

#ifdef BACKEND_TORCH
#else

namespace cytnx {

  void UniTensor::Init(const std::string name) {
    if (name == UTenType.getname(UTenType.Block)) {
      boost::intrusive_ptr<UniTensor_base> tmp(new BlockUniTensor);
      this->_impl = tmp;
    } else if (name == UTenType.getname(UTenType.BlockFermionic)) {
      boost::intrusive_ptr<UniTensor_base> tmp(new BlockFermionicUniTensor);
      this->_impl = tmp;
    } else if (name == UTenType.getname(UTenType.Dense)) {
      boost::intrusive_ptr<UniTensor_base> tmp(new DenseUniTensor);
      this->_impl = tmp;
    } else if (name == UTenType.getname(UTenType.Void)) {
      boost::intrusive_ptr<UniTensor_base> tmp(new UniTensor_base);
      this->_impl = tmp;
    } else if (name == UTenType.getname(UTenType.Sparse)) {
      cytnx_error_msg(true, "[ERROR] SparseUniTensor is deprecated.\s", "\n");
    } else {
      cytnx_error_msg(true, "[ERROR] No UniTensor type matches the string '%s'.\n", name.c_str());
    }
  }

  UniTensor UniTensor::Pow(const double &p) const { return cytnx::linalg::Pow(*this, p); }
  UniTensor &UniTensor::Pow_(const double &p) {
    cytnx::linalg::Pow_(*this, p);
    return *this;
  }

  UniTensor UniTensor::Inv(double clip) const { return cytnx::linalg::Inv(*this, clip); }
  UniTensor &UniTensor::Inv_(double clip) {
    cytnx::linalg::Inv_(*this, clip);
    return *this;
  }

  UniTensor UniTensor::Add(const UniTensor &rhs) const { return cytnx::linalg::Add(*this, rhs); }
  UniTensor UniTensor::Add(const Scalar &rhs) const {
    // cout << "lyer1: " << rhs << endl;
    return cytnx::linalg::Add(*this, rhs);
  }

  UniTensor UniTensor::Sub(const UniTensor &rhs) const { return cytnx::linalg::Sub(*this, rhs); }
  UniTensor UniTensor::Sub(const Scalar &rhs) const { return cytnx::linalg::Sub(*this, rhs); }

  UniTensor UniTensor::Div(const UniTensor &rhs) const { return cytnx::linalg::Div(*this, rhs); }
  UniTensor UniTensor::Div(const Scalar &rhs) const { return cytnx::linalg::Div(*this, rhs); }

  UniTensor UniTensor::Mul(const UniTensor &rhs) const { return cytnx::linalg::Mul(*this, rhs); }
  UniTensor UniTensor::Mul(const Scalar &rhs) const { return cytnx::linalg::Mul(*this, rhs); }

  void UniTensor::Save(const std::filesystem::path &fname, const std::string &path,
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
        // create group
        std::filesystem::path grouppath(path);
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
      fnameext += ".cytnx";
      cytnx_warning_msg(true,
                        "Missing file extension in fname '%s'. I am adding the extension '.cytnx'. "
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
  void UniTensor::Save(const char *fname, const std::string &path, const char mode) const {
    this->Save(std::filesystem::path(fname), path, mode);
  }

  UniTensor UniTensor::Load(const std::filesystem::path &fname, const std::string &path,
                            const bool restore_device) {
    UniTensor out;
    out.Load_(fname, path, restore_device);
    return out;
  }
  UniTensor UniTensor::Load(const char *fname, const std::string &path, const bool restore_device) {
    return UniTensor::Load(std::filesystem::path(fname), path, restore_device);
  }

  void UniTensor::Load_(const std::filesystem::path &fname, const std::string &path,
                        const bool restore_device) {
    std::string ext = fname.extension().string();
    if (ext == ".h5" || ext == ".hdf5" || ext == ".H5" || ext == ".HDF5" || ext == ".hdf" ||
        ext == ".HDF") {  // load HDF5
      H5::H5File h5file(fname, H5F_ACC_RDONLY);
      // open group
      H5::Group location;
      try {
        location = h5file.openGroup(path.empty() ? "/" : path);
      } catch (const H5::Exception &e) {
        std::cerr << e.getDetailMsg() << std::endl;
        cytnx_error_msg(true, "[ERROR] HDF5 path '%s' not found or is not a group in file '%s'.",
                        path.c_str(), fname.c_str());
      }
      // read data
      this->from_hdf5(location, restore_device);
      h5file.close();
    } else {  // load binary
      fstream f;
      f.open(fname, std::ios::in | std::ios::binary);
      if (!f.is_open()) {
        cytnx_error_msg(true, "[ERROR] Cannot open file '%s'.\n", fname.c_str());
      }
      this->from_binary(f, restore_device);
      f.close();
    }
  }
  void UniTensor::Load_(const char *fname, const std::string &path, const bool restore_device) {
    this->Load_(std::filesystem::path(fname), path, restore_device);
  }

  void UniTensor::to_hdf5(H5::Group &location, const bool overwrite) const {
    if (overwrite) {  // delete previous data
      // delete all entries that could be written by one of the implementations;
      // remove attributes
      if (location.attrExists("type")) location.removeAttr("type");
      if (location.attrExists("diagonal")) location.removeAttr("diagonal");
      if (location.attrExists("rowrank")) location.removeAttr("rowrank");
      if (location.attrExists("name")) location.removeAttr("name");
      if (location.attrExists("directed")) location.removeAttr("directed");
      // remove datasets
      if (location.nameExists("labels")) location.unlink("labels");
      if (location.nameExists("Tensor")) location.unlink("Tensor");
      if (location.nameExists("block_to_sectors")) location.unlink("block_to_sectors");
      // remove groups and its contents recursively
      if (location.nameExists("bonds")) location.unlink("bonds");
      if (location.nameExists("blocks")) location.unlink("blocks");
    }

    H5::DataType datatype;
    H5::Attribute attr;
    H5::DataSet dataset;
    H5::DataSpace dataspace;
    H5::StrType str_type;

    // type, write as string attribute
    std::string type = UTenType.getname(this->_impl->uten_type_id);
    str_type = H5::StrType(H5::PredType::C_S1, type.length() + 1);
    dataspace = H5::DataSpace(H5S_SCALAR);
    attr = location.createAttribute("type", str_type, dataspace);
    attr.write(str_type, type);

    // is_diag, write as attribute only for diagonal tensors
    if (this->_impl->_is_diag) {
      datatype = Type.get_hdf5_type(this->_impl->_is_diag);
      attr = location.createAttribute("diagonal", datatype, H5::DataSpace(H5S_SCALAR));
      attr.write(H5::PredType::NATIVE_INT, &this->_impl->_is_diag);
    }

    // rowrank, write as attribute
    datatype = Type.get_hdf5_type(this->_impl->_rowrank);
    attr = location.createAttribute("rowrank", datatype, H5::DataSpace(H5S_SCALAR));
    attr.write(H5::PredType::NATIVE_INT, &this->_impl->_rowrank);

    // name, write as string attribute
    str_type = H5::StrType(H5::PredType::C_S1, this->_impl->_name.length() + 1);
    dataspace = H5::DataSpace(H5S_SCALAR);
    attr = location.createAttribute("name", str_type, dataspace);
    attr.write(str_type, this->_impl->_name);

    // labels; write as string vector
    if (!this->_impl->_labels.empty()) {
      hsize_t vecdims[1] = {this->_impl->_labels.size()};
      dataspace = H5::DataSpace(1, vecdims);
      str_type = H5::StrType(H5::PredType::C_S1, H5T_VARIABLE);
      dataset = location.createDataSet("labels", str_type, dataspace);
      std::vector<const char *> c_strings;  // H5 needs cstrings
      std::string symstring;
      for (const auto &label : this->_impl->_labels) {
        c_strings.push_back(label.c_str());
      }
      dataset.write(c_strings.data(), str_type);
    }

    // bonds; write in group
    if (!this->_impl->_bonds.empty()) {
      H5::Group dir = location.createGroup("bonds");
      for (int i = 0; i < this->_impl->_bonds.size(); i++) {
        H5::Group bondgroup = dir.createGroup("Bond" + std::to_string(i));
        this->_impl->_bonds[i].to_hdf5(bondgroup, overwrite);
      }
    }

    this->_impl->to_hdf5_dispatch(location, overwrite);
  }

  void UniTensor::from_hdf5(H5::Group &location, const bool restore_device) {
    H5::DataType datatype;
    H5::Attribute attr;
    H5::StrType str_type;
    size_t size;

    // type, read from attribute
    attr = location.openAttribute("type");
    str_type = attr.getStrType();
    size = str_type.getSize() - 1;  // remove the null terminator
    std::string utenname;
    utenname.resize(size);
    attr.read(str_type, &utenname[0]);
    this->Init(utenname);

    // is_diag, read from attribute
    if (location.attrExists("diagonal")) {
      H5::Attribute attr = location.openAttribute("diagonal");
      datatype = attr.getDataType();
      cytnx_error_msg(
        datatype.getSize() != Type.get_hdf5_type(this->_impl->_is_diag).getSize(),
        "[ERROR] 'device' bit-length mismatch. File: %zu bytes, expected: %zu bytes.\n",
        datatype.getSize(), Type.get_hdf5_type(this->_impl->_is_diag).getSize());
      attr.read(datatype, &this->_impl->_is_diag);
    } else {
      this->_impl->_is_diag = false;
    }

    // rowrank, read from attribute
    attr = location.openAttribute("rowrank");
    datatype = attr.getDataType();
    cytnx_error_msg(
      datatype.getSize() != Type.get_hdf5_type(this->_impl->_rowrank).getSize(),
      "[ERROR] 'rowrank' bit-length mismatch. File: %zu bytes, expected: %zu bytes.\n",
      datatype.getSize(), Type.get_hdf5_type(this->_impl->_rowrank).getSize());
    attr.read(datatype, &this->_impl->_rowrank);

    // name, read from string attribute
    if (location.attrExists("name")) {
      attr = location.openAttribute("name");
      str_type = attr.getStrType();
      size = str_type.getSize() - 1;  // remove the null terminator
      this->_impl->_name.resize(size);
      attr.read(str_type, &this->_impl->_name[0]);
    } else {
      this->_impl->_name.clear();
    }

    // labels; read from string vector
    this->_impl->_labels.clear();
    if (location.exists("labels")) {
      H5::DataSet dataset = location.openDataSet("labels");
      H5::DataSpace dataspace = dataset.getSpace();
      hsize_t dims[1];
      dataspace.getSimpleExtentDims(dims);
      // H5T_VARIABLE requires reading into an array of char pointers (char**)
      H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
      std::vector<char *> c_strings(dims[0]);
      dataset.read(c_strings.data(), str_type);
      this->_impl->_labels.reserve(dims[0]);
      for (size_t i = 0; i < dims[0]; ++i) {
        this->_impl->_labels.push_back(std::string(c_strings[i]));
      }
      // free the space of each char* that was allocated in dataset.read()
      dataset.vlenReclaim(c_strings.data(), str_type, dataspace);
    }

    // bonds; read from group
    this->_impl->_bonds.clear();
    if (location.exists("bonds")) {
      H5::Group dir = location.openGroup("bonds");
      hsize_t idx = 0;
      while (true) {
        std::string name = "Bond" + std::to_string(idx);
        if (!dir.exists(name)) {
          break;
        }
        H5::Group bondgroup = dir.openGroup(name);
        Bond bond;
        bond.from_hdf5(bondgroup);
        this->_impl->_bonds.push_back(bond);
        idx++;
      }
      cytnx_error_msg(
        idx != this->_impl->_labels.size(),
        "[ERROR] %d bonds were found, but %d labels exist. The HDF5 data seems corrupt!\n", idx,
        this->_impl->_labels.size());
    } else {
      cytnx_error_msg(
        !this->_impl->_labels.empty(),
        "[ERROR] %d labels exist, but no bonds were found. The HDF5 data seems corrupt!\n",
        this->_impl->_labels.size());
      this->_impl->_bonds.clear();
    }

    this->_impl->from_hdf5_dispatch(location, restore_device);
    this->_impl->_is_braket_form = this->_impl->_update_braket();
  }

  void UniTensor::to_binary(std::ostream &f) const {
    cytnx_error_msg(this->_impl->uten_type_id == UTenType.Void,
                    "[ERROR][UniTensor] Cannot save an uninitialized UniTensor.%s", "\n");

    // temporary disable:
    // cytnx_error_msg(this->_impl->uten_type_id==UTenType.Sparse,"[ERROR] Save for SparseUniTensor
    // is under developing!!%s","\n");

    if (this->_impl->uten_type_id == UTenType.Sparse)
      cytnx_error_msg(this->is_contiguous() == false,
                      "[ERROR] Save for SparseUniTensor requires it to be contiguous. Call "
                      "UniTensor.contiguous() first. %s",
                      "\n");

    unsigned int IDDs = 555;
    f.write((char *)&IDDs, sizeof(unsigned int));
    // first, save common meta data:
    f.write((char *)&this->_impl->uten_type_id,
            sizeof(int));  // uten type, this is used to determine Sparse/Dense upon load
    f.write((char *)&this->_impl->_is_braket_form, sizeof(bool));
    f.write((char *)&this->_impl->_is_tag, sizeof(bool));
    f.write((char *)&this->_impl->_is_diag, sizeof(bool));
    f.write((char *)&this->_impl->_rowrank, sizeof(cytnx_int64));

    cytnx_uint32 len_name = this->_impl->_name.size();
    f.write((char *)&len_name, sizeof(cytnx_uint32));
    if (len_name != 0) {
      const char *cname = this->_impl->_name.c_str();
      f.write(cname, sizeof(char) * len_name);
    }

    cytnx_uint64 rank = this->_impl->_labels.size();
    f.write((char *)&rank, sizeof(cytnx_uint64));
    for (cytnx_uint64 i = 0; i < rank; i++) {
      size_t tmp = this->_impl->_labels[i].size();
      f.write((char *)&tmp, sizeof(this->_impl->_labels[i].size()));
    }
    for (cytnx_uint64 i = 0; i < rank; i++) {
      f.write((char *)(this->_impl->_labels[i].data()),
              sizeof(char) * this->_impl->_labels[i].size());
    }
    // f.write((char *)&(this->_impl->_labels[0]), sizeof(cytnx_int64) * rank);
    for (cytnx_uint64 i = 0; i < rank; i++) {
      this->_impl->_bonds[i].to_binary(f);
    }

    // second, let dispatch to do remaining saving.
    this->_impl->to_binary_dispatch(f);
  }

  void UniTensor::from_binary(std::istream &f, const bool restore_device) {
    unsigned int tmpIDDs;
    f.read((char *)&tmpIDDs, sizeof(unsigned int));
    cytnx_error_msg(tmpIDDs != 555, "[ERROR] the object is not a cytnx UniTensor!%s", "\n");

    int utentype;
    f.read((char *)&utentype,
           sizeof(int));  // uten type, this is used to determine Sparse/Dense upon load
    if (utentype == UTenType.Dense) {
      this->_impl = boost::intrusive_ptr<UniTensor_base>(new DenseUniTensor());
    } else if (utentype == UTenType.Sparse) {
      // temporary disable:
      // cytnx_error_msg(this->_impl->uten_type_id==UTenType.Sparse,"[ERROR] Save for
      // SparseUniTensor is under developing!!%s","\n");
      // this->_impl = boost::intrusive_ptr<UniTensor_base>(new SparseUniTensor());
      cytnx_error_msg(true,
                      "[ERROR] The file contains a SparseUniTensor, which is deprecated. It was "
                      "either saved with an old Cytnx version or something went wrong!%s",
                      "\n");
    } else if (utentype == UTenType.Block) {
      this->_impl = boost::intrusive_ptr<UniTensor_base>(new BlockUniTensor());
    } else if (utentype == UTenType.BlockFermionic) {
      this->_impl = boost::intrusive_ptr<UniTensor_base>(new BlockFermionicUniTensor());
    } else {
      cytnx_error_msg(true, "[ERROR] Unknown UniTensor type!%s", "\n");
    }

    f.read((char *)&this->_impl->_is_braket_form, sizeof(bool));
    f.read((char *)&this->_impl->_is_tag, sizeof(bool));
    f.read((char *)&this->_impl->_is_diag, sizeof(bool));
    f.read((char *)&this->_impl->_rowrank, sizeof(cytnx_int64));

    cytnx_uint32 len_name;
    f.read((char *)&len_name, sizeof(cytnx_uint32));
    if (len_name != 0) {
      char *cname = (char *)malloc(sizeof(char) * len_name);
      f.read(cname, sizeof(char) * len_name);
      this->_impl->_name = std::string(cname, len_name);
      free(cname);
    }

    cytnx_uint64 rank;
    f.read((char *)&rank, sizeof(cytnx_uint64));
    this->_impl->_labels.resize(rank);
    this->_impl->_bonds.resize(rank);
    for (cytnx_uint64 i = 0; i < rank; i++) {
      size_t tmp;
      f.read((char *)&tmp, sizeof(size_t));
      this->_impl->_labels[i].resize(tmp);
    }
    for (cytnx_uint64 i = 0; i < rank; i++) {
      f.read((char *)(this->_impl->_labels[i].data()),
             sizeof(char) * this->_impl->_labels[i].size());
    }
    // f.read((char *)&(this->_impl->_labels[0]), sizeof(cytnx_int64) * rank);
    for (cytnx_uint64 i = 0; i < rank; i++) {
      this->_impl->_bonds[i].from_binary(f);
    }

    // second, let dispatch to do remaining loading.
    this->_impl->from_binary_dispatch(f, restore_device);
  }

  // Random Generators:
  UniTensor UniTensor::normal(const cytnx_uint64 &Nelem, const double &mean, const double &std,
                              const std::vector<std::string> &in_labels, const unsigned int &seed,
                              const unsigned int &dtype, const int &device,
                              const std::string &name) {
    return UniTensor(cytnx::random::normal(Nelem, mean, std, device, seed, dtype), false, -1,
                     in_labels, name);
  }
  UniTensor UniTensor::normal(const std::vector<cytnx_uint64> &shape, const double &mean,
                              const double &std, const std::vector<std::string> &in_labels,
                              const unsigned int &seed, const unsigned int &dtype,
                              const int &device, const std::string &name) {
    return UniTensor(cytnx::random::normal(shape, mean, std, device, seed, dtype), false, -1,
                     in_labels, name);
  }

  UniTensor UniTensor::uniform(const cytnx_uint64 &Nelem, const double &low, const double &high,
                               const std::vector<std::string> &in_labels, const unsigned int &seed,
                               const unsigned int &dtype, const int &device,
                               const std::string &name) {
    return UniTensor(cytnx::random::uniform(Nelem, low, high, device, seed, dtype), false, -1,
                     in_labels, name);
  }
  UniTensor UniTensor::uniform(const std::vector<cytnx_uint64> &shape, const double &low,
                               const double &high, const std::vector<std::string> &in_labels,
                               const unsigned int &seed, const unsigned int &dtype,
                               const int &device, const std::string &name) {
    return UniTensor(cytnx::random::uniform(shape, low, high, device, seed, dtype), false, -1,
                     in_labels, name);
  }

  // Inplace Random Generators:
  UniTensor &UniTensor::normal_(const double &mean, const double &std, const unsigned int &seed) {
    if (this->uten_type() == UTenType.Dense) {
      cytnx::random::normal_(this->get_block_(), mean, std, seed);
    } else if (this->uten_type() == UTenType.Block ||
               this->uten_type() == UTenType.BlockFermionic) {
      for (auto &blk : this->get_blocks_()) {
        cytnx::random::normal_(blk, mean, std, seed);
      }
    } else {
      cytnx_error_msg(true,
                      "[ERROR] Cannot perform inplace random generation on a UniTensor which is "
                      "not Dense or Block.%s",
                      "\n");
    }
    return *this;
  }

  UniTensor &UniTensor::uniform_(const double &low, const double &high, const unsigned int &seed) {
    if (this->uten_type() == UTenType.Dense) {
      cytnx::random::uniform_(this->get_block_(), low, high, seed);
    } else if (this->uten_type() == UTenType.Block ||
               this->uten_type() == UTenType.BlockFermionic) {
      for (auto &blk : this->get_blocks_()) {
        cytnx::random::uniform_(blk, low, high, seed);
      }
    } else {
      cytnx_error_msg(true,
                      "[ERROR] Cannot perform inplace random generation on a UniTensor which is "
                      "not Dense or Block.%s",
                      "\n");
    }
    return *this;
  }

}  // namespace cytnx
#endif
