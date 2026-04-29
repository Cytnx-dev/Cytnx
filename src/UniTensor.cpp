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

  void UniTensor::Save(const std::string &fname) const {
    fstream f;  // only for binary saving, not used for hdf5
    if (std::filesystem::path(fname).has_extension()) {
      // filename extension is given
      auto ext = std::filesystem::path(fname).extension().string();
      if (ext == ".h5" || ext == ".hdf5" || ext == ".H5" || ext == ".HDF5" || ext == ".hdf" ||
          ext == ".HDF") {
        // save as hdf5
        H5::H5File h5file;
        try {
          h5file = H5::H5File(fname, H5F_ACC_TRUNC);
        } catch (H5::FileIException &error) {
          error.printErrorStack();
          cytnx_error_msg(true, "[ERROR] Cannot create HDF5 file '%s'.\n", fname.c_str());
        }
        this->to_hdf5(h5file);
        h5file.close();
        return;
      } else {  // create binary file
        f.open(fname, ios::out | ios::trunc | ios::binary);
      }
    } else {  // create binary file with standard extension
      cytnx_warning_msg(true,
                        "Missing file extension in fname '%s'. I am adding the extension '.cytnx'. "
                        "This is deprecated, please provide the file extension in the future.\n",
                        fname.c_str());
      f.open((fname + ".cytnx"), ios::out | ios::trunc | ios::binary);
    }
    // write binary
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for save.%s", "\n");
    }
    this->to_binary(f);
    f.close();
  }
  void UniTensor::Save(const char *fname) const { Save(string(fname)); }

  UniTensor UniTensor::Load(const std::string &fname, const bool restore_device) {
    UniTensor out;
    out.Load_(fname, restore_device);
    return out;
  }
  UniTensor UniTensor::Load(const char *fname, const bool restore_device) {
    return UniTensor::Load(string(fname), restore_device);
  }

  void UniTensor::Load_(const std::string &fname, const bool restore_device) {
    auto ext = std::filesystem::path(fname).extension().string();
    if (ext == ".h5" || ext == ".hdf5" || ext == ".H5" || ext == ".HDF5" || ext == ".hdf" ||
        ext == ".HDF") {
      // load hdf5
      H5::H5File h5file;
      try {
        h5file = H5::H5File(fname, H5F_ACC_RDONLY);
      } catch (H5::FileIException &error) {
        error.printErrorStack();
        cytnx_error_msg(true, "[ERROR] Cannot open HDF5 file '%s'.\n", fname.c_str());
      }
      this->from_hdf5(h5file, "Tensor", restore_device);
      h5file.close();
    } else {  // load binary
      fstream f;
      f.open(fname, ios::in | ios::binary);
      if (!f.is_open()) {
        cytnx_error_msg(true, "[ERROR] Cannot open file '%s'.\n", fname.c_str());
      }
      this->from_binary(f, restore_device);
      f.close();
    }
  }
  void UniTensor::Load_(const char *fname, const bool restore_device) {
    this->Load_(string(fname), restore_device);
  }

  void UniTensor::to_hdf5(H5::Group &location, const std::string &name) const {
    cytnx_error_msg(true, "[ERROR] Saving UniTensor to HDF5 is not implemented yet!%s", "\n");
  }
  void UniTensor::from_hdf5(H5::Group &location, const std::string &name,
                            const bool restore_device) {
    cytnx_error_msg(true, "[ERROR] Loading UniTensor from HDF5 is not implemented yet!%s", "\n");
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
