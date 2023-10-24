#include <typeinfo>
#include "UniTensor.hpp"
#include "utils/utils.hpp"

#include "linalg.hpp"
#include "random.hpp"

using namespace std;

#ifdef BACKEND_TORCH
#else

namespace cytnx {

  UniTensor UniTensor::Pow(const double &p) const { return cytnx::linalg::Pow(*this, p); }
  UniTensor &UniTensor::Pow_(const double &p) {
    cytnx::linalg::Pow_(*this, p);
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

  void UniTensor::_Save(std::fstream &f) const {
    cytnx_error_msg(!f.is_open(), "[ERROR][UniTensor] invalid fstream!.%s", "\n");
    cytnx_error_msg(this->_impl->uten_type_id == UTenType.Void,
                    "[ERROR][UniTensor] cannot save an uninitialize UniTensor.%s", "\n");

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
      this->_impl->_bonds[i]._Save(f);
    }

    // second, let dispatch to do remaining saving.
    this->_impl->_save_dispatch(f);
  }
  void UniTensor::_Load(std::fstream &f) {
    cytnx_error_msg(!f.is_open(), "[ERROR][UniTensor] invalid fstream%s", "\n");

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
                      "[ERROR] the file is SparseUniTensor which is deprecated. Either it's from a "
                      "erly version or something wrong!%s",
                      "\n");
    } else if (utentype == UTenType.Block) {
      this->_impl = boost::intrusive_ptr<UniTensor_base>(new BlockUniTensor());
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
      this->_impl->_name = std::string(cname);
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
      this->_impl->_bonds[i]._Load(f);
    }

    // second, let dispatch to do remaining loading.
    this->_impl->_load_dispatch(f);
  }

  void UniTensor::Save(const std::string &fname) const {
    fstream f;
    f.open((fname + ".cytnx"), ios::out | ios::trunc | ios::binary);
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for save.%s", "\n");
    }
    this->_Save(f);
    f.close();
  }
  void UniTensor::Save(const char *fname) const {
    fstream f;
    string ffname = string(fname) + ".cytnx";
    f.open((ffname), ios::out | ios::trunc | ios::binary);
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for save.%s", "\n");
    }
    this->_Save(f);
    f.close();
  }

  UniTensor UniTensor::Load(const std::string &fname) {
    UniTensor out;
    fstream f;
    f.open(fname, ios::in | ios::binary);
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for load. >> %s\n", fname.c_str());
    }
    out._Load(f);
    f.close();
    return out;
  }
  UniTensor UniTensor::Load(const char *fname) {
    UniTensor out;
    fstream f;
    f.open(fname, ios::in | ios::binary);
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for load. >> %s\n", fname);
    }
    out._Load(f);
    f.close();
    return out;
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
  void UniTensor::normal_(const double &mean, const double &std, const unsigned int &seed) {
    if (this->uten_type() == UTenType.Dense) {
      cytnx::random::normal_(this->get_block_(), mean, std, seed);
    } else if (this->uten_type() == UTenType.Block) {
      for (auto &blk : this->get_blocks_()) {
        cytnx::random::normal_(blk, mean, std, seed);
      }
    } else {
      cytnx_error_msg(true,
                      "[ERROR] cannot perform inplace random generation on a UniTensor which is "
                      "not Dense or Block.%s",
                      "\n");
    }
  }

  void UniTensor::uniform_(const double &low, const double &high, const unsigned int &seed) {
    if (this->uten_type() == UTenType.Dense) {
      cytnx::random::uniform_(this->get_block_(), low, high, seed);
    } else if (this->uten_type() == UTenType.Block) {
      for (auto &blk : this->get_blocks_()) {
        cytnx::random::uniform_(blk, low, high, seed);
      }
    } else {
      cytnx_error_msg(true,
                      "[ERROR] cannot perform inplace random generation on a UniTensor which is "
                      "not Dense or Block.%s",
                      "\n");
    }
  }

}  // namespace cytnx
#endif
