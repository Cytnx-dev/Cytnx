#include "UniTensor.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <typeinfo>

#include "utils/utils.hpp"
#include "linalg.hpp"
#include "random.hpp"

namespace cytnx {
  namespace {
    constexpr unsigned int kLegacyUniTensorMagic = 555;
    constexpr unsigned int kVersionedUniTensorMagic = 556;
    constexpr unsigned int kCurrentUniTensorFileVersion = 1;

    boost::intrusive_ptr<UniTensor_base> make_unitensor_impl_from_type(const int utentype) {
      if (utentype == UTenType.Dense) {
        return boost::intrusive_ptr<UniTensor_base>(new DenseUniTensor());
      }
      if (utentype == UTenType.Block) {
        return boost::intrusive_ptr<UniTensor_base>(new BlockUniTensor());
      }
      if (utentype == UTenType.BlockFermionic) {
        return boost::intrusive_ptr<UniTensor_base>(new BlockFermionicUniTensor());
      }

      cytnx_error_msg(utentype == UTenType.Void,
                      "[ERROR][UniTensor::_Load] The file contains an uninitialized UniTensor of "
                      "type Void. The file seems corrupt.%s",
                      "\n");
      cytnx_error_msg(utentype == UTenType.Sparse,
                      "[ERROR][UniTensor::_Load] The file contains a SparseUniTensor, which is "
                      "deprecated (saved with an old Cytnx version, or the file is corrupt). Use "
                      "BlockUniTensor or LinOp instead.%s",
                      "\n");
      cytnx_error_msg(true,
                      "[ERROR][UniTensor::_Load] Unknown UniTensor type id '%d' (corrupt or "
                      "incompatible file?)\n",
                      utentype);
      return boost::intrusive_ptr<UniTensor_base>();
    }
  }  // namespace

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
  UniTensor UniTensor::Add(const Scalar &rhs) const { return cytnx::linalg::Add(*this, rhs); }

  UniTensor UniTensor::Sub(const UniTensor &rhs) const { return cytnx::linalg::Sub(*this, rhs); }
  UniTensor UniTensor::Sub(const Scalar &rhs) const { return cytnx::linalg::Sub(*this, rhs); }

  UniTensor UniTensor::Div(const UniTensor &rhs) const { return cytnx::linalg::Div(*this, rhs); }
  UniTensor UniTensor::Div(const Scalar &rhs) const { return cytnx::linalg::Div(*this, rhs); }

  UniTensor UniTensor::Mul(const UniTensor &rhs) const { return cytnx::linalg::Mul(*this, rhs); }
  UniTensor UniTensor::Mul(const Scalar &rhs) const { return cytnx::linalg::Mul(*this, rhs); }

  void UniTensor::_Save(std::fstream &f) const {
    cytnx_error_msg(!f.is_open(), "[ERROR][UniTensor] invalid fstream!.%s", "\n");
    cytnx_error_msg(this->_impl->uten_type_id == UTenType.Void,
                    "[ERROR][UniTensor] Cannot save an uninitialized UniTensor.%s", "\n");
    cytnx_error_msg(this->_impl->uten_type() == UTenType.Sparse,
                    "[ERROR] SparseUniTensor is deprecated. Use BlockUniTensor or LinOp instead.%s",
                    "\n");

    unsigned int IDDs = kVersionedUniTensorMagic;
    f.write((char *)&IDDs, sizeof(unsigned int));
    unsigned int version = kCurrentUniTensorFileVersion;
    f.write((char *)&version, sizeof(unsigned int));

    // first, save common meta data:
    f.write((char *)&this->_impl->uten_type_id,
            sizeof(int));  // uten type, this is used to determine Block/Dense upon load
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
      std::size_t tmp = this->_impl->_labels[i].size();
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
    unsigned int version = 0;
    if (tmpIDDs == kVersionedUniTensorMagic) {
      f.read((char *)&version, sizeof(unsigned int));
      cytnx_error_msg(version != kCurrentUniTensorFileVersion,
                      "[ERROR][UniTensor::_Load] Unsupported UniTensor file format version '%u'.%s",
                      version, "\n");
    } else {
      cytnx_error_msg(tmpIDDs != kLegacyUniTensorMagic,
                      "[ERROR] the object is not a cytnx UniTensor!%s", "\n");
    }

    int utentype;
    f.read((char *)&utentype,
           sizeof(int));  // uten type, this is used to determine Block/Dense upon load
    this->_impl = make_unitensor_impl_from_type(utentype);

    f.read((char *)&this->_impl->_is_braket_form, sizeof(bool));
    f.read((char *)&this->_impl->_is_tag, sizeof(bool));
    f.read((char *)&this->_impl->_is_diag, sizeof(bool));
    f.read((char *)&this->_impl->_rowrank, sizeof(cytnx_int64));

    cytnx_uint32 len_name;
    f.read((char *)&len_name, sizeof(cytnx_uint32));
    if (len_name != 0) {
      this->_impl->_name.resize(len_name);
      f.read(this->_impl->_name.data(), sizeof(char) * len_name);
    }

    cytnx_uint64 rank;
    f.read((char *)&rank, sizeof(cytnx_uint64));
    this->_impl->_labels.resize(rank);
    this->_impl->_bonds.resize(rank);
    for (cytnx_uint64 i = 0; i < rank; i++) {
      std::size_t tmp;
      f.read((char *)&tmp, sizeof(std::size_t));
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
    this->_impl->_load_dispatch(f, version);
  }

  void UniTensor::Save(const std::string &fname) const {
    std::fstream f;
    if (std::filesystem::path(fname).has_extension()) {
      // filename extension is given
      f.open(fname, std::ios::out | std::ios::trunc | std::ios::binary);
    } else {
      // add filename extension
      cytnx_warning_msg(true,
                        "Missing file extension in fname '%s'. I am adding the extension '.cytnx'. "
                        "This is deprecated, please provide the file extension in the future.\n",
                        fname.c_str());
      f.open((fname + ".cytnx"), std::ios::out | std::ios::trunc | std::ios::binary);
    }
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for save.%s", "\n");
    }
    this->_Save(f);
    f.close();
  }
  void UniTensor::Save(const char *fname) const { Save(std::string(fname)); }

  UniTensor UniTensor::Load(const std::string &fname) {
    UniTensor out;
    std::fstream f;
    f.open(fname, std::ios::in | std::ios::binary);
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for load. >> %s\n", fname.c_str());
    }
    out._Load(f);
    f.close();
    return out;
  }
  UniTensor UniTensor::Load(const char *fname) { return UniTensor::Load(std::string(fname)); }

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
      auto &blocks = this->get_blocks_();
      for (size_t i = 0; i < blocks.size(); ++i) {
        cytnx::random::normal_(blocks[i], mean, std, seed + i);
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
      auto &blocks = this->get_blocks_();
      for (size_t i = 0; i < blocks.size(); ++i) {
        cytnx::random::uniform_(blocks[i], low, high, seed + i);
      }
    } else {
      cytnx_error_msg(true,
                      "[ERROR] Cannot perform inplace random generation on a UniTensor which is "
                      "not Dense or Block.%s",
                      "\n");
    }
    return *this;
  }

  cytnx_uint64 UniTensor::flattened_dimension() const {
    cytnx_error_msg(this->is_void(),
                    "[ERROR][UniTensor::flattened_dimension] the UniTensor is uninitialized.%s",
                    "\n");
    return cytnx::flattened_dimension(this->bonds(), this->is_diag());
  }

  cytnx_uint64 flattened_dimension(const std::vector<Bond> &bonds, bool is_diag) {
    if (bonds.empty()) return 1;
    // A diagonal UniTensor stores one block per sector of its first bond, with the size of
    // that sector's degeneracy (dense: one block of the first bond's dimension).
    if (is_diag) return bonds[0].dim();

    const std::vector<Symmetry> &syms = bonds[0].syms();
    for (const Bond &bond : bonds) {
      cytnx_error_msg(bond.syms() != syms,
                      "[ERROR][flattened_dimension] all bonds must carry the same symmetries.%s",
                      "\n");
    }
    if (syms.empty()) {
      std::uint64_t dim = 1;
      for (const Bond &bond : bonds) dim *= bond.dim();
      return dim;
    }

    // Fold in one bond at a time, keeping the number of basis states for each total quantum
    // number. A combination of sectors is allowed when its total quantum number is zero for
    // every symmetry, with the quantum numbers of BD_BRA bonds reversed, the same rule that
    // BlockUniTensor::Init uses to decide which blocks exist. The cost is linear in the rank,
    // where enumerating the combinations would be exponential.
    using Charge = std::vector<cytnx_int64>;
    std::map<Charge, std::uint64_t> counts;
    Charge sector(syms.size());
    Charge total(syms.size());
    for (std::size_t b = 0; b < bonds.size(); ++b) {
      const Bond &bond = bonds[b];
      const bool reversed = bond.type() == BD_BRA;
      const std::vector<std::vector<cytnx_int64>> &qnums = bond.qnums();
      const std::vector<cytnx_uint64> &degs = bond.getDegeneracies();
      std::map<Charge, std::uint64_t> next;
      for (std::size_t s = 0; s < qnums.size(); ++s) {
        for (std::size_t i = 0; i < syms.size(); ++i) {
          sector[i] = reversed ? syms[i].reverse_rule(qnums[s][i]) : qnums[s][i];
        }
        if (b == 0) {
          next[sector] += degs[s];
          continue;
        }
        for (const auto &[charge, count] : counts) {
          for (std::size_t i = 0; i < syms.size(); ++i) {
            total[i] = syms[i].combine_rule(charge[i], sector[i]);
          }
          next[total] += count * degs[s];
        }
      }
      counts = std::move(next);
    }
    const auto zero = counts.find(Charge(syms.size(), 0));
    return zero == counts.end() ? 0 : zero->second;
  }

}  // namespace cytnx
