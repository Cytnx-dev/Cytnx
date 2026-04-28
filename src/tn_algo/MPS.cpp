#include "tn_algo/MPS.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace std;

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  namespace tn_algo {

    std::ostream& operator<<(std::ostream& os, const MPS& in) {
      in._impl->Print(os);
      return os;
    }

    void MPS::to_binary(std::ostream& f) const {
      cytnx_error_msg(this->_impl->mps_type_id == MPSType.Void,
                      "[ERROR][MPS] cannot save an uninitialize MPS.%s", "\n");

      unsigned int IDDs = 109;
      f.write((char*)&IDDs, sizeof(unsigned int));

      f.write((char*)&this->_impl->mps_type_id,
              sizeof(int));  // mps type, this is used to determine Regular/iMPS upon load

      // first, save common meta data:
      f.write((char*)&this->_impl->virt_dim, sizeof(this->_impl->virt_dim));
      f.write((char*)&this->_impl->S_loc, sizeof(this->_impl->S_loc));

      // second, dispatch to do remaining saving:
      this->_impl->to_binary_dispatch(f);
    }

    void MPS::from_binary(std::istream& f, const bool restore_device) {
      unsigned int tmpIDDs;
      f.read((char*)&tmpIDDs, sizeof(unsigned int));
      cytnx_error_msg(tmpIDDs != 109, "[ERROR] the object is not a cytnx MPS!%s", "\n");

      int mpstype;
      f.read((char*)&mpstype,
             sizeof(int));  // mps type, this is used to determine Sparse/Dense upon load

      if (mpstype == MPSType.RegularMPS) {
        this->_impl = boost::intrusive_ptr<MPS_impl>(new RegularMPS());
      } else if (mpstype == MPSType.iMPS) {
        this->_impl = boost::intrusive_ptr<MPS_impl>(new iMPS());
      } else {
        cytnx_error_msg(true, "[ERROR] Unknown MPS type!%s", "\n");
      }

      // first, load common meta data:
      f.read((char*)&this->_impl->virt_dim, sizeof(this->_impl->virt_dim));
      f.read((char*)&this->_impl->S_loc, sizeof(this->_impl->S_loc));

      // second, let dispatch to do remaining loading.
      this->_impl->from_binary_dispatch(f, restore_device);
    }

    void MPS::Save(const std::string& fname) const {
      fstream f;
      if (std::filesystem::path(fname).has_extension()) {
        // filename extension is given
        f.open(fname, ios::out | ios::trunc | ios::binary);
      } else {
        // add filename extension
        cytnx_warning_msg(
          true,
          "Missing file extension in fname '%s'. I am adding the extension '.cymps'. This is "
          "deprecated, please provide the file extension in the future.\n",
          fname.c_str());
        f.open((fname + ".cymps"), ios::out | ios::trunc | ios::binary);
      }
      if (!f.is_open()) {
        cytnx_error_msg(true, "[ERROR] invalid file path for save.%s", "\n");
      }
      this->to_binary(f);
      f.close();
    }
    void MPS::Save(const char* fname) const {
      fstream f;
      string ffname = string(fname) + ".cymps";
      f.open((ffname), ios::out | ios::trunc | ios::binary);
      if (!f.is_open()) {
        cytnx_error_msg(true, "[ERROR] invalid file path for save.%s", "\n");
      }
      this->to_binary(f);
      f.close();
    }

    MPS MPS::Load(const std::string& fname, const bool restore_device) {
      MPS out;
      out.Load_(fname, restore_device);
      return out;
    }
    MPS MPS::Load(const char* fname, const bool restore_device) {
      return MPS::Load(string(fname), restore_device);
    }

    void MPS::Load_(const std::string& fname, const bool restore_device) {
      fstream f;
      f.open(fname, ios::in | ios::binary);
      if (!f.is_open()) {
        cytnx_error_msg(true, "[ERROR] Cannot open file '%s'.\n", fname.c_str());
      }
      this->from_binary(f, restore_device);
      f.close();
    }
    void MPS::Load_(const char* fname, const bool restore_device) {
      this->Load(string(fname), restore_device);
    }

  }  // namespace tn_algo

}  // namespace cytnx

#endif
