#include "tn_algo/MPS.hpp"
#include <fstream>
#include <iostream>
using namespace std;
namespace cytnx {
  namespace tn_algo {

    std::ostream& operator<<(std::ostream& os, const MPS& in) {
      in._impl->Print(os);
      return os;
    }

    void MPS::_Save(std::fstream& f) const {
      cytnx_error_msg(!f.is_open(), "[ERROR][MPS] invalid fstream!.%s", "\n");
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
      this->_impl->_save_dispatch(f);
    }

    void MPS::_Load(std::fstream& f) {
      cytnx_error_msg(!f.is_open(), "[ERROR][MPS] invalid fstream%s", "\n");
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
      this->_impl->_load_dispatch(f);
    }

    void MPS::Save(const std::string& fname) const {
      fstream f;
      f.open((fname + ".cymps"), ios::out | ios::trunc | ios::binary);
      if (!f.is_open()) {
        cytnx_error_msg(true, "[ERROR] invalid file path for save.%s", "\n");
      }
      this->_Save(f);
      f.close();
    }
    void MPS::Save(const char* fname) const {
      fstream f;
      string ffname = string(fname) + ".cymps";
      f.open((ffname), ios::out | ios::trunc | ios::binary);
      if (!f.is_open()) {
        cytnx_error_msg(true, "[ERROR] invalid file path for save.%s", "\n");
      }
      this->_Save(f);
      f.close();
    }

    MPS MPS::Load(const std::string& fname) {
      MPS out;
      fstream f;
      f.open(fname, ios::in | ios::binary);
      if (!f.is_open()) {
        cytnx_error_msg(true, "[ERROR] invalid file path for load.%s", "\n");
      }
      out._Load(f);
      f.close();
      return out;
    }
    MPS MPS::Load(const char* fname) {
      MPS out;
      fstream f;
      f.open(fname, ios::in | ios::binary);
      if (!f.is_open()) {
        cytnx_error_msg(true, "[ERROR] invalid file path for load.%s", "\n");
      }
      out._Load(f);
      f.close();
      return out;
    }

  }  // namespace tn_algo

}  // namespace cytnx
