#include "tn_algo/MPS.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

#include "H5Cpp.h"

using namespace std;

#ifdef BACKEND_TORCH
#else

namespace cytnx {
  namespace tn_algo {

    std::ostream &operator<<(std::ostream &os, const MPS &in) {
      in._impl->Print(os);
      return os;
    }

    void MPS::Save(const std::filesystem::path &fname, const std::string &path,
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
        fnameext += ".cymps";
        cytnx_warning_msg(
          true,
          "Missing file extension in fname '%s'. I am adding the extension '.cymps'. This is "
          "deprecated, please provide the file extension in the future.\n",
          fname.string().c_str());
        if (mode == 'x') {
          cytnx_error_msg(std::filesystem::exists(fnameext),
                          "[ERROR] File %s already exists. Use mode 'w' to overwrite.",
                          fnameext.string().c_str());
        } else {
          cytnx_error_msg(mode != 'w', "[ERROR] Unknown mode '%c' for writing to binary file.",
                          mode);
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
    void MPS::Save(const char *fname, const std::string &path, const char mode) const {
      this->Save(std::filesystem::path(fname), path, mode);
    }

    MPS MPS::Load(const std::filesystem::path &fname, const std::string &path,
                  const bool restore_device) {
      MPS out;
      out.Load_(fname, path, restore_device);
      return out;
    }
    MPS MPS::Load(const char *fname, const std::string &path, const bool restore_device) {
      return MPS::Load(std::filesystem::path(fname), path, restore_device);
    }

    void MPS::Load_(const std::filesystem::path &fname, const std::string &path,
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
                          path.c_str(), fname.string().c_str());
        }
        // read data
        this->from_hdf5(location, restore_device);
        h5file.close();
      } else {  // load binary
        fstream f;
        f.open(fname, std::ios::in | std::ios::binary);
        if (!f.is_open()) {
          cytnx_error_msg(true, "[ERROR] Cannot open file '%s'.\n", fname.string().c_str());
        }
        this->from_binary(f, restore_device);
        f.close();
      }
    }
    void MPS::Load_(const char *fname, const std::string &path, const bool restore_device) {
      this->Load_(std::filesystem::path(fname), path, restore_device);
    }

    void MPS::to_hdf5(H5::Group &location, const bool overwrite) const {
      cytnx_error_msg(true, "[ERROR] Saving MPS to HDF5 is not implemented yet!%s", "\n");
    }
    void MPS::from_hdf5(H5::Group &location, const bool restore_device) {
      cytnx_error_msg(true, "[ERROR] Loading MPS from HDF5 is not implemented yet!%s", "\n");
    }

    void MPS::to_binary(std::ostream &f) const {
      cytnx_error_msg(this->_impl->mps_type_id == MPSType.Void,
                      "[ERROR][MPS] Cannot save an uninitialize MPS.%s", "\n");

      unsigned int IDDs = 109;
      f.write((char *)&IDDs, sizeof(unsigned int));

      f.write((char *)&this->_impl->mps_type_id,
              sizeof(int));  // mps type, this is used to determine Regular/iMPS upon load

      // first, save common meta data:
      f.write((char *)&this->_impl->virt_dim, sizeof(this->_impl->virt_dim));
      f.write((char *)&this->_impl->S_loc, sizeof(this->_impl->S_loc));

      // second, dispatch to do remaining saving:
      this->_impl->to_binary_dispatch(f);
    }

    void MPS::from_binary(std::istream &f, const bool restore_device) {
      unsigned int tmpIDDs;
      f.read((char *)&tmpIDDs, sizeof(unsigned int));
      cytnx_error_msg(tmpIDDs != 109, "[ERROR] the object is not a cytnx MPS!%s", "\n");

      int mpstype;
      f.read((char *)&mpstype,
             sizeof(int));  // mps type, this is used to determine Sparse/Dense upon load

      if (mpstype == MPSType.RegularMPS) {
        this->_impl = boost::intrusive_ptr<MPS_impl>(new RegularMPS());
      } else if (mpstype == MPSType.iMPS) {
        this->_impl = boost::intrusive_ptr<MPS_impl>(new iMPS());
      } else {
        cytnx_error_msg(true, "[ERROR] Unknown MPS type!%s", "\n");
      }

      // first, load common meta data:
      f.read((char *)&this->_impl->virt_dim, sizeof(this->_impl->virt_dim));
      f.read((char *)&this->_impl->S_loc, sizeof(this->_impl->S_loc));

      // second, let dispatch to do remaining loading.
      this->_impl->from_binary_dispatch(f, restore_device);
    }
  }  // namespace tn_algo

}  // namespace cytnx

#endif
