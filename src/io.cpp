#include "io.hpp"

#include <vector>
#include <string>
#include <filesystem>

#include "H5Cpp.h"

namespace cytnx {
  namespace io {

    H5::Group create_group(H5::Group &file, const std::string &path) {
      std::filesystem::path grouppath(path);
      std::filesystem::path subpath;
      std::string groupfolder = "/";
      for (const auto &part : grouppath) {
        if (part.empty()) continue;
        subpath /= part;
        groupfolder = subpath.generic_string(); // use generic_string() to avoid incompatibilites between file systems
        if (!file.nameExists(groupfolder)) file.createGroup(groupfolder);
      }
      H5::Group location = file.openGroup(groupfolder);
      return location;
    }

    H5::H5File open(const std::filesystem::path &fname, IoMode mode) {
      H5::H5File h5file;

      // Enable reuse of space after data is deleted;
      // Set the strategy: FSM_AGGR is standard for free-space management
      // Parameters: strategy, persist (true), threshold (default 1: track all free-space
      // sections)
      H5::FileCreatPropList fcpl;
      fcpl.setFileSpaceStrategy(H5F_FSPACE_STRATEGY_FSM_AGGR, true, 1);

      // Persistent free space requires HDF5 1.10.x format or later
      H5::FileAccPropList fapl;
      fapl.setLibverBounds(H5F_LIBVER_V110, H5F_LIBVER_LATEST);

      // open file
      switch (mode) {
        case ACC_TRUNC:
          h5file = H5::H5File(fname, H5F_ACC_TRUNC, fcpl, fapl);
          break;
        case ACC_NOREPLACE:
          h5file = H5::H5File(fname, H5F_ACC_EXCL, fcpl, fapl);
          break;
        case ACC_IN:
          h5file = H5::H5File(fname, 	H5F_ACC_RDONLY, fcpl, fapl);
          break;
        case ACC_INOUT:
          if (std::filesystem::exists(fname)) {
            h5file = H5::H5File(fname, H5F_ACC_RDWR, H5::FileCreatPropList::DEFAULT, fapl);
          } else {
            h5file = H5::H5File(fname, H5F_ACC_EXCL, fcpl, fapl);
          }
          break;
        default:
          cytnx_error_msg(true, "[ERROR] Unknown mode '%d' for writing to HDF5 file.", mode);
      }

      return h5file;
    }

    // void close(H5::H5File &file) {
    //   file.close();
    // }

    void Save(const savable_class &object, H5::Group &file, const std::string &name, const std::string &path, bool overwrite) {
        std::visit([&](const auto &concreteObj) {
            H5::Group location = create_group(file, path);
            concreteObj.to_hdf5(location, name, overwrite);
        }, object);
    }

    void Load(savable_class &object, H5::Group &file, const std::string &name, const std::string &path, bool restore_device) {
      std::visit([&](auto &concreteObj) {
        // open path
        H5::Group location;
        if (path.empty())
          location = file;
        else {
          try {
            location = file.openGroup(path);
          } catch (const H5::Exception &e) {
            std::cerr << e.getDetailMsg() << std::endl;
            cytnx_error_msg(true, "[ERROR] HDF5 path '%s' not found or is not a group.\n", path.c_str());
          }
        }
        if constexpr (requires { concreteObj.from_hdf5(location, name, restore_device); }) {
        // Class supports restore_device
          concreteObj.from_hdf5(location, name, restore_device);
        } else {
          concreteObj.from_hdf5(location, name);
        }
      }, object);
    }

  }  // namespace io
}  // namespace cytnx
