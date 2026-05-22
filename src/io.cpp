#include "io.hpp"

#include <vector>
#include <string>
#include <filesystem>

#include "H5Cpp.h"

namespace cytnx {
  namespace io {

    namespace internal {
      H5::DataSet create_dataset(H5::DataType &datatype, const std::vector<hsize_t> &dimensions,
                                 H5::Group &container, const std::string &name, bool overwrite) {
        H5::DataSet dataset;
        // check if old dataset can be overwritten
        bool create_new = true;
        if (container.nameExists(name)) {  // we can overwrite the data if type and size match
          cytnx_error_msg(
            !overwrite,
            "Dataset (or group) '%s' already exists. Use argument overwrite = true to overwrite.\n",
            name);
          if (container.childObjType(name) == H5O_TYPE_DATASET) {
            dataset = container.openDataSet(name);
            H5::DataSpace olddataspace = dataset.getSpace();
            auto rank = olddataspace.getSimpleExtentNdims();
            if (olddataspace.getSimpleExtentType() == H5S_SIMPLE &&
                dataset.getDataType() == datatype && rank == dimensions.size()) {
              create_new = false;
              std::vector<hsize_t> olddims(rank);
              olddataspace.getSimpleExtentDims(olddims.data());
              for (int i = 0; i < rank; i++) {
                if (olddims[i] != dimensions[i]) {
                  create_new = true;
                  break;
                }
              }
            }
            if (!create_new) return dataset;
            // otherwise, create a new dataset one now
            dataset.close();
          }
          container.unlink(name);
        }
        // create a new dataset
        H5::DataSpace dataspace(dimensions.size(), dimensions.data());
        dataset = container.createDataSet(name, datatype, dataspace);
        return dataset;
      }
    }  // namespace internal

    H5::Group create_group(H5::Group &container, const std::string &path, bool recursive) {
      if (recursive) {
        std::filesystem::path grouppath(path);
        std::filesystem::path subpath;
        std::string groupfolder = "/";
        for (const auto &part : grouppath) {
          if (part.empty()) continue;
          subpath /= part;
          groupfolder = subpath.generic_string();  // use generic_string() to avoid
                                                   // incompatibilities between file systems
          if (!container.nameExists(groupfolder)) container.createGroup(groupfolder);
        }
        H5::Group group = container.openGroup(groupfolder);
        return group;
      } else {  // simple version without recursion
        H5::Group group;
        if (container.nameExists(path)) {
          group = container.openGroup(path);
        } else {
          group = container.createGroup(path);
        }
        return group;
      }
    }

    void save_attribute(const Scalar_list &object, H5::H5Object &container, const std::string &name,
                        bool overwrite) {
      std::visit(
        [&](const auto &scalar) {
          H5::DataType datatype = Type.get_hdf5_type(scalar);
          if (container.attrExists(name)) {
            cytnx_error_msg(
              !overwrite,
              "Attribute '%s' already exists. Use argument overwrite = true to overwrite.\n", name);
            H5::Attribute oldattr = container.openAttribute(name);
            if (oldattr.getSpace().getSimpleExtentType() == H5S_SCALAR &&
                oldattr.getDataType() == datatype) {
              oldattr.write(datatype, &scalar);
              return;
            }  // else: remove and create again
            container.removeAttr(name);
          }
          H5::Attribute attr = container.createAttribute(name, datatype, H5::DataSpace(H5S_SCALAR));
          attr.write(datatype, &scalar);
        },
        object);
    }

    void save_attribute(const std::string &object, H5::H5Object &container, const std::string &name,
                        bool overwrite) {
      H5::StrType str_type =
        H5::StrType(H5::PredType::C_S1, object.length() + 1);  // include NULL terminator
      if (container.attrExists(name)) {
        cytnx_error_msg(
          !overwrite,
          "Attribute '%s' already exists. Use argument overwrite = true to overwrite.\n", name);
        H5::Attribute oldattr = container.openAttribute(name);
        if (oldattr.getSpace().getSimpleExtentType() == H5S_SCALAR &&
            oldattr.getStrType() == str_type) {
          oldattr.write(str_type, object);
          return;
        }  // else: remove and create again
        container.removeAttr(name);
      }
      H5::Attribute attr = container.createAttribute(name, str_type, H5::DataSpace(H5S_SCALAR));
      attr.write(str_type, object);
    }

    void save_attribute(const std::vector<std::string> &object, H5::H5Object &container,
                        const std::string &name, bool overwrite) {
      hsize_t vecdims[1] = {object.size()};
      H5::DataSpace dataspace = H5::DataSpace(1, vecdims);
      H5::StrType str_type = H5::StrType(H5::PredType::C_S1, H5T_VARIABLE);
      if (container.attrExists(name)) {  // simply delete old dataset
        cytnx_error_msg(
          !overwrite,
          "Attribute '%s' already exists. Use argument overwrite = true to overwrite.\n", name);
        container.removeAttr(name);
      }
      // create a new dataset
      H5::Attribute attr = container.createAttribute(name, str_type, dataspace);
      std::vector<const char *> c_strings;  // H5 needs cstrings
      c_strings.reserve(object.size());
      for (const auto &elem : object) {
        c_strings.push_back(elem.c_str());
      }
      attr.write(str_type, c_strings.data());
    }

    H5::DataSet save_dataset(const Vector_list &object, H5::Group &container,
                             const std::string &name, bool overwrite) {
      H5::DataSet dataset;
      std::visit(
        [&](const auto &vec) {
          cytnx_error_msg(vec.empty(), "Cannot write an empty vector to a dataset!\s", "\n");

          H5::DataType datatype = Type.get_hdf5_type(vec[0]);
          dataset = internal::create_dataset(datatype, {vec.size()}, container, name, overwrite);

          using Value_type = typename std::decay_t<decltype(vec)>::value_type;
          if constexpr (std::is_same_v<Value_type, bool>) {
            // H5cpp writes bool as unsigned char
            std::vector<unsigned char> flat_bools(vec.begin(), vec.end());
            dataset.write(flat_bools.data(), datatype);
          } else {
            dataset.write(vec.data(), datatype);
          }
        },
        object);
      return dataset;
    }

    H5::DataSet save_dataset(const std::vector<std::string> &object, H5::Group &container,
                             const std::string &name, bool overwrite) {
      hsize_t vecdims[1] = {object.size()};
      H5::DataSpace dataspace = H5::DataSpace(1, vecdims);
      H5::StrType str_type = H5::StrType(H5::PredType::C_S1, H5T_VARIABLE);
      if (container.nameExists(name)) {  // simply delete old dataset
        cytnx_error_msg(
          !overwrite,
          "Dataset (or group) '%s' already exists. Use argument overwrite = true to overwrite.\n",
          name);
        container.unlink(name);
      }
      // create a new dataset
      H5::DataSet dataset = container.createDataSet(name, str_type, dataspace);
      std::vector<const char *> c_strings;  // H5 needs cstrings
      c_strings.reserve(object.size());
      for (const auto &elem : object) {
        c_strings.push_back(elem.c_str());
      }
      dataset.write(c_strings.data(), str_type);
      return dataset;
    }

    bool remove_attribute(H5::H5Object &container, const std::string &name, bool overwrite) {
      if (container.attrExists(name)) {
        cytnx_error_msg(
          !overwrite, "Attribute '%s' exists. Use argument overwrite = true to remove it.\n", name);
        container.removeAttr(name);
        return true;
      }
      return false;
    }

    H5::DataSet save_dataset(const Matrix_list &object, H5::Group &container,
                             const std::string &name, bool overwrite) {
      H5::DataSet dataset;
      std::visit(
        [&](const auto &vec) {
          cytnx_error_msg(vec.empty() || vec[0].empty(),
                          "Cannot write an empty vector to a dataset!\s", "\n");
          hsize_t rownum = vec.size();
          hsize_t colnum = vec[0].size();
          for (hsize_t i = 1; i < vec.size(); ++i) {
            cytnx_error_msg(vec[i].size() != colnum,
                            "[ERROR] object[%d] has different size than object[0]. A rectanglular "
                            "matrix is expected as input!\n",
                            i);
          }

          H5::DataType datatype = Type.get_hdf5_type(vec[0][0]);
          dataset =
            internal::create_dataset(datatype, {rownum, colnum}, container, name, overwrite);

          using Inner_vec_type = typename std::decay_t<decltype(vec)>::value_type;
          using Scalar_type = typename Inner_vec_type::value_type;
          if constexpr (std::is_same_v<Scalar_type, bool>) {
            std::vector<unsigned char> flat(rownum * colnum);  // flatten vector<vector>
            for (hsize_t i = 0; i < vec.size(); ++i) {
              std::copy(vec[i].begin(), vec[i].end(), flat.begin() + i * colnum);
            }
            dataset.write(flat.data(), datatype);
          } else {
            std::vector<Scalar_type> flat(rownum * colnum);  // flatten vector<vector>
            for (hsize_t i = 0; i < vec.size(); ++i) {
              std::copy(vec[i].begin(), vec[i].end(), flat.begin() + i * colnum);
            }
            dataset.write(flat.data(), datatype);
          }
        },
        object);
      return dataset;
    }

    bool unlink(H5::Group &container, const std::string &name, bool overwrite) {
      if (container.nameExists(name)) {
        cytnx_error_msg(
          !overwrite, "Dataset or group '%s' exists. Use argument overwrite = true to remove it.\n",
          name);
        container.unlink(name);
        return true;
      }
      return false;
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
          h5file = H5::H5File(fname, H5F_ACC_RDONLY, fcpl, fapl);
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

    // void close(H5::H5File &container) {
    //   container.close();
    // }

    void Save(const savable_class &object, H5::Group &container, const std::string &name,
              const std::string &path, bool overwrite) {
      std::visit(
        [&](const auto &concreteObj) {
          H5::Group group = create_group(container, path);
          concreteObj.to_hdf5(group, name, overwrite);
        },
        object);
    }

    void Load(savable_class &object, H5::Group &container, const std::string &name,
              const std::string &path) {
      std::visit(
        [&](auto &concreteObj) {
          H5::Group group = (path.empty() ? container : container.openGroup(path));
          concreteObj.from_hdf5(group, name);
        },
        object);
    }

    void Load(loadable_to_device &object, H5::Group &container, const std::string &name,
              const std::string &path, bool restore_device) {
      std::visit(
        [&](auto &concreteObj) {
          H5::Group group = (path.empty() ? container : container.openGroup(path));
          concreteObj.from_hdf5(group, name, restore_device);
        },
        object);
    }

  }  // namespace io
}  // namespace cytnx
