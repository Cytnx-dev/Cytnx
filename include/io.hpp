#ifndef CYTNX_IO_H_
#define CYTNX_IO_H_

#include <string>
#include <filesystem>

#include "H5Cpp.h"

#include "backend/Storage.hpp"
#include "Bond.hpp"
#include "tn_algo/MPS.hpp"
#include "Symmetry.hpp"
#include "Tensor.hpp"
#include "Type.hpp"
#include "UniTensor.hpp"
#include "cytnx_error.hpp"

namespace cytnx {
  /**
  @namespace cytnx::io
  @brief IO functions for saving an loading to HDF5 file format.
  */
  namespace io {

    namespace internal {
      ///@cond
      /**
       * @brief Check if a dataset of correct type and size exists, otherwise create a new one
       * @param[in] datatype datatype to be written
       * @param[in] dimensions size of the tensor to be written
       * @param[in] container group to create dataset in
       * @param[in] name the name of the dataset
       * @param[in] overwrite if true, previous data will be overwritten or deleted.
       * @returns an opened dataset, compatible for writing
       */
      H5::DataSet create_dataset(H5::DataType &datatype, const std::vector<hsize_t> &dimensions,
                                 H5::Group &container, const std::string &name, bool overwrite);
      ///@endcond
    }  // namespace internal

    /**
     * @brief Create a group, given a path that can contain subpathes
     * @details Opens the group or creates it newly
     * @param[in] container root group
     * @param[in] path a path that can contain subpathes
     * @param[in] recursive if the path can contain subpathes, such that groups in group are opened
     * or created
     * @returns the opened group or a newly created group
     */
    H5::Group create_group(H5::Group &container, const std::string &path, bool recursive = true);

    /**
     * @brief Remove a group if dataset if it exists
     * @param[in] container group or file from which the attribute will be removed
     * @param[in] name the name of the dataset or group
     * @param[in] overwrite if true, the data will be removed. If false, an error is thrown if the
     * data exists
     * @returns true if the name existed, false otherwise
     */
    bool unlink(H5::Group &container, const std::string &name, bool overwrite = true);

    /**
     * @brief Save scalar to an attribute
     * @param[in] object a scalar of any type that is supported by cytnx
     * @param[in] container file, group or dataset; should be opened for writing
     * @param[in] name the name of the attribute
     * @param[in] overwrite if true, overwrite previous data in the container
     */
    void save_attribute(const Scalar_list &object, H5::H5Object &container, const std::string &name,
                        bool overwrite = false);
    /**
     * @brief Save string to an attribute
     * @see save_attribute(const Scalar_list &object, H5::Group &container, const std::string &name)
     */
    void save_attribute(const std::string &object, H5::H5Object &container, const std::string &name,
                        bool overwrite = false);
    /**
     * @brief Save vector of strings to an attribute
     * @see save_attribute(const Scalar_list &object, H5::Group &container, const std::string &name)
     */
    void save_attribute(const std::vector<std::string> &object, H5::H5Object &container,
                        const std::string &name, bool overwrite = false);

    /**
     * @brief Load scalar from an attribute
     * @param[out] object a scalar of any type that is supported by cytnx; needs to be compatible
     * with the data format in the file.
     * @param[in] container file, group or dataset
     * @param[in] name the name of the attribute
     */
    template <typename T>
    void load_attribute(T &object, H5::H5Object &container, const std::string &name) {
      H5::Attribute attr = container.openAttribute(name);
      H5::DataType datatype = attr.getDataType();
      H5::DataType scalartype = Type.get_hdf5_type(object);
      cytnx_error_msg(datatype.getClass() != scalartype.getClass(),
                      "[ERROR] Attribute '%s' data type class mismatch.\n", name.c_str());
      attr.read(scalartype, &object);
    }
    /**
     * @brief Load string from an attribute
     * @see load_attribute(T &object, H5::H5Object &container, const std::string &name)
     */
    void load_attribute(std::string &object, H5::H5Object &container, const std::string &name);

    /**
     * @brief Remove an attribute if it exists
     * @param[in] container file, group or dataset from which the attribute will be removed
     * @param[in] name the name of the attribute
     * @param[in] overwrite if true, the attribute will be removed. If false, an error is thrown if
     * the attribute exists
     * @returns true if the attribute existed, false otherwise
     */
    bool remove_attribute(H5::H5Object &container, const std::string &name, bool overwrite = true);

    /**
     * @brief Save data to a dataset
     * @param[in] object a vector of supported types
     * @param[in] container file, group or dataset; should be opened for writing
     * @param[in] name the name of the dataset
     * @param[in] overwrite if true, overwrite previous data in the container
     * @returns the dataset that was written to
     */
    H5::DataSet save_dataset(const Vector_list &object, H5::Group &container,
                             const std::string &name, bool overwrite = false);
    /**
     * @brief Save string to a dataset
     * @see save_dataset(const Vector_list &object, H5::Group &container, const std::string &name,
     * bool overwrite)
     */
    H5::DataSet save_dataset(const std::vector<std::string> &object, H5::Group &container,
                             const std::string &name, bool overwrite = false);
    /**
     * @brief Save a matrix of a supported type
     * @param[in] object a vector of a vector based on a supported type; if the outer vector has \em
     * rownum elements, and all inner vectors have \em colnum elements, then the result will be
     * written as a \em rownum \em x \em colnum matrix.
     * @see save_dataset(const Vector_list &object, H5::Group &container, const std::string &name,
     * bool overwrite)
     * @note All inner vectors need to have the same length, such that the data structure
     * corresponds to a rectangular matrix.
     */
    H5::DataSet save_dataset(const Matrix_list &object, H5::Group &container,
                             const std::string &name, bool overwrite = false);

    /**
     * @brief Load data from a dataset
     * @param[out] object vector of supported types
     * @param[in] container file, group or dataset
     * @param[in] name the name of the attribute
     */
    template <typename T>
    void load_dataset(std::vector<T> &object, H5::H5Object &container, const std::string &name) {
      H5::DataSet dataset = container.openDataSet(name);
      H5::DataSpace dataspace = dataset.getSpace();
      object.resize(dataspace.getSimpleExtentNpoints());
      H5::DataType datatype = dataset.getDataType();
      T scalar;
      H5::DataType scalartype = Type.get_hdf5_type(scalar);
      cytnx_error_msg(datatype.getClass() != scalartype.getClass(),
                      "[ERROR] Dataset '%s' type class mismatch.\n", name.c_str());
      dataset.read(object.data(), scalartype);
    }

    /**
     * @brief Load vector of strings from a dataset
     * @param[out] object vector of strings
     * @see void load_dataset(std::vector<T> &object, H5::H5Object &container, const std::string
     * &name)
     */
    void load_dataset(std::vector<std::string> &object, H5::H5Object &container,
                      const std::string &name);
    /**
     * @brief Load a matrix of a supported type
     * @param[out] object a vector of a vector based on a supported type
     * @see save_dataset(const Matrix_list &object, H5::Group &container, const std::string &name,
     * bool overwrite)
     * @see void load_dataset(std::vector<T> &object, H5::H5Object &container, const std::string
     * &name)
     */
    template <typename T>
    void load_dataset(std::vector<std::vector<T>> &object, H5::H5Object &container,
                      const std::string &name) {
      H5::DataSet dataset = container.openDataSet(name);
      H5::DataSpace dataspace = dataset.getSpace();
      cytnx_error_msg(
        dataspace.getSimpleExtentNdims() != 2,
        "[ERROR] Dataset '%s' should be a two-dimensional array. The HDF5 data seems corrupt!\n",
        name.c_str());
      hsize_t dims[2];
      dataspace.getSimpleExtentDims(dims);
      hsize_t rownum = dims[0];
      hsize_t colnum = dims[1];
      H5::DataType datatype = dataset.getDataType();
      T scalar;
      H5::DataType scalartype = Type.get_hdf5_type(scalar);
      cytnx_error_msg(datatype.getClass() != scalartype.getClass(),
                      "[ERROR] Dataset '%s' type class mismatch.\n", name.c_str());
      // Read HDF5 data into a flattened temporary vector
      std::vector<T> flat(rownum * colnum);
      dataset.read(flat.data(), scalartype);
      // Reconstruct the vector of vectors
      object.assign(rownum, std::vector<T>(colnum));
      for (hsize_t i = 0; i < rownum; ++i) {
        std::copy(flat.begin() + i * colnum, flat.begin() + (i + 1) * colnum, object[i].begin());
      }
    }

    /**
     * @brief Input/output file access mode for HDF5 files.
     */
    enum IoMode : int {
      ACC_TRUNC, /**< Open file for reading and writing; overwrites if file exists; creates new file
                    otherwise */
      ACC_NOREPLACE, /**< Open file for reading and writing; fails if file exists; creates new file
                        otherwise */
      ACC_IN, /**< Open file for reading only; fails if file does not exist */
      ACC_INOUT /**< Open file for reading and writing; opens an existing file; creates new file
                   otherwise */
    };

    /**
     * @brief Open HDF5 file
     * @details Use file.close() to close the file after use.
     * @param[in] fname file name
     * @param[in] mode the write mode:\n
     *  ACC_TRUNC     Open file for reading and writing; overwrites if file exists; creates new file
     * otherwise\n ACC_NOREPLACE Open file for reading and writing; fails if file exists; creates
     * new file otherwise\n ACC_IN        Open file for reading only; fails if file does not exist\n
     *  ACC_INOUT     Open file for reading and writing; opens an existing file; creates new file
     * otherwise
     * @returns the file handle
     * @note The file ending should be one of ".h5", ".hdf5", ".H5", ".HDF5", ".hdf".
     */
    H5::H5File open(const std::filesystem::path &fname, IoMode mode = ACC_TRUNC);

    /**
     * @brief Classes that can be saved and loaded to/from HDF5
     */
    using savable_class = std::variant<Bond, tn_algo::MPS, Storage, Symmetry, Tensor, UniTensor>;

    /**
     * @brief Save object to HDF5
     * @details Can be used with most cytnx classes as objects.
     * @param[in] object the cytnx object to be saved
     * @param[in] container HDF5 object that should be opened for writing; can be a file or a group
     * @param[in] path path inside the file; a path '/foo/bar/Obj' will write the object to the
     * dataset 'Obj' the group '/foo/bar' in the file.
     * @param[in] name the name of the object to save to
     * @param[in] overwrite if true, overwrite previous data in the container
     * @note The file ending should be one of ".h5", ".hdf5", ".H5", ".HDF5", ".hdf".
     * @see Load(), open()
     */
    void Save(const savable_class &object, H5::Group &container, const std::string &name,
              const std::string &path = "", bool overwrite = false);

    /**
     * @brief Open HDF5 file, save object, and close file.
     * @param[in] object the cytnx object to be saved
     * @param[in] fname file name
     * @param[in] path path inside the file; a path '/foo/bar/Obj' will write the object to the
     * dataset 'Obj' the group '/foo/bar' in the file.
     * @param[in] name the name of the attribute, dataset, or group
     * @param[in] mode the write mode:\n
     *  `w` write; creates a new file. If the given file exists, its contents are destroyed.\n
     *  `x` exclusive; creates a new file. Fails if the given file exists already.\n
     *  `a` append. opens for writing without overwriting any existing content. Creates the file if
     * it doesn't exist.\n `u` update; opens for writing. Existing content will be
     * updated(overwritten). Creates the file if it doesn't exist.
     * @see Save(const savable_class &object, H5::Group &file, const std::string &name, const
     * std::string &path, bool overwrite), open()
     */

    /// @cond
    // void Save(const savable_class object, const std::filesystem::path &fname, const std::string
    // &name, const std::string &path = "", const char mode = 'w')  {
    //     cytnx_error_msg(mode == 'r', "Mode 'r' is read-only and not available for writing
    //     files.%s", "\n"); bool overwrite = (mode == 'u'); if (mode == 'a')
    //         mode = 'u';
    //     hid_t file = open(fname, mode);
    //     Save(object, file, name, path, overwrite);
    //     Close(file);
    // }
    /// @endcond

    /**
     * @brief Load object from HDF5 file
     * @details Can be used with most cytnx classes as objects.
     * @param[out] object an object of the correct type that will be modified (inline)
     * @param[in] container HDF5 object that should be opened for writing; can be a file or a group
     * @param[in] name the name of the attribute, dataset, or group
     * @param[in] path path inside the file; a path /foo/bar/Obj will read the object from the
     * dataset 'Obj' the group '/foo/bar' in the file.
     * @param[in] name the name of the attribute, dataset, or group
     * @pre The file must be an object which was saved by Save().
     * @see Save(), open()
     */
    void Load(savable_class &object, H5::Group &container, const std::string &name,
              const std::string &path = "");

    /**
     * @brief savable_class objects that can be loaded to different devices
     */
    using loadable_to_device = std::variant<tn_algo::MPS, Storage, Tensor, UniTensor>;

    /**
     * @brief Load object from HDF5 file and restore the device the data was saved to
     * @param[in] restore_device whether to try restoring the device on which the data is stored; if
     * false, the data will be kept on the CPU. Only effects objects that contain Tensor or Storage.
     * @see Load(savable_class &object, H5::Group &container, const std::string &name, const
     * std::string &path)
     */
    void Load(loadable_to_device &object, H5::Group &container, const std::string &name,
              const std::string &path, bool restore_device);

    /// @cond
    /**
     * @brief Open HDF5 file, load object, and close file.
     * @param[in] fname file name
     * @see Load(const savable_class object, const std::string &name, const std::filesystem::path
     * &fname, const std::string &path, bool restore_device), open()
     */
    // void Load(const savable_class object, const std::filesystem::path &fname, const std::string
    // &name, const std::string &path = "", bool restore_device = true) {
    //     hid_t file = open(fname, 'r');
    //     Load(object, file, name, path, restore_device);
    //     Close(file);
    // }
    /// @endcond

  }  // namespace io
}  // namespace cytnx

#endif  // CYTNX_IO_H_
