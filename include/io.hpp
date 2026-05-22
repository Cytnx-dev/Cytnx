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

    /**
     * @brief Create a group, given a path that can contain subpathes
     * @details Opens the group or creates it newly
     * @param[in] container root group
     * @param[in] path a path that can contain subpathes
     * @returns the opened group or a newly created group
     */
    H5::Group create_group(H5::Group &container, const std::string &path);

    /**
     * @brief Save data to an attribute
     * @param[in] object a scalar of any type that is supported by cytnx
     * @param[in] container group; should be opened for writing
     * @param[in] name the name of the attribute
     * @param[in] overwrite if true, overwrite previous data in the container
     */
    void save_attribute(const Scalar_list &object, H5::Group &container, const std::string &name,
                        bool overwrite = false);
    /**
     * @brief Save string to an attribute
     * @see save_attribute(const Scalar_list &object, H5::Group &container, const std::string &name)
     */
    void save_attribute(const std::string &object, H5::Group &container, const std::string &name,
                        bool overwrite = false);

    /**
     * @brief Save data to a dataset
     * @param[in] object a vector of supported types
     * @param[in] container group; should be opened for writing
     * @param[in] name the name of the dataset
     */
    void save_dataset(const std::vector<std::string> &object, H5::Group &container,
                      const std::string &name);

    /**
     * @brief Remove an attribute if it exists
     * @param[in] container group or file from which the attribute will be removed
     * @param[in] name the name of the attribute
     * @param[in] overwrite if true, the attribute will be removed. If false, an error is thrown if
     * the attribute exists
     */
    void remove_attribute(H5::Group &container, const std::string &name, bool overwrite = true);

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
     * @param[in] object an object of the correct type that will be modified (inline)
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
