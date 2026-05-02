#include "backend/Storage.hpp"

#include <filesystem>
#include <iostream>

#include "H5Cpp.h"

using namespace std;

namespace cytnx {

  Storage_init_interface __SII;

  std::ostream &operator<<(std::ostream &os, const Storage &in) {
    in.print();
    return os;
  }

  bool Storage::operator==(const Storage &rhs) {
    cytnx_error_msg(this->dtype() != rhs.dtype(),
                    "[ERROR] Cannot compare two Storage with different type.%s", "\n");
    if (this->size() != rhs.size()) return false;

    switch (this->dtype()) {
      case Type.ComplexDouble:
        for (cytnx_uint64 i = 0; i < this->size(); i++) {
          if (this->at<cytnx_complex128>(i) != rhs.at<cytnx_complex128>(i)) return false;
        }
        break;
      case Type.ComplexFloat:
        for (cytnx_uint64 i = 0; i < this->size(); i++) {
          if (this->at<cytnx_complex64>(i) != rhs.at<cytnx_complex64>(i)) return false;
        }
        break;
      case Type.Double:
        for (cytnx_uint64 i = 0; i < this->size(); i++) {
          if (this->at<cytnx_double>(i) != rhs.at<cytnx_double>(i)) return false;
        }
        break;
      case Type.Float:
        for (cytnx_uint64 i = 0; i < this->size(); i++) {
          if (this->at<cytnx_float>(i) != rhs.at<cytnx_float>(i)) return false;
        }
        break;
      case Type.Int64:
        for (cytnx_uint64 i = 0; i < this->size(); i++) {
          if (this->at<cytnx_int64>(i) != rhs.at<cytnx_int64>(i)) return false;
        }
        break;
      case Type.Uint64:
        for (cytnx_uint64 i = 0; i < this->size(); i++) {
          if (this->at<cytnx_uint64>(i) != rhs.at<cytnx_uint64>(i)) return false;
        }
        break;
      case Type.Int32:
        for (cytnx_uint64 i = 0; i < this->size(); i++) {
          if (this->at<cytnx_int32>(i) != rhs.at<cytnx_int32>(i)) return false;
        }
        break;
      case Type.Uint32:
        for (cytnx_uint64 i = 0; i < this->size(); i++) {
          if (this->at<cytnx_uint32>(i) != rhs.at<cytnx_uint32>(i)) return false;
        }
        break;
      case Type.Int16:
        for (cytnx_uint64 i = 0; i < this->size(); i++) {
          if (this->at<cytnx_int16>(i) != rhs.at<cytnx_int16>(i)) return false;
        }
        break;
      case Type.Uint16:
        for (cytnx_uint64 i = 0; i < this->size(); i++) {
          if (this->at<cytnx_uint16>(i) != rhs.at<cytnx_uint16>(i)) return false;
        }
        break;
      case Type.Bool:
        for (cytnx_uint64 i = 0; i < this->size(); i++) {
          if (this->at<cytnx_bool>(i) != rhs.at<cytnx_bool>(i)) return false;
        }
        break;
      default:
        cytnx_error_msg(true, "[ERROR] fatal internal, Storage has invalid type.%s", "\n");
    }
    return true;
  }
  bool Storage::operator!=(const Storage &rhs) { return !(*this == rhs); }

  void Storage::Save(const std::filesystem::path &fname, const std::string &path,
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
        // split path into group and name
        std::filesystem::path p(path);
        std::filesystem::path grouppath = p.parent_path();
        std::string datasetname = p.filename().string();
        if (datasetname.empty()) datasetname = "Storage";
        // create group
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
        this->to_hdf5(location, overwrite, datasetname);
        h5file.close();
        return;
      } else {  // create binary file
        if (mode == 'x') {
          cytnx_error_msg(std::filesystem::exists(fname),
                          "[ERROR] File %s already exists. Use mode 'w' to overwrite.", fname);
        } else {
          cytnx_error_msg(mode != 'w', "[ERROR] Unknown mode '%c' for writing to binary file.",
                          mode);
        }
        f.open(fname, std::ios::out | std::ios::trunc | std::ios::binary);
      }
    } else {  // create binary file with standard extension
      std::filesystem::path fnameext = fname;
      fnameext += ".cyst";
      cytnx_warning_msg(true,
                        "Missing file extension in fname '%s'. I am adding the extension '.cyst'. "
                        "This is deprecated, please provide the file extension in the future.\n",
                        fname.c_str());
      if (mode == 'x') {
        cytnx_error_msg(std::filesystem::exists(fnameext),
                        "[ERROR] File %s already exists. Use mode 'w' to overwrite.",
                        fnameext.c_str());
      } else {
        cytnx_error_msg(mode != 'w', "[ERROR] Unknown mode '%c' for writing to binary file.", mode);
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
  void Storage::Save(const char *fname, const std::string &path, const char mode) const {
    this->Save(std::filesystem::path(fname), path, mode);
  }

  Storage Storage::Load(const std::filesystem::path &fname, const std::string &path,
                        const bool restore_device) {
    Storage out;
    out.Load_(fname, path, restore_device);
    return out;
  }
  Storage Storage::Load(const char *fname, const std::string &path, const bool restore_device) {
    return Storage::Load(std::filesystem::path(fname), path, restore_device);
  }

  void Storage::Load_(const std::filesystem::path &fname, const std::string &path,
                      const bool restore_device) {
    std::string ext = fname.extension().string();
    if (ext == ".h5" || ext == ".hdf5" || ext == ".H5" || ext == ".HDF5" || ext == ".hdf" ||
        ext == ".HDF") {  // load HDF5
      H5::H5File h5file(fname, H5F_ACC_RDONLY);
      // split path into group and name
      std::filesystem::path p(path);
      std::string grouppath = p.parent_path().generic_string();
      std::string datasetname = p.filename().string();
      if (datasetname.empty()) datasetname = "Storage";
      // open group
      H5::Group location;
      try {
        location = h5file.openGroup(grouppath.empty() ? "/" : grouppath);
      } catch (const H5::Exception &e) {
        std::cerr << e.getDetailMsg() << std::endl;
        cytnx_error_msg(true, "[ERROR] HDF5 path '%s' not found or is not a group in file '%s'.",
                        grouppath.c_str(), fname.c_str());
      }
      // read data
      this->from_hdf5(location, datasetname, restore_device);
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
  void Storage::Load_(const char *fname, const std::string &path, const bool restore_device) {
    this->Load_(std::filesystem::path(fname), path, restore_device);
  }

  void Storage::to_hdf5(H5::Group &location, const bool overwrite, const std::string &name) const {
    if (overwrite) {  // delete previous data
      if (location.nameExists(name)) location.unlink(name);
    }

    hsize_t Nelem = this->size();
    H5::DataSpace dataspace(1, &Nelem);
    H5::DataType datatype = Type.dtype_to_hdf5_type(this->dtype());
    H5::DataSet dataset = location.createDataSet(name, datatype, dataspace);
    this->data_to_hdf5(dataset, datatype);
    if (this->device() != Device.cpu) {
      H5::Attribute attr =
        dataset.createAttribute("device", H5::PredType::NATIVE_INT, H5::DataSpace(H5S_SCALAR));
      int device = this->device();
      attr.write(H5::PredType::NATIVE_INT, &device);
    }
  }

  void Storage::from_hdf5(H5::Group &location, const std::string &name, const bool restore_device) {
    H5::DataSet dataset = location.openDataSet(name);
    H5::DataType datatype = dataset.getDataType();
    unsigned int dtype = Type.from_hdf5_type(datatype);
    H5::DataSpace dataspace = dataset.getSpace();
    auto Nelem = dataspace.getSimpleExtentNpoints();

    int device = Device.cpu;
    if (restore_device && dataset.attrExists("device")) {
      H5::Attribute attr = dataset.openAttribute("device");
      datatype = dataset.getDataType();
      cytnx_error_msg(
        datatype.getSize() != sizeof(int),
        "[ERROR] 'device' bit-length mismatch. File: %zu bytes, expected: %zu bytes.\n",
        datatype.getSize(), sizeof(int));
      attr.read(datatype, &device);
    }

    this->data_from_hdf5(dataset, Nelem, dtype, datatype, device);
  }

  void Storage::data_to_hdf5(H5::DataSet &dataset, H5::DataType &hdf5type) const {
    if (this->device() == Device.cpu) {
      dataset.write(this->data(), hdf5type);
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(this->device()));
      void *htmp = malloc(Type.typeSize(this->dtype()) * this->size());
      checkCudaErrors(cudaMemcpy(htmp, this->_impl->data(),
                                 Type.typeSize(this->dtype()) * this->size(),
                                 cudaMemcpyDeviceToHost));
      dataset.write(htmp, hdf5type);
      free(htmp);
#else
      cytnx_error_msg(true, "ERROR internal fatal error in Save Storage%s", "\n");
#endif
    }
  }

  void Storage::data_from_hdf5(H5::DataSet &dataset, const cytnx_uint64 &Nelem,
                               const unsigned int &dtype, H5::DataType &hdf5type,
                               const int &device) {
    this->_impl = __SII.USIInit[dtype]();
    this->_impl->Init(Nelem, device, false);

    if (device == Device.cpu) {
      dataset.read(this->_impl->data(), hdf5type);
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(device));
      void *htmp = malloc(Type.typeSize(dtype) * Nelem);
      dataset.read(htmp, hdf5type);
      checkCudaErrors(cudaMemcpy(this->_impl->data(), htmp, Type.typeSize(dtype) * Nelem,
                                 cudaMemcpyHostToDevice));
      free(htmp);
#else
      cytnx_error_msg(true, "ERROR internal fatal error in Load Storage%s", "\n");
#endif
    }
  }

  void Storage::to_binary(std::ostream &f) const {
    unsigned int IDDs = 999;
    f.write((char *)&IDDs, sizeof(unsigned int));
    auto write_number = [&f](auto number) {
      f.write(reinterpret_cast<char *>(&number), sizeof(number));
    };
    write_number(this->size());
    write_number(this->dtype());
    write_number(this->device());

    this->data_to_binary(f);
  }

  void Storage::from_binary(std::istream &f, const bool restore_device) {
    unsigned long long Nelem;
    unsigned int dtype;
    int device;
    // checking IDD
    unsigned int tmpIDDs;
    f.read((char *)&tmpIDDs, sizeof(unsigned int));
    if (tmpIDDs != 999) {
      cytnx_error_msg(true, "[ERROR] the Load file is not the Storage object!\n", "%s");
    }

    f.read((char *)&Nelem, sizeof(unsigned long long));
    f.read((char *)&dtype, sizeof(unsigned int));
    f.read((char *)&device, sizeof(int));

    if (restore_device) {
      if (device != Device.cpu && device >= Device.Ngpus) {
        cytnx_warning_msg(true,
                          "[Warning!!] the original device ID does not exists. the tensor will be "
                          "put on CPU, please use .to() or .to_() to move to desire devices.%s",
                          "\n");
        device = Device.cpu;
      }
    } else {
      device = Device.cpu;
    }
    this->data_from_binary(f, Nelem, dtype, device);
  }

  void Storage::data_to_binary(std::ostream &f) const {
    if (this->device() == Device.cpu) {
      f.write((char *)this->_impl->data(), Type.typeSize(this->dtype()) * this->size());
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(this->device()));
      void *htmp = malloc(Type.typeSize(this->dtype()) * this->size());
      checkCudaErrors(cudaMemcpy(htmp, this->_impl->data(),
                                 Type.typeSize(this->dtype()) * this->size(),
                                 cudaMemcpyDeviceToHost));
      f.write((char *)htmp, Type.typeSize(this->dtype()) * this->size());
      free(htmp);
#else
      cytnx_error_msg(true, "ERROR internal fatal error in Save Storage%s", "\n");
#endif
    }
  }

  void Storage::data_from_binary(std::istream &f, const cytnx_uint64 &Nelem,
                                 const unsigned int &dtype, const int &device) {
    // before enter this func, make sure
    // 1. dtype is not void.
    // 2. the Nelement is consistent and smaller than the file size, and should not be zero!
    this->_impl = __SII.USIInit[dtype]();
    this->_impl->Init(Nelem, device, false);
    if (device == Device.cpu) {
      f.read((char *)this->_impl->data(), Type.typeSize(dtype) * Nelem);
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(device));
      void *htmp = malloc(Type.typeSize(dtype) * Nelem);
      f.read((char *)htmp, Type.typeSize(dtype) * Nelem);
      checkCudaErrors(cudaMemcpy(this->_impl->data(), htmp, Type.typeSize(dtype) * Nelem,
                                 cudaMemcpyHostToDevice));
      free(htmp);
#else
      cytnx_error_msg(true, "ERROR internal fatal error in Load Storage%s", "\n");
#endif
    }
  }

  void Storage::Tofile(const std::filesystem::path &fname) const {
    fstream f;
    f.open(fname, ios::out | ios::trunc | ios::binary);
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for save.%s", "\n");
    }
    this->data_to_binary(f);
    f.close();
  }
  void Storage::Tofile(const char *fname) const {
    fstream f;
    string ffname = string(fname);
    f.open(ffname, ios::out | ios::trunc | ios::binary);
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for save.%s", "\n");
    }
    this->data_to_binary(f);
    f.close();
  }
  void Storage::Tofile(fstream &f) const {
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for save.%s", "\n");
    }
    this->data_to_binary(f);
  }

  Storage Storage::Fromfile(const char *fname, const unsigned int &dtype, const cytnx_int64 &count,
                            const int device) {
    return Storage::Fromfile(std::filesystem::path(fname), dtype, count, device);
  }
  Storage Storage::Fromfile(const std::filesystem::path &fname, const unsigned int &dtype,
                            const cytnx_int64 &count, const int device) {
    cytnx_error_msg(dtype == Type.Void, "[ERROR] Cannot have Void dtype.%s", "\n");
    cytnx_error_msg(count == 0, "[ERROR] count cannot be zero!%s", "\n");

    Storage out;
    cytnx_uint64 Nbytes;
    cytnx_uint64 Nelem;

    // check size:
    ifstream jf;
    jf.open(fname, ios::ate | ios::binary);
    if (!jf.is_open()) {
      cytnx_error_msg(true, "[ERROR] Cannot open file '%s'.\n", fname.c_str());
    }
    Nbytes = jf.tellg();
    jf.close();

    fstream f;
    // check if type match?
    cytnx_error_msg(Nbytes % Type.typeSize(dtype),
                    "[ERROR] the total size of file is not an interval of assigned dtype.%s", "\n");

    // check count smaller than Nelem:
    if (count < 0)
      Nelem = Nbytes / Type.typeSize(dtype);
    else {
      cytnx_error_msg(count > Nelem, "[ERROR] count exceed the total # of elements %d in file.\n",
                      Nelem);
      Nelem = count;
    }

    f.open(fname, ios::in | ios::binary);
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] Cannot open file '%s'.\n", fname.c_str());
    }
    out.data_from_binary(f, Nelem, dtype, device);
    f.close();
    return out;
  }

  Scalar::Sproxy Storage::operator()(const cytnx_uint64 &idx) {
    Scalar::Sproxy out(this->_impl, idx);
    return out;
  }

  template <class T>
  std::vector<T> Storage::vector() {
    T tmp;
    cytnx_error_msg(Type.cy_typeid(tmp) != this->dtype(),
                    "[ERROR] the dtype of current Storage does not match assigned vector type.%s",
                    "\n");

    std::vector<T> out(this->size());
    Storage S;
    if (this->device() != Device.cpu) {
      S = this->to(Device.cpu);
      memcpy(&out[0], S.data(), sizeof(T) * this->size());
    } else {
      memcpy(&out[0], this->data(), sizeof(T) * this->size());
    }

    return out;
  }

  template <>
  std::vector<cytnx_bool> Storage::vector<cytnx_bool>() {
    bool tmp;
    cytnx_error_msg(this->dtype() != Type.Bool,
                    "[ERROR] the dtype of current Storage does not match assigned vector type.%s",
                    "\n");

    std::vector<bool> out(this->size());
    Storage S;
    if (this->device() != Device.cpu) {
      S = this->to(Device.cpu);
    } else {
      S = *this;
    }
    for (cytnx_uint64 i = 0; i < S.size(); i++) {
      out[i] = S.at<bool>(i);
    }

    return out;
  }

  template std::vector<cytnx_complex128> Storage::vector<cytnx_complex128>();
  template std::vector<cytnx_complex64> Storage::vector<cytnx_complex64>();
  template std::vector<cytnx_double> Storage::vector<cytnx_double>();
  template std::vector<cytnx_float> Storage::vector<cytnx_float>();
  template std::vector<cytnx_uint64> Storage::vector<cytnx_uint64>();
  template std::vector<cytnx_int64> Storage::vector<cytnx_int64>();
  template std::vector<cytnx_uint32> Storage::vector<cytnx_uint32>();
  template std::vector<cytnx_int32> Storage::vector<cytnx_int32>();
  template std::vector<cytnx_uint16> Storage::vector<cytnx_uint16>();
  template std::vector<cytnx_int16> Storage::vector<cytnx_int16>();
  // template std::vector<cytnx_bool> Storage::vector<cytnx_bool>();

}  // namespace cytnx
