#include "backend/Storage.hpp"

#include <filesystem>
#include <iostream>

using namespace std;

namespace cytnx {

  Storage_init_interface __SII;

  std::ostream &operator<<(std::ostream &os, const Storage &in) {
    in.print();
    return os;
  }

  bool Storage::operator==(const Storage &rhs) {
    cytnx_error_msg(this->dtype() != rhs.dtype(),
                    "[ERROR] cannot compare two Storage with different type.%s", "\n");
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

  void Storage::Save(const std::string &fname) const {
    fstream f;
    if (std::filesystem::path(fname).has_extension()) {
      // filename extension is given
      f.open(fname, ios::out | ios::trunc | ios::binary);
    } else {
      // add filename extension
      cytnx_warning_msg(true,
                        "Missing file extension in fname '%s'. I am adding the extension '.cyst'. "
                        "This is deprecated, please provide the file extension in the future.\n",
                        fname.c_str());
      f.open((fname + ".cyst"), ios::out | ios::trunc | ios::binary);
    }
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for save.%s", "\n");
    }
    this->to_binary(f);
    f.close();
  }
  void Storage::Save(const char *fname) const { this->Save(string(fname)); }

  void Storage::Tofile(const std::string &fname) const {
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

  Storage Storage::Fromfile(const char *fname, const unsigned int &dtype, const cytnx_int64 &count,
                            const bool restore_device) {
    return Storage::Fromfile(string(fname), dtype, count, restore_device);
  }
  Storage Storage::Fromfile(const std::string &fname, const unsigned int &dtype,
                            const cytnx_int64 &count, const bool restore_device) {
    cytnx_error_msg(dtype == Type.Void, "[ERROR] cannot have Void dtype.%s", "\n");
    cytnx_error_msg(count == 0, "[ERROR] count cannot be zero!%s", "\n");

    Storage out;
    cytnx_uint64 Nbytes;
    cytnx_uint64 Nelem;

    // check size:
    ifstream jf;
    // std::cout << fname << std::endl;
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
    out.data_from_binary(f, Nelem, dtype, restore_device);
    f.close();
    return out;
  }

  Storage Storage::Load(const std::string &fname, const bool restore_device) {
    Storage out;
    out.Load_(fname, restore_device);
    return out;
  }
  Storage Storage::Load(const char *fname, const bool restore_device) {
    return Storage::Load(string(fname), restore_device);
  }

  void Storage::Load_(const std::string &fname, const bool restore_device) {
    fstream f;
    f.open(fname, ios::in | ios::binary);
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] Cannot open file '%s'.\n", fname.c_str());
    }
    this->from_binary(f, restore_device);
    f.close();
  }
  void Storage::Load_(const char *fname, const bool restore_device) {
    this->Load_(string(fname), restore_device);
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
