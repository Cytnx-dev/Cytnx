#ifdef UNI_OMP
  #include <omp.h>
#endif
#include "Storage.hpp"

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
    f.open((fname + ".cyst"), ios::out | ios::trunc | ios::binary);
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for save.%s", "\n");
    }
    this->_Save(f);
    f.close();
  }
  void Storage::Save(const char *fname) const {
    fstream f;
    string ffname = string(fname) + ".cyst";
    f.open(ffname, ios::out | ios::trunc | ios::binary);
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for save.%s", "\n");
    }
    this->_Save(f);
    f.close();
  }
  void Storage::Tofile(const std::string &fname) const {
    fstream f;
    f.open(fname, ios::out | ios::trunc | ios::binary);
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for save.%s", "\n");
    }
    this->_Savebinary(f);
    f.close();
  }
  void Storage::Tofile(const char *fname) const {
    fstream f;
    string ffname = string(fname);
    f.open(ffname, ios::out | ios::trunc | ios::binary);
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for save.%s", "\n");
    }
    this->_Savebinary(f);
    f.close();
  }
  void Storage::Tofile(fstream &f) const {
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for save.%s", "\n");
    }
    this->_Savebinary(f);
  }

  void Storage::_Save(fstream &f) const {
    // header
    // check:
    cytnx_error_msg(!f.is_open(), "[ERROR] invalid fstream!.%s", "\n");

    unsigned int IDDs = 999;
    f.write((char *)&IDDs, sizeof(unsigned int));
    f.write((char *)&this->size(), sizeof(unsigned long long));
    f.write((char *)&this->dtype(), sizeof(unsigned int));
    f.write((char *)&this->device(), sizeof(int));

    // data:
    if (this->device() == Device.cpu) {
      f.write((char *)this->_impl->Mem, Type.typeSize(this->dtype()) * this->size());
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(this->device()));
      void *htmp = malloc(Type.typeSize(this->dtype()) * this->size());
      checkCudaErrors(cudaMemcpy(htmp, this->_impl->Mem,
                                 Type.typeSize(this->dtype()) * this->size(),
                                 cudaMemcpyDeviceToHost));
      f.write((char *)htmp, Type.typeSize(this->dtype()) * this->size());
      free(htmp);

#else
      cytnx_error_msg(true, "ERROR internal fatal error in Save Storage%s", "\n");
#endif
    }
  }
  void Storage::_Savebinary(fstream &f) const {
    // header
    // check:
    cytnx_error_msg(!f.is_open(), "[ERROR] invalid fstream!.%s", "\n");

    // data:
    if (this->device() == Device.cpu) {
      f.write((char *)this->_impl->Mem, Type.typeSize(this->dtype()) * this->size());
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(this->device()));
      void *htmp = malloc(Type.typeSize(this->dtype()) * this->size());
      checkCudaErrors(cudaMemcpy(htmp, this->_impl->Mem,
                                 Type.typeSize(this->dtype()) * this->size(),
                                 cudaMemcpyDeviceToHost));
      f.write((char *)htmp, Type.typeSize(this->dtype()) * this->size());
      free(htmp);

#else
      cytnx_error_msg(true, "ERROR internal fatal error in Save Storage%s", "\n");
#endif
    }
  }

  Storage Storage::Fromfile(const char *fname, const unsigned int &dtype,
                            const cytnx_int64 &count) {
    return Storage::Fromfile(string(fname), dtype, count);
  }
  Storage Storage::Fromfile(const std::string &fname, const unsigned int &dtype,
                            const cytnx_int64 &count) {
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
      cytnx_error_msg(true, "[ERROR] invalid file path for load.%s", "\n");
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
      cytnx_error_msg(true, "[ERROR] invalid file path for load.%s", "\n");
    }
    out._Loadbinary(f, dtype, Nelem);
    f.close();
    return out;
  }
  Storage Storage::Load(const std::string &fname) {
    Storage out;
    fstream f;
    f.open(fname, ios::in | ios::binary);
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for load.%s", "\n");
    }
    out._Load(f);
    f.close();
    return out;
  }
  Storage Storage::Load(const char *fname) {
    Storage out;
    fstream f;
    f.open(fname, ios::in | ios::binary);
    if (!f.is_open()) {
      cytnx_error_msg(true, "[ERROR] invalid file path for load.%s", "\n");
    }
    out._Load(f);
    f.close();
    return out;
  }
  void Storage::_Load(fstream &f) {
    // header
    unsigned long long sz;
    unsigned int dt;
    int dv;

    // check:
    cytnx_error_msg(!f.is_open(), "[ERROR] invalid fstream!.%s", "\n");

    // checking IDD
    unsigned int tmpIDDs;
    f.read((char *)&tmpIDDs, sizeof(unsigned int));
    if (tmpIDDs != 999) {
      cytnx_error_msg(true, "[ERROR] the Load file is not the Storage object!\n", "%s");
    }

    f.read((char *)&sz, sizeof(unsigned long long));
    f.read((char *)&dt, sizeof(unsigned int));
    f.read((char *)&dv, sizeof(int));

    if (dv != Device.cpu) {
      if (dv >= Device.Ngpus) {
        cytnx_warning_msg(true,
                          "[Warning!!] the original device ID does not exists. the tensor will be "
                          "put on CPU, please use .to() or .to_() to move to desire devices.%s",
                          "\n");
        dv = -1;
      }
    }

    this->_impl = __SII.USIInit[dt]();
    this->_impl->Init(sz, dv);

    // data:
    if (dv == Device.cpu) {
      f.read((char *)this->_impl->Mem, Type.typeSize(dt) * sz);
    } else {
#ifdef UNI_GPU
      checkCudaErrors(cudaSetDevice(dv));
      void *htmp = malloc(Type.typeSize(dt) * sz);
      f.read((char *)htmp, Type.typeSize(dt) * sz);
      checkCudaErrors(
        cudaMemcpy(this->_impl->Mem, htmp, Type.typeSize(dt) * sz, cudaMemcpyHostToDevice));
      free(htmp);

#else
      cytnx_error_msg(true, "ERROR internal fatal error in Load Storage%s", "\n");
#endif
    }
  }

  void Storage::_Loadbinary(fstream &f, const unsigned int &dtype, const cytnx_uint64 &Nelem) {
    // before enter this func, makesure
    // 1. dtype is not void.
    // 2. the Nelement is consistent and smaller than the file size, and should not be zero!

    // check:
    cytnx_error_msg(!f.is_open(), "[ERROR] invalid fstream!.%s", "\n");

    this->_impl = __SII.USIInit[dtype]();
    this->_impl->Init(Nelem, Device.cpu);

    f.read((char *)this->_impl->Mem, Type.typeSize(dtype) * Nelem);
  }

  Scalar::Sproxy Storage::operator()(const cytnx_uint64 &idx) {
    Scalar::Sproxy out(this->_impl, idx);
    return out;
  }

}  // namespace cytnx
