#include "Storage.hpp"

#include <iostream>

using namespace std;

namespace cytnx {

  std::ostream &operator<<(std::ostream &os, Storage &in) {
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
}  // namespace cytnx
