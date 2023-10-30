#include "backend_torch/Type_convert.hpp"
#include "cytnx_error.hpp"
#include "Type.hpp"

using namespace std;
namespace cytnx {

  TypeCvrt_class::TypeCvrt_class() {
    //_t2c = vector<Tor2Cy_io>(N_Type);
  }

  torch::TensorOptions TypeCvrt_class::Cy2Tor(const unsigned int &dtype, const int &device) {
    auto options = torch::TensorOptions();
    if (device < 0) {
      options.device(torch::kCPU);
    } else {
      options.device(torch::kCUDA, device);
    }

    switch (dtype) {
      case Type.Double:
        return options.dtype(torch::kFloat64);
      case Type.Float:
        return options.dtype(torch::kFloat32);
      case Type.ComplexDouble:
        return options.dtype(torch::kComplexDouble);
      case Type.ComplexFloat:
        return options.dtype(torch::kComplexFloat);
      case Type.Int64:
        return options.dtype(torch::kInt64);
      case Type.Int32:
        return options.dtype(torch::kInt32);
      case Type.Int16:
        return options.dtype(torch::kInt16);
      case Type.Uint16:
        cytnx_error_msg(true, "[ERROR] Torch type does not have Uint16.%s", "\n");
        return options;
      case Type.Uint32:
        cytnx_error_msg(true, "[ERROR] Torch type does not have Uint32.%s", "\n");
        return options;
      case Type.Uint64:
        cytnx_error_msg(true, "[ERROR] Torch type does not have Uint64.%s", "\n");
        return options;
      case Type.Bool:
        cytnx_error_msg(true, "[ERROR] Torch type does not have Bool.%s", "\n");
        return options;
      case Type.Void:
        cytnx_error_msg(true, "[ERROR] Torch type does not have Void.%s", "\n");
        return options;
    };
  }

  unsigned int TypeCvrt_class::Tor2Cy(const torch::ScalarType &scalar_type) {
    if (scalar_type == torch::kFloat64) {
      return Type.Double;
    } else if (scalar_type == torch::kFloat32) {
      return Type.Float;
    } else if (scalar_type == torch::kComplexDouble) {
      return Type.ComplexDouble;
    } else if (scalar_type == torch::kComplexFloat) {
      return Type.ComplexFloat;
    } else if (scalar_type == torch::kInt64) {
      return Type.Int64;
    } else if (scalar_type == torch::kInt32) {
      return Type.Int32;
    } else if (scalar_type == torch::kInt16) {
      return Type.Int16;
    } else {
      cytnx_error_msg(true, "[ERROR] Invalid Torch type that is not support in cytnx.%s", "\n");
    }
  }

  torch::ScalarType TypeCvrt_class::tStr2Tor_ST(const std::string &dtype_str) {
    if (dtype_str == "torch.float64") {
      return torch::kFloat64;
    } else if (dtype_str == "torch.float32") {
      return torch::kFloat32;
    } else if (dtype_str == "torch.complex128") {
      return torch::kComplexDouble;
    } else if (dtype_str == "torch.complex64") {
      return torch::kComplexFloat;
    } else if (dtype_str == "torch.int64") {
      return torch::kInt64;
    } else if (dtype_str == "torch.int32") {
      return torch::kInt32;
    } else if (dtype_str == "torch.int64") {
      return torch::kInt64;
    } else {
      cytnx_error_msg(true, "[ERROR] Invalid Torch type that is not support in cytnx.%s", "\n");
    }
  }

  torch::Device TypeCvrt_class::tStr2Tor_Dv(const std::string &device_str) {
    if (device_str == "torch.cpu") {
      return torch::kCPU;
    } else {
      cytnx_error_msg(true, "[ERROR] Invalid Torch type that is not support in cytnx.%s", "\n");
    }
  }

  TypeCvrt_class type_converter;

}  // namespace cytnx
