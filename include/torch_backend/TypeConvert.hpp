#ifndef _TYPECONVERT_H_
#define _TYPECONVERT_H_

// cytnx:
#include "Type.hpp"
#include "Device.hpp"

// pytorch:
#include <torch/torch.h>

namespace cytnx {

  /// @cond
  // typedef torch::TensorOptions (*Tor2Cy_io)(const unsigned int &dtype, const unsigned int
  // &device);

  class TypeCvrt_class {
   public:
    // Cast
    // std::vector<Tor2Cy_io> _t2c;
    TypeCvrt_class();
    torch::TensorOptions Cy2Tor(const unsigned int &dtype, const int &device);
    unsigned int Tor2Cy(const torch::ScalarType &scalar_type);
    torch::ScalarType tStr2Tor_ST(const std::string &dtype_str);
    torch::Device tStr2Tor_Dv(const std::string &device_str);
  };
  extern TypeCvrt_class type_converter;

  /// @endcond

}  // namespace cytnx

#endif
