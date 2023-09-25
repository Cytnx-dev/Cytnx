#ifndef _TORCH_BKND_TYPE_CONVERT_H_
#define _TORCH_BKND_TYPE_CONVERT_H_

#ifdef BACKEND_TORCH

  #include <torch/torch.h>
namespace cytnx {

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
}  // namespace cytnx
#endif  // BACKEND_TORCH header guard

#endif
