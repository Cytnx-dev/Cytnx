#include "Device.hpp"
#include "cytnx_error.hpp"

namespace cytnx {

#ifdef UNI_GPU
  void Device_class::cudaDeviceSynchronize() { cudaDeviceSynchronize(); }
#else
  // See Device.cpp
#endif

}  // namespace cytnx
