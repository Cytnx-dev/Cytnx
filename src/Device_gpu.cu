#include "Device.hpp"
#include "cytnx_error.hpp"

using namespace std;
namespace cytnx {

#ifdef UNI_GPU
  void Device_class::cudaDeviceSynchronize() { cudaDeviceSynchronize(); }
#else
  // See Device.cpp
#endif

}  // namespace cytnx
