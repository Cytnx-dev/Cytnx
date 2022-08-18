#ifndef _H_Device_
#define _H_Device_
#include <vector>
#include <string>

namespace cytnx {

  /// @cond
  struct __device {
    enum __pybind_device { cpu = -1, cuda = 0 };
  };

  class Device_class {
   public:
    enum : int { cpu = -1, cuda = 0 };
    int Ngpus;
    int Ncpus;
    std::vector<std::vector<bool>> CanAccessPeer;
    Device_class();
    void Print_Property();
    std::string getname(const int &device_id);
    // void cudaDeviceSynchronize();
  };
  /// @endcond

  extern Device_class Device;
}  // namespace cytnx
#endif
