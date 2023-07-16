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
    ~Device_class();
    // void cudaDeviceSynchronize();
  };
  /// @endcond

  /**
   * @brief data on which devices.
   *
   * @details This is the variable about the data on which devices .\n
   *     You can use it as following:
   *     \code
   *     int device = Device.cpu;
   *     \endcode
   *     The supported enumerations are as following:
   *
   *  enumeration  |  description
   * --------------|--------------------
   *  cpu          |  -1, on cpu
   *  cuda         |  0, on cuda
   */
  extern Device_class Device;
}  // namespace cytnx
#endif
