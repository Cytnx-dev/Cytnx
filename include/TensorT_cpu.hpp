#ifndef CYTNX_TENSORT_CPU_HPP_
#define CYTNX_TENSORT_CPU_HPP_

#include "Device.hpp"
#include "cytnx_error.hpp"

namespace cytnx {

  struct host_space {};

  struct host_access {
    using space = host_space;
  };

  namespace tensor_t_detail {

    template <class Access>
    Access make_access(int device);

    template <>
    inline host_access make_access<host_access>(int device) {
      cytnx_error_msg(device != Device.cpu,
                      "[ERROR] Attempt to create a host TensorT from Tensor on device %d.%s",
                      device, "\n");
      return {};
    }

    inline int access_device(host_access) { return Device.cpu; }

    inline bool access_accepts_device(host_access, int device) { return device == Device.cpu; }

  }  // namespace tensor_t_detail

}  // namespace cytnx

#endif  // CYTNX_TENSORT_CPU_HPP_
