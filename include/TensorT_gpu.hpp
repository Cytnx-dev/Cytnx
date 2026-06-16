#ifndef CYTNX_TENSORT_GPU_HPP_
#define CYTNX_TENSORT_GPU_HPP_

#ifndef UNI_GPU
  #error "TensorT_gpu.hpp requires UNI_GPU"
#endif

#include "Device.hpp"
#include "TensorT_cpu.hpp"
#include "cytnx_error.hpp"

namespace cytnx {

  struct cuda_space {};

  struct cuda_access {
    using space = cuda_space;
    int device = Device.cuda;
  };

  namespace tensor_t_detail {

    template <>
    inline cuda_access make_access<cuda_access>(int device) {
      cytnx_error_msg(device < Device.cuda,
                      "[ERROR] Attempt to create a CUDA TensorT from CPU Tensor.%s", "\n");
      return cuda_access{device};
    }

    inline int access_device(cuda_access access) { return access.device; }

  }  // namespace tensor_t_detail

}  // namespace cytnx

#endif  // CYTNX_TENSORT_GPU_HPP_
