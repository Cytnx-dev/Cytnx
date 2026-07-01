#include "cuSetZeros_gpu.hpp"

namespace cytnx {
  namespace utils_internal {
#ifdef UNI_GPU
    void cuSetZeros(void *c_ptr, const cytnx_uint64 &bytes) {
      checkCudaErrors(cudaMemset(c_ptr, 0, bytes));
    }
#endif
  }  // namespace utils_internal
}  // namespace cytnx
