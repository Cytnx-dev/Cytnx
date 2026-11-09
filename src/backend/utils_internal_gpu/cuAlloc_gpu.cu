#include "cuAlloc_gpu.hpp"

namespace cytnx {
  namespace utils_internal {
#ifdef UNI_GPU
    // void* Calloc_cpu(const cytnx_uint64 &N, const cytnx_uint64 &perelem_bytes){
    //     return calloc(M,perelem_bytes);
    // }
    // An empty (zero-byte) buffer is represented by a null pointer, matching the
    // Storage destructor's `data() != nullptr` free guard. cudaMallocManaged(0)
    // is not guaranteed to return a pointer that cudaFree later accepts (unlike
    // glibc malloc(0), which the CPU path relies on), so a zero-length request
    // that reached the driver would crash on destruction with
    // cudaErrorInvalidValue (#1089). Return nullptr instead; every caller copies
    // via cudaMemcpy*(..., 0) (a no-op) and frees with cudaFree(nullptr) (valid).
    void* cuCalloc_gpu(const cytnx_uint64& N, const cytnx_uint64& perelem_bytes) {
      if (N == 0 || perelem_bytes == 0) return nullptr;
      void* ptr;
      checkCudaErrors(cudaMallocManaged((void**)&ptr, perelem_bytes * N));
      checkCudaErrors(cudaMemset(ptr, 0, perelem_bytes * N));
      return ptr;
    }
    void* cuMalloc_gpu(const cytnx_uint64& bytes) {
      if (bytes == 0) return nullptr;
      void* ptr;
      checkCudaErrors(cudaMallocManaged(&ptr, bytes));
      return ptr;
    }
#endif
  }  // namespace utils_internal
}  // namespace cytnx
