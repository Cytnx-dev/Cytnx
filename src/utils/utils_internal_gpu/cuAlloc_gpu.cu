#include "cuAlloc_gpu.hpp"

using namespace std;

namespace cytnx {
  namespace utils_internal {
#ifdef UNI_GPU
    // void* Calloc_cpu(const cytnx_uint64 &N, const cytnx_uint64 &perelem_bytes){
    //     return calloc(M,perelem_bytes);
    // }
    void* cuCalloc_gpu(const cytnx_uint64& N, const cytnx_uint64& perelem_bytes) {
      void* ptr;
      checkCudaErrors(cudaMallocManaged((void**)&ptr, perelem_bytes * N));
      checkCudaErrors(cudaMemset(ptr, 0, perelem_bytes * N));
      return ptr;
    }
    void* cuMalloc_gpu(const cytnx_uint64& bytes) {
      void* ptr;
      checkCudaErrors(cudaMallocManaged(&ptr, bytes));
      return ptr;
    }
#endif
  }  // namespace utils_internal
}  // namespace cytnx
