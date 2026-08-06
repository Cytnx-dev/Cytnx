#include "cuLibraryHandle_gpu.hpp"

#if defined(UNI_GPU)

  #include <mutex>
  #include <unordered_map>

  #include "cuda_runtime_api.h"

namespace cytnx {
  namespace utils_internal {

    cublasHandle_t get_cublas_handle() {
      int device = -1;
      checkCudaErrors(cudaGetDevice(&device));

      // Never destroyed by design -- see the note on the declaration. Function-local statics also
      // keep initialization lazy, so a CPU-only run never creates a CUDA context here.
      static std::mutex mutex;
      static auto* handles = new std::unordered_map<int, cublasHandle_t>();

      std::lock_guard<std::mutex> lock(mutex);
      auto found = handles->find(device);
      if (found != handles->end()) {
        return found->second;
      }

      cublasHandle_t handle;
      checkCudaErrors(cublasCreate(&handle));
      handles->emplace(device, handle);
      return handle;
    }

    cusolverDnHandle_t get_cusolverdn_handle() {
      int device = -1;
      checkCudaErrors(cudaGetDevice(&device));

      static std::mutex mutex;
      static auto* handles = new std::unordered_map<int, cusolverDnHandle_t>();

      std::lock_guard<std::mutex> lock(mutex);
      auto found = handles->find(device);
      if (found != handles->end()) {
        return found->second;
      }

      cusolverDnHandle_t handle;
      checkCudaErrors(cusolverDnCreate(&handle));
      handles->emplace(device, handle);
      return handle;
    }

  }  // namespace utils_internal
}  // namespace cytnx

#endif
