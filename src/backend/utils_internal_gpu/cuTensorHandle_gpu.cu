#include "cuTensorHandle_gpu.hpp"

#if defined(UNI_GPU) && defined(UNI_CUTENSOR)

  #include <mutex>
  #include <unordered_map>

  #include "cuda_runtime_api.h"

namespace cytnx {
  namespace utils_internal {

    cutensorHandle_t GetCutensorHandle() {
      int device = -1;
      checkCudaErrors(cudaGetDevice(&device));

      // Never destroyed by design -- see the note on the declaration. Function-local statics also
      // keep initialization lazy, so a CPU-only run never creates a CUDA context here.
      static std::mutex mutex;
      static auto* handles = new std::unordered_map<int, cutensorHandle_t>();

      std::lock_guard<std::mutex> lock(mutex);
      auto found = handles->find(device);
      if (found != handles->end()) {
        return found->second;
      }

      cutensorHandle_t handle;
      checkCudaErrors(cutensorCreate(&handle));
      handles->emplace(device, handle);
      return handle;
    }

  }  // namespace utils_internal
}  // namespace cytnx

#endif
