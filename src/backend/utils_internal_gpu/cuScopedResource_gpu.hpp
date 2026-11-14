#ifndef CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUSCOPEDRESOURCE_GPU_H_
#define CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUSCOPEDRESOURCE_GPU_H_

// backend/utils_internal_gpu and backend/linalg_internal_gpu are added by CMake only under
// USE_CUDA, so this header can never legitimately be compiled without UNI_GPU. Fail loudly rather
// than expanding to nothing: an accidental include from a CPU translation unit would otherwise
// surface as "DeviceBuffer is not a member of cytnx::utils_internal", pointing at the use site
// instead of the include.
#if !defined(UNI_GPU)
  #error "cuScopedResource_gpu.hpp is CUDA-only; include it from a .cu under backend/*_gpu."
#endif

#include <cstddef>
#include <utility>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "cytnx_error.hpp"
#include "Type.hpp"
#include "backend/utils_internal_gpu/cuAlloc_gpu.hpp"

namespace cytnx {
  namespace utils_internal {

    /**
     * @brief Owns a `cudaMalloc`ed device buffer and frees it on scope exit.
     *
     * `T` is the element type only: it fixes `sizeof(T)` for the allocation and the pointer type
     * `get()` hands back. It carries no CUDA-specific requirement -- these buffers hold `int`,
     * `cytnx_double` and `cuDoubleComplex` alike, and nothing here is ever dereferenced on the
     * host unless the storage came from `managed()`.
     *
     * The cuSOLVER wrappers in `linalg_internal_gpu/` raise their LAPACK-`info` error *before*
     * their cleanup block, so an unconverged factorization used to leak every device allocation
     * in the function (#1146). Ownership makes cleanup independent of the exit path, which manual
     * `cudaFree` calls at the tail of a function cannot be -- especially where a function has more
     * than one throw site and each has a different set of live allocations.
     *
     * A zero-element request leaves the buffer empty and allocates nothing. `cudaMalloc(ptr, 0)`
     * yields a pointer that cannot be freed, which is the defect fixed for `Storage` in #1126;
     * `lwork` legitimately comes back as 0 from the cuSOLVER buffer-size queries, so this case is
     * reachable here rather than hypothetical.
     */
    template <class T>
    class DeviceBuffer {
     public:
      DeviceBuffer() = default;

      explicit DeviceBuffer(std::size_t count) {
        if (count == 0) return;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&ptr), count * sizeof(T)));
      }

      ~DeviceBuffer() {
        // Deliberately unchecked: a destructor must not terminate the process, and
        // `checkCudaErrors` exits on failure.
        if (ptr) cudaFree(ptr);
      }

      DeviceBuffer(const DeviceBuffer &) = delete;
      DeviceBuffer &operator=(const DeviceBuffer &) = delete;

      DeviceBuffer(DeviceBuffer &&other) noexcept : ptr(std::exchange(other.ptr, nullptr)) {}

      DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
        if (this != &other) {
          if (ptr) cudaFree(ptr);
          ptr = std::exchange(other.ptr, nullptr);
        }
        return *this;
      }

      /**
       * @brief Allocates `count` elements of CUDA *managed* memory and owns them.
       *
       * The constructor uses `cudaMalloc`, whose memory is device-only. This factory uses
       * `cuMalloc_gpu` (`cudaMallocManaged`), which is also host-accessible -- `cuDet_internal`
       * needs that, because it reads the factorized diagonal directly from the host after the
       * cuSOLVER call.
       *
       * Only the allocation differs: `cudaFree` releases both kinds, so teardown, move semantics
       * and the zero-element case are identical to the constructor's.
       */
      static DeviceBuffer managed(std::size_t count) {
        DeviceBuffer buffer;
        if (count == 0) return buffer;
        buffer.ptr = reinterpret_cast<T *>(cuMalloc_gpu(count * sizeof(T)));
        return buffer;
      }

      T *get() const { return ptr; }

     private:
      T *ptr = nullptr;
    };

    // [Note] There is no scoped cuSOLVER handle type here. #1146 originally gave the handle
    // scope-bound ownership so a throwing `info` check could not leak it; #1144 then made handles
    // shared per device and process-lifetime, which removes them from the set of leakable
    // resources entirely rather than freeing them correctly. See
    // `cuLibraryHandle_gpu.hpp::get_cusolverdn_handle`.

    /**
     * @brief Owns a `gesvdjInfo_t` (Jacobi SVD parameters) and destroys it on scope exit.
     *
     * `cuGeSvd_internal.cu` created one per call and never destroyed it -- `cusolverDnDestroy-
     * GesvdjInfo` appeared nowhere in the repo -- leaking ~2 KB of host memory per GPU SVD
     * (#1145).
     */
    class GesvdjInfo {
     public:
      GesvdjInfo() { checkCudaErrors(cusolverDnCreateGesvdjInfo(&info)); }

      ~GesvdjInfo() {
        if (info) cusolverDnDestroyGesvdjInfo(info);
      }

      GesvdjInfo(const GesvdjInfo &) = delete;
      GesvdjInfo &operator=(const GesvdjInfo &) = delete;

      gesvdjInfo_t get() const { return info; }

     private:
      gesvdjInfo_t info = nullptr;
    };

  }  // namespace utils_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUSCOPEDRESOURCE_GPU_H_
