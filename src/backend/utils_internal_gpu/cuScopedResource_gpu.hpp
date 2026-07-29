#ifndef CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUSCOPEDRESOURCE_GPU_H_
#define CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUSCOPEDRESOURCE_GPU_H_

#include <cstddef>
#include <utility>

#include "cytnx_error.hpp"
#include "Type.hpp"

#if defined(UNI_GPU)
  #include <cuda_runtime.h>
  #include <cusolverDn.h>
#endif

namespace cytnx {
  namespace utils_internal {

#if defined(UNI_GPU)

    /**
     * @brief Owns a `cudaMalloc`ed device buffer and frees it on scope exit.
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
       * @brief Takes ownership of a pointer allocated elsewhere, e.g. by `cuMalloc_gpu`.
       *
       * Needed where the buffer must be *managed* memory rather than plain device memory --
       * `cuDet_internal` reads the factorized diagonal from the host, which only works because
       * `cuMalloc_gpu` returns `cudaMallocManaged` memory. `cudaFree` releases both kinds, so
       * ownership is uniform even though allocation is not.
       */
      static DeviceBuffer adopt(T *p) {
        DeviceBuffer buffer;
        buffer.ptr = p;
        return buffer;
      }

      T *get() const { return ptr; }

     private:
      T *ptr = nullptr;
    };

    /// Owns a cuSOLVER dense handle and destroys it on scope exit.
    class CusolverDnHandle {
     public:
      CusolverDnHandle() { checkCudaErrors(cusolverDnCreate(&handle)); }

      ~CusolverDnHandle() {
        if (handle) cusolverDnDestroy(handle);
      }

      CusolverDnHandle(const CusolverDnHandle &) = delete;
      CusolverDnHandle &operator=(const CusolverDnHandle &) = delete;

      cusolverDnHandle_t get() const { return handle; }

     private:
      cusolverDnHandle_t handle = nullptr;
    };

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

#endif  // UNI_GPU

  }  // namespace utils_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUSCOPEDRESOURCE_GPU_H_
