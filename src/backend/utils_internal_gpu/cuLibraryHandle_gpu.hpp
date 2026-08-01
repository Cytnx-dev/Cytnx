#ifndef CYTNX_BACKEND_UTILS_INTERNAL_GPU_CULIBRARYHANDLE_GPU_H_
#define CYTNX_BACKEND_UTILS_INTERNAL_GPU_CULIBRARYHANDLE_GPU_H_

#include "cytnx_error.hpp"
#include "Type.hpp"

#if defined(UNI_GPU)
  #include <cublas_v2.h>
  #include <cusolverDn.h>
#endif

namespace cytnx {
  namespace utils_internal {

#if defined(UNI_GPU)

    /**
     * @brief The shared cuBLAS handle for the CUDA device current at the time of the call,
     * creating it on first use.
     *
     * `cublasCreate` + `cublasDestroy` costs ~330 us in steady state, and `linalg_internal_gpu/`
     * used to pay it around every single call (#1144). Measured against the work it wraps, a
     * `cublasDgemm` on an n x n double matrix:
     *
     *     n=32   dgemm   9.5 us   handle  186.5 us   handle is 19.6x the work
     *     n=128  dgemm  31.1 us   handle  195.3 us   handle is  6.3x the work
     *     n=512  dgemm 462.4 us   handle  299.3 us   handle is  0.6x the work
     *
     * Below roughly n=400 the handle cost more than the multiply it existed to perform, and
     * tensor-network workloads spend much of their time under that size.
     *
     * Keyed by device, not global: a handle is bound to whichever device was current when it was
     * created, so one process-wide handle would silently drive the wrong GPU in a multi-device
     * process. Callers get the handle for the current device, which preserves each call site's
     * existing device selection exactly.
     *
     * Sharing is safe here because no live call site mutates handle state -- there is no
     * `cublasSetStream`, `cublasSetPointerMode`, `cublasSetMathMode` or `cublasSetAtomicsMode`
     * anywhere in the compiled sources. If per-call streams are ever introduced, that state must
     * be set per call or this cache revisited.
     *
     * Thread-safe. Handles intentionally live for the whole process and are never destroyed: a
     * static destructor would run after main, by which point the CUDA runtime may already have
     * torn down the context that owns them, making `cublasDestroy` undefined behaviour. The cost
     * is bounded at one handle per device actually used.
     */
    cublasHandle_t get_cublas_handle();

    /**
     * @brief The shared cuSOLVER dense handle for the CUDA device current at the time of the call,
     * creating it on first use.
     *
     * `cusolverDnCreate` + `cusolverDnDestroy` costs ~456 us in steady state and was likewise paid
     * per call (#1144). A cuSOLVER handle also holds a substantial device-side workspace -- ~12 MB,
     * which is why leaking one per call dominated #1146 -- so reusing it removes repeated
     * allocation of that workspace as well as the setup cost.
     *
     * Same device-keying, thread-safety and lifetime rationale as `get_cublas_handle`. The only
     * `cusolverDnSetStream` in the tree is in `cuEig_internal.cu`, which is orphaned source that
     * is never compiled (#1142), so no live call site mutates handle state.
     */
    cusolverDnHandle_t get_cusolverdn_handle();

#endif  // UNI_GPU

  }  // namespace utils_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_UTILS_INTERNAL_GPU_CULIBRARYHANDLE_GPU_H_
