#ifndef CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUTENSORHANDLE_GPU_H_
#define CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUTENSORHANDLE_GPU_H_

#include "Type.hpp"
#include "cytnx_error.hpp"

#if defined(UNI_GPU) && defined(UNI_CUTENSOR)
  #include <cutensor.h>
#endif

namespace cytnx {
  namespace utils_internal {

#if defined(UNI_GPU) && defined(UNI_CUTENSOR)

    /**
     * @brief The shared cuTENSOR handle for the CUDA device that is current at the time of the
     * call, creating it on first use.
     *
     * `cutensorCreate` costs roughly 3.4 ms -- flat, regardless of tensor size -- because it
     * initializes the library context and its plan cache. Cytnx used to pay that on *every*
     * `contiguous()`, `permute()`, and `Tensordot`, which made an 8-element permute take 3.5 ms
     * (#1132). Handles are meant to be created once and reused, so they are cached here.
     *
     * Keyed by device, not global: cuTENSOR binds a handle to whichever device was current when
     * `cutensorCreate` ran, and documents that it stays bound. Using one handle across devices
     * would silently drive the wrong GPU. Callers therefore get the handle for the current
     * device, which preserves each call site's existing device selection exactly.
     *
     * Reusing the handle also restores its 64-entry plan cache, which previously died with the
     * handle after a single use -- repeated permutes of the same shape now reuse a plan.
     *
     * Sharing one handle across threads is sanctioned by cuTENSOR, not merely assumed: every
     * entry point Cytnx calls (`cutensorPermute`, `cutensorContract`, `cutensorCreatePlan`,
     * `cutensorCreateTensorDescriptor`, `cutensorCreatePlanPreference`,
     * `cutensorEstimateWorkspaceSize`, `cutensorPlanGetAttribute`) is documented
     * "no reentrant, and thread-safe", and the plan cache "can be shared across different threads
     * in a thread-safe manner". The one non-thread-safe call, `cutensorHandleResizePlanCache`,
     * is never used here -- the cache stays at its default capacity.
     *
     * This accessor is itself thread-safe. The handles intentionally live for the whole process
     * and are never destroyed:
     * a static destructor would run after main, by which point the CUDA runtime may already have
     * torn down the context that owns them, making `cutensorDestroy` undefined behaviour. The
     * cost is bounded -- one handle per device actually used.
     */
    cutensorHandle_t GetCutensorHandle();

#endif

  }  // namespace utils_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUTENSORHANDLE_GPU_H_
