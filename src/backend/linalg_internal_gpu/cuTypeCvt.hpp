#ifndef CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUTYPECVT_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUTYPECVT_H_

#include <complex>

#include "Type.hpp"

namespace cytnx {
  namespace linalg_internal {

    // Map a Cytnx CPU value type to the CUDA-native type used inside device kernels.
    // Only complex differs: std::complex<T> -> cuda::std::complex<T>; every other type
    // is identical on host and device. This keeps the *logical* GPU dispatch on the
    // ordinary Cytnx value types (as_storage_variant() + type_promote_t, shared with the
    // CPU path) and confines the CUDA-native complex representation to the kernel-launch
    // boundary, rather than maintaining a parallel Type_list_gpu / type_promote_gpu_t
    // hierarchy (#1013).
    template <typename T>
    struct to_cuda {
      using type = T;
    };
    template <typename T>
    struct to_cuda<std::complex<T>> {
      using type = cuda::std::complex<T>;
    };
    template <typename T>
    using to_cuda_t = typename to_cuda<T>::type;

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUTYPECVT_H_
