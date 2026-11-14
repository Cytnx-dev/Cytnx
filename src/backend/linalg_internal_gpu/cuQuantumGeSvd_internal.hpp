#ifndef CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUQUANTUMGESVD_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUQUANTUMGESVD_INTERNAL_H_

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"
#include "cytnx_error.hpp"
#include "backend/lapack_wrapper.hpp"
#include "Tensor.hpp"
// #include "backend/linalg_internal_interface.hpp"

#ifdef UNI_GPU
  #ifdef UNI_CUQUANTUM
    #include <cutensornet.h>
    #include <cuda_runtime.h>
  #endif
#endif

namespace cytnx {
  namespace linalg_internal {

#ifdef UNI_GPU
  #ifdef UNI_CUQUANTUM
    // Full (untruncated) SVD Tin = U diag(S) vT on the GPU with cuTensorNet. U, S and vT must be
    // allocated by the caller as contiguous tensors of shapes {rows, k}, {k} and {k, cols} with
    // k = min(rows, cols); U and vT have the dtype of Tin and S the matching real dtype.
    // Truncation is done by the callers with cudaMemcpyTruncation.
    void cuQuantumGeSvd_internal_cd(const Tensor &Tin, Tensor &U, Tensor &S, Tensor &vT);
    void cuQuantumGeSvd_internal_cf(const Tensor &Tin, Tensor &U, Tensor &S, Tensor &vT);
    void cuQuantumGeSvd_internal_d(const Tensor &Tin, Tensor &U, Tensor &S, Tensor &vT);
    void cuQuantumGeSvd_internal_f(const Tensor &Tin, Tensor &U, Tensor &S, Tensor &vT);
  #endif
#endif

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUQUANTUMGESVD_INTERNAL_H_
