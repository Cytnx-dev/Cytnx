#ifndef CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUDAMEMCPYTRUNCATION_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUDAMEMCPYTRUNCATION_H_

#include <iostream>
#include <vector>

#include "Tensor.hpp"
#include "Type.hpp"
#include "cytnx_error.hpp"

namespace cytnx {
  namespace linalg_internal {

#ifdef UNI_GPU
    // GPU counterpart of linalg_internal::memcpyTruncation; truncates the packed Svd/Gesvd output
    // tens = [S, U?, vT?] in place and appends the error tensor when return_err != 0. See the CPU
    // declaration in linalg_internal_cpu/memcpyTruncation.hpp for the full contract.
    void cudaMemcpyTruncation(std::vector<Tensor> &tens, cytnx_uint64 keepdim, double err,
                              bool is_U, bool is_vT, unsigned int return_err, cytnx_uint64 mindim);
#endif

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUDAMEMCPYTRUNCATION_H_
