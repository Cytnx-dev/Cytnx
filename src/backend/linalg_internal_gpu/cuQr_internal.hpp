#ifndef CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUQR_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUQR_INTERNAL_H_

#include "Tensor.hpp"

namespace cytnx {
  namespace linalg_internal {

#ifdef UNI_GPU
    /**
     * @brief Reduced QR factorization of a dense matrix on the GPU with cuSOLVER (`geqrf` followed
     * by `orgqr` / `ungqr`).
     *
     * With K = min(M, N), factors the M x N matrix `in` as `in = Q R`, where Q (M x K) has
     * orthonormal columns and R (K x N) is upper triangular (upper trapezoidal when M < N) with a
     * real diagonal. The factorization is the unpivoted LAPACK Householder one, so Q and R agree
     * with the CPU path (`QR_ii`) up to rounding.
     *
     * The scalar factors of the Householder reflectors (`tau`) are used internally and not
     * returned.
     *
     * @param[in] in contiguous rank-2 M x N tensor on a CUDA device; Float, Double, ComplexFloat
     * or ComplexDouble. Not modified. Both extents must be nonzero and fit a 32-bit `int`.
     * @param[out] Q pre-allocated M x K tensor, same dtype and device as `in`.
     * @param[out] R pre-allocated K x N tensor, same dtype and device as `in`.
     *
     * The CUDA device of `in` must be current on the calling thread.
     */
    void cuQr(const Tensor &in, Tensor &Q, Tensor &R);
#endif

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUQR_INTERNAL_H_
