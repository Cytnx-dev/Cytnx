#ifndef CYTNX_BACKEND_LINALG_INTERNAL_CPU_MEMCPYTRUNCATION_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_CPU_MEMCPYTRUNCATION_H_

#include <vector>

#include "Tensor.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace linalg_internal {

    /**
     * @brief Truncate an svd output to its leading singular values, in place.
     * @param[in,out] tens the packed output of Svd/Gesvd: tens[0] = S (always present), followed by
     *   U (iff is_U) and then vT (iff is_vT). The contained tensors are replaced in place by their
     *   truncated versions. When return_err != 0, the error tensor (terr) is appended as the final
     *   entry, so the result is [S, U?, vT?, terr?].
     * @param[in] keepdim maximum number of singular values to keep.
     * @param[in] err singular values < err are dropped (down to mindim).
     * @param[in] is_U whether U is present in tens and should be truncated.
     * @param[in] is_vT whether vT is present in tens and should be truncated.
     * @param[in] return_err selects the error tensor (terr) appended to tens; terr is real and has
     *   the same dtype as S:
     *   - 0: nothing is appended.
     *   - 1: a 1-element tensor holding the first (largest) discarded singular value.
     *   - 2: a tensor holding all discarded singular values (in descending order).
     *   If no truncation occurs there are no discarded values, so a 1-element zero tensor is
     *   appended (the truncation error is zero).
     * @param[in] mindim minimum number of singular values to keep.
     * @pre S is a non-empty real floating-point vector (float or double);
     */
    void memcpyTruncation(std::vector<Tensor> &tens, cytnx_uint64 keepdim, double err, bool is_U,
                          bool is_vT, unsigned int return_err, cytnx_uint64 mindim);

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_CPU_MEMCPYTRUNCATION_H_
