#ifndef CYTNX_BACKEND_LINALG_INTERNAL_CPU_BATCH_MATMUL_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_CPU_BATCH_MATMUL_INTERNAL_H_

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// Batch_matmul
    void Batch_matmul_internal_cd(std::vector<void *> &c_arrs, std::vector<void *> &a_arrs,
                                  std::vector<void *> &b_arrs, const Scalar &alpha,
                                  const Scalar &beta const std::vector<cytnx_int64> &ms,
                                  const std::vector<cytnx_int64> &ns,
                                  const std::vector<cytnx_int64> &ks,
                                  const std::vector<cytnx_int64> &grp_counts)

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_CPU_BATCH_MATMUL_INTERNAL_H_
