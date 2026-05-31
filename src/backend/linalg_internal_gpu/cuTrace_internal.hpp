#ifndef CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUTRACE_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUTRACE_INTERNAL_H_

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"
#include "Tensor.hpp"

namespace cytnx {
  namespace linalg_internal {

    void cuTrace_internal_cd(bool is_2d, Tensor &out, const Tensor &Tn, cytnx_uint64 Ndiag,
                             cytnx_uint64 Nelem, const std::vector<cytnx_uint64> &accu,
                             const std::vector<cytnx_uint64> &remain_rank_id,
                             const std::vector<cytnx_int64> &shape, cytnx_uint64 ax1,
                             cytnx_uint64 ax2);

    void cuTrace_internal_cf(bool is_2d, Tensor &out, const Tensor &Tn, cytnx_uint64 Ndiag,
                             cytnx_uint64 Nelem, const std::vector<cytnx_uint64> &accu,
                             const std::vector<cytnx_uint64> &remain_rank_id,
                             const std::vector<cytnx_int64> &shape, cytnx_uint64 ax1,
                             cytnx_uint64 ax2);

    void cuTrace_internal_d(bool is_2d, Tensor &out, const Tensor &Tn, cytnx_uint64 Ndiag,
                            cytnx_uint64 Nelem, const std::vector<cytnx_uint64> &accu,
                            const std::vector<cytnx_uint64> &remain_rank_id,
                            const std::vector<cytnx_int64> &shape, cytnx_uint64 ax1,
                            cytnx_uint64 ax2);

    void cuTrace_internal_f(bool is_2d, Tensor &out, const Tensor &Tn, cytnx_uint64 Ndiag,
                            cytnx_uint64 Nelem, const std::vector<cytnx_uint64> &accu,
                            const std::vector<cytnx_uint64> &remain_rank_id,
                            const std::vector<cytnx_int64> &shape, cytnx_uint64 ax1,
                            cytnx_uint64 ax2);

    void cuTrace_internal_u64(bool is_2d, Tensor &out, const Tensor &Tn, cytnx_uint64 Ndiag,
                              cytnx_uint64 Nelem, const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, cytnx_uint64 ax1,
                              cytnx_uint64 ax2);

    void cuTrace_internal_i64(bool is_2d, Tensor &out, const Tensor &Tn, cytnx_uint64 Ndiag,
                              cytnx_uint64 Nelem, const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, cytnx_uint64 ax1,
                              cytnx_uint64 ax2);

    void cuTrace_internal_u32(bool is_2d, Tensor &out, const Tensor &Tn, cytnx_uint64 Ndiag,
                              cytnx_uint64 Nelem, const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, cytnx_uint64 ax1,
                              cytnx_uint64 ax2);

    void cuTrace_internal_i32(bool is_2d, Tensor &out, const Tensor &Tn, cytnx_uint64 Ndiag,
                              cytnx_uint64 Nelem, const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, cytnx_uint64 ax1,
                              cytnx_uint64 ax2);

    void cuTrace_internal_u16(bool is_2d, Tensor &out, const Tensor &Tn, cytnx_uint64 Ndiag,
                              cytnx_uint64 Nelem, const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, cytnx_uint64 ax1,
                              cytnx_uint64 ax2);

    void cuTrace_internal_i16(bool is_2d, Tensor &out, const Tensor &Tn, cytnx_uint64 Ndiag,
                              cytnx_uint64 Nelem, const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, cytnx_uint64 ax1,
                              cytnx_uint64 ax2);

    void cuTrace_internal_b(bool is_2d, Tensor &out, const Tensor &Tn, cytnx_uint64 Ndiag,
                            cytnx_uint64 Nelem, const std::vector<cytnx_uint64> &accu,
                            const std::vector<cytnx_uint64> &remain_rank_id,
                            const std::vector<cytnx_int64> &shape, cytnx_uint64 ax1,
                            cytnx_uint64 ax2);

  }  // namespace linalg_internal

}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUTRACE_INTERNAL_H_
