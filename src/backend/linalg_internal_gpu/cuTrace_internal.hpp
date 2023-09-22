#ifndef __cuTrace_internal_H__
#define __cuTrace_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"
#include "Tensor.hpp"

namespace cytnx {
  namespace linalg_internal {

    void cuTrace_internal_cd(const bool &is_2d, Tensor &out, const Tensor &Tn,
                             const cytnx_uint64 &Ndiag, const int &Nomp, const cytnx_uint64 &Nelem,
                             const std::vector<cytnx_uint64> &accu,
                             const std::vector<cytnx_uint64> &remain_rank_id,
                             const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                             const cytnx_uint64 &ax2);

    void cuTrace_internal_cf(const bool &is_2d, Tensor &out, const Tensor &Tn,
                             const cytnx_uint64 &Ndiag, const int &Nomp, const cytnx_uint64 &Nelem,
                             const std::vector<cytnx_uint64> &accu,
                             const std::vector<cytnx_uint64> &remain_rank_id,
                             const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                             const cytnx_uint64 &ax2);

    void cuTrace_internal_d(const bool &is_2d, Tensor &out, const Tensor &Tn,
                            const cytnx_uint64 &Ndiag, const int &Nomp,

                            const cytnx_uint64 &Nelem, const std::vector<cytnx_uint64> &accu,
                            const std::vector<cytnx_uint64> &remain_rank_id,
                            const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                            const cytnx_uint64 &ax2);

    void cuTrace_internal_f(const bool &is_2d, Tensor &out, const Tensor &Tn,
                            const cytnx_uint64 &Ndiag, const int &Nomp,

                            const cytnx_uint64 &Nelem, const std::vector<cytnx_uint64> &accu,
                            const std::vector<cytnx_uint64> &remain_rank_id,
                            const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                            const cytnx_uint64 &ax2);

    void cuTrace_internal_u64(const bool &is_2d, Tensor &out, const Tensor &Tn,
                              const cytnx_uint64 &Ndiag, const int &Nomp,

                              const cytnx_uint64 &Nelem, const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                              const cytnx_uint64 &ax2);

    void cuTrace_internal_i64(const bool &is_2d, Tensor &out, const Tensor &Tn,
                              const cytnx_uint64 &Ndiag, const int &Nomp,

                              const cytnx_uint64 &Nelem, const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                              const cytnx_uint64 &ax2);

    void cuTrace_internal_u32(const bool &is_2d, Tensor &out, const Tensor &Tn,
                              const cytnx_uint64 &Ndiag, const int &Nomp,

                              const cytnx_uint64 &Nelem, const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                              const cytnx_uint64 &ax2);

    void cuTrace_internal_i32(const bool &is_2d, Tensor &out, const Tensor &Tn,
                              const cytnx_uint64 &Ndiag, const int &Nomp,

                              const cytnx_uint64 &Nelem, const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                              const cytnx_uint64 &ax2);

    void cuTrace_internal_u16(const bool &is_2d, Tensor &out, const Tensor &Tn,
                              const cytnx_uint64 &Ndiag, const int &Nomp,

                              const cytnx_uint64 &Nelem, const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                              const cytnx_uint64 &ax2);

    void cuTrace_internal_i16(const bool &is_2d, Tensor &out, const Tensor &Tn,
                              const cytnx_uint64 &Ndiag, const int &Nomp,

                              const cytnx_uint64 &Nelem, const std::vector<cytnx_uint64> &accu,
                              const std::vector<cytnx_uint64> &remain_rank_id,
                              const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                              const cytnx_uint64 &ax2);

    void cuTrace_internal_b(const bool &is_2d, Tensor &out, const Tensor &Tn,
                            const cytnx_uint64 &Ndiag, const int &Nomp,

                            const cytnx_uint64 &Nelem, const std::vector<cytnx_uint64> &accu,
                            const std::vector<cytnx_uint64> &remain_rank_id,
                            const std::vector<cytnx_int64> &shape, const cytnx_uint64 &ax1,
                            const cytnx_uint64 &ax2);

  }  // namespace linalg_internal

}  // namespace cytnx

#endif
