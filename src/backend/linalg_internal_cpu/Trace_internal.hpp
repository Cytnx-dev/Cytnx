#ifndef CYTNX_BACKEND_LINALG_INTERNAL_CPU_TRACE_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_CPU_TRACE_INTERNAL_H_

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"
#include "Tensor.hpp"

namespace cytnx {
  namespace linalg_internal {

    Tensor Trace_internal_cd(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2);
    Tensor Trace_internal_cf(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2);
    Tensor Trace_internal_d(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2);
    Tensor Trace_internal_f(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2);
    Tensor Trace_internal_u64(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2);
    Tensor Trace_internal_i64(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2);
    Tensor Trace_internal_u32(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2);
    Tensor Trace_internal_i32(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2);
    Tensor Trace_internal_u16(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2);
    Tensor Trace_internal_i16(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2);
    Tensor Trace_internal_b(const Tensor &Tn, cytnx_uint64 ax1, cytnx_uint64 ax2);

  }  // namespace linalg_internal

}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_CPU_TRACE_INTERNAL_H_
