#ifndef CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUOUTER_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUOUTER_INTERNAL_H_

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// Typed GPU dispatch for Outer, replacing the legacy cuOuter_ii dtype-pair
    /// table (#1003). Outer.cpp promotes both operands to the output dtype
    /// before calling in, so out/Lin/Rin share out->dtype(); dispatch is on that
    /// single type (floating/complex -> cuBLAS GER, integer/bool -> custom
    /// kernel). j1, j2 are the two input extents.
    void cuOuter_dispatch(boost::intrusive_ptr<Storage_base> &out,
                          boost::intrusive_ptr<Storage_base> &Lin,
                          boost::intrusive_ptr<Storage_base> &Rin, const cytnx_uint64 &j1,
                          const cytnx_uint64 &j2);

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUOUTER_INTERNAL_H_
