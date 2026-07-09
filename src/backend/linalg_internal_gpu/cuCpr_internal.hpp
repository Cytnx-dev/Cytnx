#ifndef CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUCPR_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUCPR_INTERNAL_H_

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// cuCpr: element-wise comparison on the GPU, computing the result in the
    /// promoted comparison type via cuda::std::complex-based kernels (mirrors
    /// cuMul_dispatch). Output is always Bool.
    void cuCpr_dispatch(boost::intrusive_ptr<Storage_base> &out,
                        boost::intrusive_ptr<Storage_base> &Lin,
                        boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                        const std::vector<cytnx_uint64> &shape,
                        const std::vector<cytnx_uint64> &invmapper_L,
                        const std::vector<cytnx_uint64> &invmapper_R);

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUCPR_INTERNAL_H_
