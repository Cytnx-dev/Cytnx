#ifndef CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUMOD_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUMOD_INTERNAL_H_

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// cuMod: element-wise modulo on the GPU, computed in the promoted type via
    /// cuda::std::complex-based kernels (mirrors cuMul_dispatch). Integral
    /// operands use %, floating operands use fmod; complex operands are
    /// rejected (modulo is undefined for complex).
    void cuMod_dispatch(boost::intrusive_ptr<Storage_base> &out,
                        boost::intrusive_ptr<Storage_base> &Lin,
                        boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                        const std::vector<cytnx_uint64> &shape,
                        const std::vector<cytnx_uint64> &invmapper_L,
                        const std::vector<cytnx_uint64> &invmapper_R);

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUMOD_INTERNAL_H_
