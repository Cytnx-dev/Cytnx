#ifndef CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUTENSORDOT_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUTENSORDOT_INTERNAL_H_

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"
#include "Tensor.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// cuTensordot
    void cuTensordot_internal_cd(Tensor &out, const Tensor &Lin, const Tensor &Rin,
                                 const std::vector<cytnx_uint64> &idxl,
                                 const std::vector<cytnx_uint64> &idxr);

    void cuTensordot_internal_cf(Tensor &out, const Tensor &Lin, const Tensor &Rin,
                                 const std::vector<cytnx_uint64> &idxl,
                                 const std::vector<cytnx_uint64> &idxr);

    void cuTensordot_internal_d(Tensor &out, const Tensor &Lin, const Tensor &Rin,
                                const std::vector<cytnx_uint64> &idxl,
                                const std::vector<cytnx_uint64> &idxr);

    void cuTensordot_internal_f(Tensor &out, const Tensor &Lin, const Tensor &Rin,
                                const std::vector<cytnx_uint64> &idxl,
                                const std::vector<cytnx_uint64> &idxr);

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUTENSORDOT_INTERNAL_H_
