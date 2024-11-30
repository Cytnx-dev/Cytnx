#ifndef CYTNX_BACKEND_ALGO_INTERNAL_GPU_CUSPLIT_INTERNAL_H_
#define CYTNX_BACKEND_ALGO_INTERNAL_GPU_CUSPLIT_INTERNAL_H_

#include "backend/algo_internal_interface.hpp"
#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace algo_internal {

    void cuvSplit_internal(std::vector<char *> &targ_ptrs, const char *In_ptr,
                           const std::vector<cytnx_uint64> &dims, const cytnx_uint64 &Col_dim,
                           const cytnx_uint64 &ElemSize);

    void cuhSplit_internal(std::vector<char *> &targ_ptrs, const char *In_ptr,
                           const std::vector<cytnx_uint64> &dims, const cytnx_uint64 &Row_dim,
                           const cytnx_uint64 &ElemSize);

  }  // namespace algo_internal

}  // namespace cytnx

#endif  // CYTNX_BACKEND_ALGO_INTERNAL_GPU_CUSPLIT_INTERNAL_H_
