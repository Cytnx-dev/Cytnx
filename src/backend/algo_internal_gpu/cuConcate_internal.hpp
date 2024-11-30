#ifndef CYTNX_BACKEND_ALGO_INTERNAL_GPU_CUCONCATE_INTERNAL_H_
#define CYTNX_BACKEND_ALGO_INTERNAL_GPU_CUCONCATE_INTERNAL_H_

#include "backend/algo_internal_interface.hpp"
#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace algo_internal {

    void cuvConcate_internal(char *out_ptr, std::vector<void *> &ins,
                             const std::vector<cytnx_uint64> &lens, const cytnx_uint64 &ElemSize);

    void cuhConcate_internal(char *out_ptr, std::vector<char *> &ins,
                             const std::vector<cytnx_uint64> &lens, const cytnx_uint64 &Dshare,
                             const cytnx_uint64 &Dtot, const cytnx_uint64 &ElemSize);

  }  // namespace algo_internal

}  // namespace cytnx

#endif  // CYTNX_BACKEND_ALGO_INTERNAL_GPU_CUCONCATE_INTERNAL_H_
