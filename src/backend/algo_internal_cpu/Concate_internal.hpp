#ifndef CYTNX_BACKEND_ALGO_INTERNAL_CPU_CONCATE_INTERNAL_H_
#define CYTNX_BACKEND_ALGO_INTERNAL_CPU_CONCATE_INTERNAL_H_

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace algo_internal {

    void vConcate_internal(char* out_ptr, std::vector<void*>& ins,
                           const std::vector<cytnx_uint64>& lens, const cytnx_uint64& ElemSize);

    void hConcate_internal(char* out_ptr, std::vector<char*>& ins,
                           const std::vector<cytnx_uint64>& lens, const cytnx_uint64& Dshare,
                           const cytnx_uint64& Dtot, const cytnx_uint64& ElemSize);

  }  // namespace algo_internal

}  // namespace cytnx

#endif  // CYTNX_BACKEND_ALGO_INTERNAL_CPU_CONCATE_INTERNAL_H_
