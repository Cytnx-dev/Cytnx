#ifndef __Concate_internal_H__
#define __Concate_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace algo_internal {

    void vConcate_internal(char *out_ptr, std::vector<void *> &ins,
                           const std::vector<cytnx_uint64> &lens, const cytnx_uint64 &ElemSize);

    void hConcate_internal(char *out_ptr, std::vector<char *> &ins,
                           const std::vector<cytnx_uint64> &lens, const cytnx_uint64 &Dshare,
                           const cytnx_uint64 &Dtot, const cytnx_uint64 &ElemSize);

  }  // namespace algo_internal

}  // namespace cytnx

#endif
