#ifndef __Concate_internal_H__
#define __Concate_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace algo_internal {

    void Concate_internal(char *out_ptr, std::vector<void*> &ins, const std::vector<cytnx_uint64> &lens, const cytnx_uint64 &ElemSize);
       


  }  // namespace algo_internal

}  // namespace cytnx

#endif
