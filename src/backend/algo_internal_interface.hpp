#ifndef CYTNX_BACKEND_ALGO_INTERNAL_INTERFACE_H_
#define CYTNX_BACKEND_ALGO_INTERNAL_INTERFACE_H_

#include <iostream>
#include <vector>

#include "Type.hpp"
#include "backend/Storage.hpp"
#include "backend/algo_internal_cpu/Sort_internal.hpp"
#include "backend/algo_internal_cpu/Concate_internal.hpp"
#include "backend/algo_internal_cpu/Split_internal.hpp"

#ifdef UNI_GPU
  #include "backend/algo_internal_gpu/cuSort_internal.cuh"
  #include "backend/algo_internal_gpu/cuConcate_internal.hpp"
  #include "backend/algo_internal_gpu/cuSplit_internal.hpp"
#endif

namespace cytnx {

  namespace algo_internal {
    class algo_internal_interface {
     public:
      algo_internal_interface();
    };
    extern algo_internal_interface aii;
  }  // namespace algo_internal

}  // namespace cytnx

#endif  // CYTNX_BACKEND_ALGO_INTERNAL_INTERFACE_H_
