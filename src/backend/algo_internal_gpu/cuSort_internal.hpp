#ifndef CYTNX_BACKEND_ALGO_INTERNAL_GPU_CUSORT_INTERNAL_H_
#define CYTNX_BACKEND_ALGO_INTERNAL_GPU_CUSORT_INTERNAL_H_

#include <cstdint>

#include <boost/intrusive_ptr.hpp>

#include "Type.hpp"
#include "backend/Storage.hpp"

// Macro to declare sorting functions for different data types.
#define CYTNX_SORT_INTERNAL_FUNC(Device, DType)                               \
  void Device##Sort_internal_##DType(boost::intrusive_ptr<Storage_base>& out, \
                                     const cytnx_uint64& stride, const cytnx_uint64& Nelem)

namespace cytnx {

  namespace algo_internal {

    CYTNX_SORT_INTERNAL_FUNC(cuda, ComplexDouble);
    CYTNX_SORT_INTERNAL_FUNC(cuda, ComplexFloat);
    CYTNX_SORT_INTERNAL_FUNC(cuda, Double);
    CYTNX_SORT_INTERNAL_FUNC(cuda, Float);
    CYTNX_SORT_INTERNAL_FUNC(cuda, Uint64);
    CYTNX_SORT_INTERNAL_FUNC(cuda, Int64);
    CYTNX_SORT_INTERNAL_FUNC(cuda, Uint32);
    CYTNX_SORT_INTERNAL_FUNC(cuda, Int32);
    CYTNX_SORT_INTERNAL_FUNC(cuda, Uint16);
    CYTNX_SORT_INTERNAL_FUNC(cuda, Int16);
    CYTNX_SORT_INTERNAL_FUNC(cuda, Bool);

  }  // namespace algo_internal

}  // namespace cytnx

#endif  // CYTNX_BACKEND_ALGO_INTERNAL_GPU_CUSORT_INTERNAL_H_
