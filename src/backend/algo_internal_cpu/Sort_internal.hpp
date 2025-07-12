#ifndef CYTNX_BACKEND_ALGO_INTERNAL_CPU_SORT_INTERNAL_H_
#define CYTNX_BACKEND_ALGO_INTERNAL_CPU_SORT_INTERNAL_H_

#include <algorithm>
#include <cstdint>

#include <boost/intrusive_ptr.hpp>

#include "Type.hpp"
#include "backend/Storage.hpp"

// Macro to declare sorting functions for different data types.
#define CYTNX_SORT_INTERNAL_FUNC(Device, DType)                                                   \
  void Sort_internal_##DType(boost::intrusive_ptr<Storage_base>& out, const cytnx_uint64& stride, \
                             const cytnx_uint64& Nelem)

namespace cytnx {

  namespace algo_internal {

    CYTNX_SORT_INTERNAL_FUNC(/*cpu*/, ComplexDouble);
    CYTNX_SORT_INTERNAL_FUNC(/*cpu*/, ComplexFloat);
    CYTNX_SORT_INTERNAL_FUNC(/*cpu*/, Double);
    CYTNX_SORT_INTERNAL_FUNC(/*cpu*/, Float);
    CYTNX_SORT_INTERNAL_FUNC(/*cpu*/, Uint64);
    CYTNX_SORT_INTERNAL_FUNC(/*cpu*/, Int64);
    CYTNX_SORT_INTERNAL_FUNC(/*cpu*/, Uint32);
    CYTNX_SORT_INTERNAL_FUNC(/*cpu*/, Int32);
    CYTNX_SORT_INTERNAL_FUNC(/*cpu*/, Uint16);
    CYTNX_SORT_INTERNAL_FUNC(/*cpu*/, Int16);
    CYTNX_SORT_INTERNAL_FUNC(/*cpu*/, Bool);

  }  // namespace algo_internal

}  // namespace cytnx

#endif  // CYTNX_BACKEND_ALGO_INTERNAL_CPU_SORT_INTERNAL_H_
