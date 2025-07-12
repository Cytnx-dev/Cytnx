#include "Sort_internal.hpp"

#include <algorithm>
#include <iostream>

#include "backend/algo_internal_interface.hpp"

// Macro to generate complex number sorting functions.
// Creates a comparison function and sort implementation for complex types.
#define CYTNX_INTERNAL_COMPLEX_SORT(Device, DType, CytnxDType)                                \
  bool Device##_compare_##DType(CytnxDType a, CytnxDType b) {                                 \
    if (real(a) == real(b)) return imag(a) < imag(b);                                         \
    return real(a) < real(b);                                                                 \
  }                                                                                           \
  void Device##Sort_internal_##DType(boost::intrusive_ptr<Storage_base>& out,                 \
                                     const cytnx_uint64& stride, const cytnx_uint64& Nelem) { \
    auto* p = reinterpret_cast<CytnxDType*>(out->data());                                     \
    cytnx_uint64 num_iterations = Nelem / stride;                                             \
    for (cytnx_uint64 i = 0; i < num_iterations; ++i) {                                       \
      std::sort(p + i * stride, p + i * stride + stride, Device##_compare_##DType);           \
    }                                                                                         \
  }

// Macro to generate standard sorting functions.
#define CYTNX_INTERNAL_SORT(Device, DType, CytnxDType)                                        \
  void Device##Sort_internal_##DType(boost::intrusive_ptr<Storage_base>& out,                 \
                                     const cytnx_uint64& stride, const cytnx_uint64& Nelem) { \
    auto* p = reinterpret_cast<CytnxDType*>(out->data());                                     \
    cytnx_uint64 num_iterations = Nelem / stride;                                             \
    for (cytnx_uint64 i = 0; i < num_iterations; ++i) {                                       \
      std::sort(p + i * stride, p + i * stride + stride);                                     \
    }                                                                                         \
  }

namespace cytnx {

  namespace algo_internal {

    // Generate sorting functions for all supported data types.
    CYTNX_INTERNAL_COMPLEX_SORT(/*cpu*/, ComplexDouble, cytnx_complex128)
    CYTNX_INTERNAL_COMPLEX_SORT(/*cpu*/, ComplexFloat, cytnx_complex64)
    CYTNX_INTERNAL_SORT(/*cpu*/, Double, double)
    CYTNX_INTERNAL_SORT(/*cpu*/, Float, float)
    CYTNX_INTERNAL_SORT(/*cpu*/, Uint64, cytnx_uint64)
    CYTNX_INTERNAL_SORT(/*cpu*/, Int64, cytnx_int64)
    CYTNX_INTERNAL_SORT(/*cpu*/, Uint32, cytnx_uint32)
    CYTNX_INTERNAL_SORT(/*cpu*/, Int32, cytnx_int32)
    CYTNX_INTERNAL_SORT(/*cpu*/, Uint16, cytnx_uint16)
    CYTNX_INTERNAL_SORT(/*cpu*/, Int16, cytnx_int16)

    void Sort_internal_Bool(boost::intrusive_ptr<Storage_base>& out, const cytnx_uint64& stride,
                            const cytnx_uint64& Nelem) {
      /*
      cytnx_bool *p = (cytnx_bool*)out->Mem;
      cytnx_uint64 Niter = Nelem/stride;
      for(cytnx_uint64 i=0;i<Niter;i++)
          std::sort(p+i*stride,p+i*stride+stride);
      */
      cytnx_error_msg(true, "[ERROR] cytnx currently does not have bool type sort.%s", "\n");
    }

  }  // namespace algo_internal

}  // namespace cytnx
