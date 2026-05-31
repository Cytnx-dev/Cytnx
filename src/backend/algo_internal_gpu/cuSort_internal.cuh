#ifndef CYTNX_BACKEND_ALGO_INTERNAL_GPU_CUSORT_INTERNAL_H_
#define CYTNX_BACKEND_ALGO_INTERNAL_GPU_CUSORT_INTERNAL_H_

#include <boost/intrusive_ptr.hpp>
#include <thrust/sort.h>

#include "Type.hpp"
#include "backend/Storage.hpp"

namespace cytnx {

  namespace algo_internal {

    // handle float2, double2
    template <typename T>
    concept Vec2Like = requires(T t) {
      t.x;
      t.y;
    };
    template <typename T>
    __host__ __device__ bool cuSortCompare(const T& a, const T& b) {
      if constexpr (is_complex_v<T>) {
        return (real(a) == real(b)) ? (imag(a) < imag(b)) : (real(a) < real(b));
      } else if constexpr (Vec2Like<T>) {
        return (a.x == b.x) ? (a.y < b.y) : (a.x < b.x);
      } else {
        return a < b;
      }
    }

    template <typename T>
    void cuSortInternalImpl(boost::intrusive_ptr<Storage_base>& out, const cytnx_uint64& stride,
                            const cytnx_uint64& Nelem) {
      T* p = reinterpret_cast<T*>(out->data());
      cytnx_uint64 Niter = Nelem / stride;
      for (cytnx_uint64 i = 0; i < Niter; i++)
        thrust::sort(p + i * stride, p + i * stride + stride, cuSortCompare<T>);
    }

    template <cytnx_bool>
    void cuSortInternalImpl(boost::intrusive_ptr<Storage_base>& out, const cytnx_uint64& stride,
                            const cytnx_uint64& Nelem) {
      cytnx_error_msg(true, "[ERROR] cytnx currently does not have bool type sort.%s", "\n");
    }

  }  // namespace algo_internal

}  // namespace cytnx

#endif  // CYTNX_BACKEND_ALGO_INTERNAL_GPU_CUSORT_INTERNAL_H_
