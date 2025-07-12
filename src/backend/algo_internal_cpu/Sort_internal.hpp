#ifndef CYTNX_BACKEND_ALGO_INTERNAL_CPU_SORT_INTERNAL_H_
#define CYTNX_BACKEND_ALGO_INTERNAL_CPU_SORT_INTERNAL_H_

#include <algorithm>

#include <boost/intrusive_ptr.hpp>

#include "Type.hpp"
#include "backend/Storage.hpp"

namespace cytnx {

  namespace algo_internal {

    template <typename T>
    bool SortCompare(const T& a, const T& b) {
      if constexpr (is_complex_v<T>) {
        return (real(a) == real(b)) ? (imag(a) < imag(b)) : (real(a) < real(b));
      } else {
        return a < b;
      }
    }

    template <typename T>
    void SortInternalImpl(boost::intrusive_ptr<Storage_base>& out, const cytnx_uint64& stride,
                          const cytnx_uint64& Nelem) {
      T* p = reinterpret_cast<T*>(out->data());
      cytnx_uint64 Niter = Nelem / stride;
      for (cytnx_uint64 i = 0; i < Niter; i++)
        std::sort(p + i * stride, p + i * stride + stride, SortCompare<T>);
    }

    template <cytnx_bool>
    void SortInternalImpl(boost::intrusive_ptr<Storage_base>& out, const cytnx_uint64& stride,
                          const cytnx_uint64& Nelem) {
      cytnx_error_msg(true, "[ERROR] cytnx currently does not have bool type sort.%s", "\n");
    }

  }  // namespace algo_internal

}  // namespace cytnx

#endif  // CYTNX_BACKEND_ALGO_INTERNAL_CPU_SORT_INTERNAL_H_
