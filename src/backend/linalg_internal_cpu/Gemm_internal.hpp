#ifndef CYTNX_BACKEND_LINALG_INTERNAL_CPU_GEMM_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_CPU_GEMM_INTERNAL_H_

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// Gemm
    void Gemm_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &inl,
                          const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                          const cytnx_int64 &Comm, const cytnx_int64 &Nr, const Scalar &a,
                          const Scalar &b);
    void Gemm_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &inl,
                          const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                          const cytnx_int64 &Comm, const cytnx_int64 &Nr, const Scalar &a,
                          const Scalar &b);
    void Gemm_internal_d(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &inl,
                         const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                         const cytnx_int64 &Comm, const cytnx_int64 &Nr, const Scalar &a,
                         const Scalar &b);
    void Gemm_internal_f(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &inl,
                         const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                         const cytnx_int64 &Comm, const cytnx_int64 &Nr, const Scalar &a,
                         const Scalar &b);

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_CPU_GEMM_INTERNAL_H_
