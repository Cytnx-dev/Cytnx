#ifndef __cuGemm_internal_H__
#define __cuGemm_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace linalg_internal {

    void cuGemm_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &inl,
                            const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                            const cytnx_int64 &Comm, const cytnx_int64 &Nr, const Scalar &a,
                            const Scalar &b);
    void cuGemm_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &inl,
                            const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                            const cytnx_int64 &Comm, const cytnx_int64 &Nr, const Scalar &a,
                            const Scalar &b);
    void cuGemm_internal_d(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &inl,
                           const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                           const cytnx_int64 &Comm, const cytnx_int64 &Nr, const Scalar &a,
                           const Scalar &b);
    void cuGemm_internal_f(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &inl,
                           const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int64 &Ml,
                           const cytnx_int64 &Comm, const cytnx_int64 &Nr, const Scalar &a,
                           const Scalar &b);

  }  // namespace linalg_internal
}  // namespace cytnx

#endif
