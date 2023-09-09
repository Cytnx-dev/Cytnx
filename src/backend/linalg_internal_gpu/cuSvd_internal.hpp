#ifndef __cuSvd_internal_H__
#define __cuSvd_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"
#include "cytnx_error.hpp"
#include "backend/lapack_wrapper.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// cuSvd
    void cuSvd_internal_cd(const boost::intrusive_ptr<Storage_base> &in,
                           boost::intrusive_ptr<Storage_base> &U,
                           boost::intrusive_ptr<Storage_base> &vT,
                           boost::intrusive_ptr<Storage_base> &s, const cytnx_int64 &M,
                           const cytnx_int64 &N);
    void cuSvd_internal_cf(const boost::intrusive_ptr<Storage_base> &in,
                           boost::intrusive_ptr<Storage_base> &U,
                           boost::intrusive_ptr<Storage_base> &vT,
                           boost::intrusive_ptr<Storage_base> &s, const cytnx_int64 &M,
                           const cytnx_int64 &N);
    void cuSvd_internal_d(const boost::intrusive_ptr<Storage_base> &in,
                          boost::intrusive_ptr<Storage_base> &U,
                          boost::intrusive_ptr<Storage_base> &vT,
                          boost::intrusive_ptr<Storage_base> &s, const cytnx_int64 &M,
                          const cytnx_int64 &N);
    void cuSvd_internal_f(const boost::intrusive_ptr<Storage_base> &in,
                          boost::intrusive_ptr<Storage_base> &U,
                          boost::intrusive_ptr<Storage_base> &vT,
                          boost::intrusive_ptr<Storage_base> &s, const cytnx_int64 &M,
                          const cytnx_int64 &N);

  }  // namespace linalg_internal
}  // namespace cytnx

#endif
