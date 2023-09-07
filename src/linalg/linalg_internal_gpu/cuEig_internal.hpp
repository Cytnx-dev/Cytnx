#ifndef __cuEig_internal_H__
#define __cuEig_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// cuEig
    void cuEig_internal_cd(const boost::intrusive_ptr<Storage_base> &in,
                           boost::intrusive_ptr<Storage_base> &e,
                           boost::intrusive_ptr<Storage_base> &v, const cytnx_int64 &L);
    void cuEig_internal_cf(const boost::intrusive_ptr<Storage_base> &in,
                           boost::intrusive_ptr<Storage_base> &e,
                           boost::intrusive_ptr<Storage_base> &v, const cytnx_int64 &L);
    void cuEig_internal_d(const boost::intrusive_ptr<Storage_base> &in,
                          boost::intrusive_ptr<Storage_base> &e,
                          boost::intrusive_ptr<Storage_base> &v, const cytnx_int64 &L);
    void cuEig_internal_f(const boost::intrusive_ptr<Storage_base> &in,
                          boost::intrusive_ptr<Storage_base> &e,
                          boost::intrusive_ptr<Storage_base> &v, const cytnx_int64 &L);
  }  // namespace linalg_internal
}  // namespace cytnx

#endif
