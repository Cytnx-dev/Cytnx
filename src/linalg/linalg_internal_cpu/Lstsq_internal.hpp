#ifndef __Lstsq_internal_H__
#define __Lstsq_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace linalg_internal {
    /// Lstsq
    void Lstsq_internal_d(boost::intrusive_ptr<Storage_base> &in,
                          boost::intrusive_ptr<Storage_base> &b,
                          boost::intrusive_ptr<Storage_base> &s,
                          boost::intrusive_ptr<Storage_base> &r, const cytnx_int64 &M,
                          const cytnx_int64 &N, const cytnx_int64 &nrhs, const cytnx_float &rcond);
    void Lstsq_internal_f(boost::intrusive_ptr<Storage_base> &in,
                          boost::intrusive_ptr<Storage_base> &b,
                          boost::intrusive_ptr<Storage_base> &s,
                          boost::intrusive_ptr<Storage_base> &r, const cytnx_int64 &M,
                          const cytnx_int64 &N, const cytnx_int64 &nrhs, const cytnx_float &rcond);
    void Lstsq_internal_cd(boost::intrusive_ptr<Storage_base> &in,
                           boost::intrusive_ptr<Storage_base> &b,
                           boost::intrusive_ptr<Storage_base> &s,
                           boost::intrusive_ptr<Storage_base> &r, const cytnx_int64 &M,
                           const cytnx_int64 &N, const cytnx_int64 &nrhs, const cytnx_float &rcond);
    void Lstsq_internal_cf(boost::intrusive_ptr<Storage_base> &in,
                           boost::intrusive_ptr<Storage_base> &b,
                           boost::intrusive_ptr<Storage_base> &s,
                           boost::intrusive_ptr<Storage_base> &r, const cytnx_int64 &M,
                           const cytnx_int64 &N, const cytnx_int64 &nrhs, const cytnx_float &rcond);
  }  // namespace linalg_internal
}  // namespace cytnx

#endif
