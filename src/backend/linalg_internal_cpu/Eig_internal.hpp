#ifndef CYTNX_BACKEND_LINALG_INTERNAL_CPU_EIG_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_CPU_EIG_INTERNAL_H_

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// Eig
    void Eig_internal_cd(const boost::intrusive_ptr<Storage_base> &in,
                         boost::intrusive_ptr<Storage_base> &e,
                         boost::intrusive_ptr<Storage_base> &v, const cytnx_int64 &L);
    void Eig_internal_cf(const boost::intrusive_ptr<Storage_base> &in,
                         boost::intrusive_ptr<Storage_base> &e,
                         boost::intrusive_ptr<Storage_base> &v, const cytnx_int64 &L);
    void Eig_internal_d(const boost::intrusive_ptr<Storage_base> &in,
                        boost::intrusive_ptr<Storage_base> &e,
                        boost::intrusive_ptr<Storage_base> &v, const cytnx_int64 &L);
    void Eig_internal_f(const boost::intrusive_ptr<Storage_base> &in,
                        boost::intrusive_ptr<Storage_base> &e,
                        boost::intrusive_ptr<Storage_base> &v, const cytnx_int64 &L);
  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_CPU_EIG_INTERNAL_H_
