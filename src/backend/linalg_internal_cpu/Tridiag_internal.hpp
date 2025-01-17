#ifndef CYTNX_BACKEND_LINALG_INTERNAL_CPU_TRIDIAG_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_CPU_TRIDIAG_INTERNAL_H_

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// Tridiag
    void Tridiag_internal_d(const boost::intrusive_ptr<Storage_base> &diag,
                            const boost::intrusive_ptr<Storage_base> &s_diag,
                            boost::intrusive_ptr<Storage_base> &S,
                            boost::intrusive_ptr<Storage_base> &U, const cytnx_int64 &L,
                            bool throw_excp = false);
    void Tridiag_internal_f(const boost::intrusive_ptr<Storage_base> &diag,
                            const boost::intrusive_ptr<Storage_base> &s_diag,
                            boost::intrusive_ptr<Storage_base> &S,
                            boost::intrusive_ptr<Storage_base> &U, const cytnx_int64 &L,
                            bool throw_excp = false);

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_CPU_TRIDIAG_INTERNAL_H_
