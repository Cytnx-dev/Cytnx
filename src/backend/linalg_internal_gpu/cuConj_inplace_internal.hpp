#ifndef CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUCONJ_INPLACE_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUCONJ_INPLACE_INTERNAL_H_

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace linalg_internal {

    /// cuConj (in place): typed GPU dispatch (#1003). Complex dtypes only (real Conj is a no-op).
    void cuConj_inplace_dispatch(boost::intrusive_ptr<Storage_base> &inout, cytnx_uint64 Nelem);

    void cuConj_inplace_internal_cd(boost::intrusive_ptr<Storage_base> &ten,
                                    const cytnx_uint64 &Nelem);

    void cuConj_inplace_internal_cf(boost::intrusive_ptr<Storage_base> &ten,
                                    const cytnx_uint64 &Nelem);

  }  // namespace linalg_internal

}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUCONJ_INPLACE_INTERNAL_H_
