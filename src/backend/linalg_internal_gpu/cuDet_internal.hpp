#ifndef CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUDET_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUDET_INTERNAL_H_

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// cuDet
    void cuDet_internal_cd(void* out, const boost::intrusive_ptr<Storage_base>& in,
                           const cytnx_uint64& N);

    void cuDet_internal_cf(void* out, const boost::intrusive_ptr<Storage_base>& in,
                           const cytnx_uint64& N);
    void cuDet_internal_d(void* out, const boost::intrusive_ptr<Storage_base>& in,
                          const cytnx_uint64& N);
    void cuDet_internal_f(void* out, const boost::intrusive_ptr<Storage_base>& in,
                          const cytnx_uint64& N);

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUDET_INTERNAL_H_
