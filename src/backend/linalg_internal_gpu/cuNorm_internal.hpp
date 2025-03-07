#ifndef CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUNORM_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUNORM_INTERNAL_H_

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// cuNorm
    void cuNorm_internal_cd(void* out, const boost::intrusive_ptr<Storage_base>& in);
    void cuNorm_internal_cf(void* out, const boost::intrusive_ptr<Storage_base>& in);
    void cuNorm_internal_d(void* out, const boost::intrusive_ptr<Storage_base>& in);
    void cuNorm_internal_f(void* out, const boost::intrusive_ptr<Storage_base>& in);

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUNORM_INTERNAL_H_
