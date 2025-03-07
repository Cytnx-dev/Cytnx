#ifndef CYTNX_BACKEND_LINALG_INTERNAL_CPU_NORM_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_CPU_NORM_INTERNAL_H_

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// Norm
    void Norm_internal_cd(void* out, const boost::intrusive_ptr<Storage_base>& in);
    void Norm_internal_cf(void* out, const boost::intrusive_ptr<Storage_base>& in);
    void Norm_internal_d(void* out, const boost::intrusive_ptr<Storage_base>& in);
    void Norm_internal_f(void* out, const boost::intrusive_ptr<Storage_base>& in);

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_CPU_NORM_INTERNAL_H_
