#ifndef __cuNorm_internal_H__
#define __cuNorm_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "Storage.hpp"
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

#endif
