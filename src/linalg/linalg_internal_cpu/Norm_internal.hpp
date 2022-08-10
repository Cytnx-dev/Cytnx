#ifndef __Norm_internal_H__
#define __Norm_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "Storage.hpp"
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

#endif
