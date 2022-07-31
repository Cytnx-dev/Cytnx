#ifndef __Det_internal_H__
#define __Det_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// Det
    void Det_internal_cd(void* out, const boost::intrusive_ptr<Storage_base>& in,
                         const cytnx_uint64& N);
    void Det_internal_cf(void* out, const boost::intrusive_ptr<Storage_base>& in,
                         const cytnx_uint64& N);
    void Det_internal_d(void* out, const boost::intrusive_ptr<Storage_base>& in,
                        const cytnx_uint64& N);
    void Det_internal_f(void* out, const boost::intrusive_ptr<Storage_base>& in,
                        const cytnx_uint64& N);

  }  // namespace linalg_internal
}  // namespace cytnx

#endif
