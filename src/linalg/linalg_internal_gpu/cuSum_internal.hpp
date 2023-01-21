#ifndef __cuSum_internal_H__
#define __cuSum_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// cuSum
    void cuSum_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                          const char &type);

  }  // namespace linalg_internal
}  // namespace cytnx

#endif
