#ifndef __Conj_inplace_internal_H__
#define __Conj_inplace_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "Storage.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace linalg_internal {

    void Conj_inplace_internal_cf(boost::intrusive_ptr<Storage_base> &ten,
                                  const cytnx_uint64 &Nelem);
    void Conj_inplace_internal_cd(boost::intrusive_ptr<Storage_base> &ten,
                                  const cytnx_uint64 &Nelem);

  }  // namespace linalg_internal

}  // namespace cytnx

#endif
