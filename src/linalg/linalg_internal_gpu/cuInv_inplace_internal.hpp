#ifndef __cuInv_inplace_internal_H__
#define __cuInv_inplace_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "Storage.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace linalg_internal {

    void cuInv_inplace_internal_d(boost::intrusive_ptr<Storage_base> &ten,
                                  const cytnx_uint64 &Nelem, const double &clip);
    void cuInv_inplace_internal_f(boost::intrusive_ptr<Storage_base> &ten,
                                  const cytnx_uint64 &Nelem, const double &clip);
    void cuInv_inplace_internal_cd(boost::intrusive_ptr<Storage_base> &ten,
                                   const cytnx_uint64 &Nelem, const double &clip);
    void cuInv_inplace_internal_cf(boost::intrusive_ptr<Storage_base> &ten,
                                   const cytnx_uint64 &Nelem, const double &clip);

  }  // namespace linalg_internal

}  // namespace cytnx
#endif
