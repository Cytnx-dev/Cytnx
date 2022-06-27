#ifndef __Inv_inplace_internal_H__
#define __Inv_inplace_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace linalg_internal {

    void Inv_inplace_internal_d(boost::intrusive_ptr<Storage_base> &iten, const cytnx_int32 &L);
    void Inv_inplace_internal_f(boost::intrusive_ptr<Storage_base> &iten, const cytnx_int32 &L);
    void Inv_inplace_internal_cd(boost::intrusive_ptr<Storage_base> &iten, const cytnx_int32 &L);
    void Inv_inplace_internal_cf(boost::intrusive_ptr<Storage_base> &iten, const cytnx_int32 &L);

  }  // namespace linalg_internal

}  // namespace cytnx

#endif
