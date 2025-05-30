#ifndef CYTNX_BACKEND_LINALG_INTERNAL_CPU_INV_INPLACE_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_CPU_INV_INPLACE_INTERNAL_H_

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace linalg_internal {

    /// This is a inplace function, with only floating type support.
    void Inv_inplace_internal_d(boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                                const double &clip);

    void Inv_inplace_internal_f(boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                                const double &clip);

    void Inv_inplace_internal_cd(boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                                 const double &clip);

    void Inv_inplace_internal_cf(boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                                 const double &clip);

  }  // namespace linalg_internal

}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_CPU_INV_INPLACE_INTERNAL_H_
