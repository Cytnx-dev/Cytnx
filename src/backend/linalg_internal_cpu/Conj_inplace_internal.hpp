#ifndef CYTNX_BACKEND_LINALG_INTERNAL_CPU_CONJ_INPLACE_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_CPU_CONJ_INPLACE_INTERNAL_H_

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace linalg_internal {

    void Conj_inplace_internal_cf(boost::intrusive_ptr<Storage_base> &ten,
                                  const cytnx_uint64 &Nelem);
    void Conj_inplace_internal_cd(boost::intrusive_ptr<Storage_base> &ten,
                                  const cytnx_uint64 &Nelem);

  }  // namespace linalg_internal

}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_CPU_CONJ_INPLACE_INTERNAL_H_
