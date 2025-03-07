#ifndef CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUEXP_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUEXP_INTERNAL_H_

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace linalg_internal {

    void cuExp_internal_d(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem);

    void cuExp_internal_f(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem);

    void cuExp_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten,
                           const cytnx_uint64 &Nelem);

    void cuExp_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten,
                           const cytnx_uint64 &Nelem);

  }  // namespace linalg_internal

}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUEXP_INTERNAL_H_
