#ifndef CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUPOW_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUPOW_INTERNAL_H_

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace linalg_internal {

    void cuPow_internal_d(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                          const cytnx_double &p);

    void cuPow_internal_f(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                          const cytnx_double &p);

    void cuPow_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                           const cytnx_double &p);

    void cuPow_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                           const cytnx_double &p);

  }  // namespace linalg_internal

}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUPOW_INTERNAL_H_
