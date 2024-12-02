#ifndef CYTNX_BACKEND_ALGO_INTERNAL_GPU_CUSORT_INTERNAL_H_
#define CYTNX_BACKEND_ALGO_INTERNAL_GPU_CUSORT_INTERNAL_H_

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace algo_internal {

    void cuSort_internal_cd(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                            const cytnx_uint64 &Nelem);

    void cuSort_internal_cf(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                            const cytnx_uint64 &Nelem);

    void cuSort_internal_d(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                           const cytnx_uint64 &Nelem);

    void cuSort_internal_f(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                           const cytnx_uint64 &Nelem);

    void cuSort_internal_u64(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                             const cytnx_uint64 &Nelem);

    void cuSort_internal_i64(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                             const cytnx_uint64 &Nelem);

    void cuSort_internal_u32(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                             const cytnx_uint64 &Nelem);

    void cuSort_internal_i32(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                             const cytnx_uint64 &Nelem);

    void cuSort_internal_u16(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                             const cytnx_uint64 &Nelem);

    void cuSort_internal_i16(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                             const cytnx_uint64 &Nelem);

    void cuSort_internal_b(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                           const cytnx_uint64 &Nelem);

  }  // namespace algo_internal

}  // namespace cytnx

#endif  // CYTNX_BACKEND_ALGO_INTERNAL_GPU_CUSORT_INTERNAL_H_
