#ifndef __Sort_internal_H__
#define __Sort_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace algo_internal {

    void Sort_internal_cd(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                          const cytnx_uint64 &Nelem);

    void Sort_internal_cf(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                          const cytnx_uint64 &Nelem);

    void Sort_internal_d(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                         const cytnx_uint64 &Nelem);

    void Sort_internal_f(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                         const cytnx_uint64 &Nelem);

    void Sort_internal_u64(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                           const cytnx_uint64 &Nelem);

    void Sort_internal_i64(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                           const cytnx_uint64 &Nelem);

    void Sort_internal_u32(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                           const cytnx_uint64 &Nelem);

    void Sort_internal_i32(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                           const cytnx_uint64 &Nelem);

    void Sort_internal_u16(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                           const cytnx_uint64 &Nelem);

    void Sort_internal_i16(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                           const cytnx_uint64 &Nelem);

    void Sort_internal_b(boost::intrusive_ptr<Storage_base> &out, const cytnx_uint64 &stride,
                         const cytnx_uint64 &Nelem);

  }  // namespace algo_internal

}  // namespace cytnx

#endif
