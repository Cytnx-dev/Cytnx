#ifndef __Diag_internal_H__
#define __Diag_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace linalg_internal {

    void Diag_internal_b(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                         const cytnx_bool &isrank2);
    void Diag_internal_i16(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                           const cytnx_bool &isrank2);

    void Diag_internal_u16(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                           const cytnx_bool &isrank2);

    void Diag_internal_i32(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                           const cytnx_bool &isrank2);

    void Diag_internal_u32(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                           const cytnx_bool &isrank2);

    void Diag_internal_i64(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                           const cytnx_bool &isrank2);

    void Diag_internal_u64(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                           const cytnx_bool &isrank2);

    void Diag_internal_d(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                         const cytnx_bool &isrank2);

    void Diag_internal_f(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                         const cytnx_bool &isrank2);

    void Diag_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                          const cytnx_bool &isrank2);

    void Diag_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                          const cytnx_bool &isrank2);

  }  // namespace linalg_internal

}  // namespace cytnx

#endif
