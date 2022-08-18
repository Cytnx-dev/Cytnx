#ifndef __cuDiag_internal_H__
#define __cuDiag_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "Storage.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace linalg_internal {

    void cuDiag_internal_b(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                           const cytnx_bool &isrank2);
    void cuDiag_internal_i16(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                             const cytnx_bool &isrank2);

    void cuDiag_internal_u16(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                             const cytnx_bool &isrank2);

    void cuDiag_internal_i32(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                             const cytnx_bool &isrank2);

    void cuDiag_internal_u32(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                             const cytnx_bool &isrank2);

    void cuDiag_internal_i64(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                             const cytnx_bool &isrank2);

    void cuDiag_internal_u64(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                             const cytnx_bool &isrank2);

    void cuDiag_internal_d(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                           const cytnx_bool &isrank2);

    void cuDiag_internal_f(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                           const cytnx_bool &isrank2);

    void cuDiag_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                            const cytnx_bool &isrank2);

    void cuDiag_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &L,
                            const cytnx_bool &isrank2);

  }  // namespace linalg_internal

}  // namespace cytnx

#endif
