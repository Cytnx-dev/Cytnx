#ifndef __Matmul_internal_H__
#define __Matmul_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// Matmul
    void Matmul_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &inl,
                            const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int32 &Ml,
                            const cytnx_int32 &Comm, const cytnx_int32 &Nr);
    void Matmul_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &inl,
                            const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int32 &Ml,
                            const cytnx_int32 &Comm, const cytnx_int32 &Nr);
    void Matmul_internal_d(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &inl,
                           const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int32 &Ml,
                           const cytnx_int32 &Comm, const cytnx_int32 &Nr);
    void Matmul_internal_f(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &inl,
                           const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int32 &Ml,
                           const cytnx_int32 &Comm, const cytnx_int32 &Nr);
    void Matmul_internal_i64(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &inl,
                             const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int32 &Ml,
                             const cytnx_int32 &Comm, const cytnx_int32 &Nr);
    void Matmul_internal_u64(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &inl,
                             const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int32 &Ml,
                             const cytnx_int32 &Comm, const cytnx_int32 &Nr);
    void Matmul_internal_i32(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &inl,
                             const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int32 &Ml,
                             const cytnx_int32 &Comm, const cytnx_int32 &Nr);
    void Matmul_internal_u32(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &inl,
                             const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int32 &Ml,
                             const cytnx_int32 &Comm, const cytnx_int32 &Nr);
    void Matmul_internal_i16(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &inl,
                             const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int32 &Ml,
                             const cytnx_int32 &Comm, const cytnx_int32 &Nr);
    void Matmul_internal_u16(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &inl,
                             const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int32 &Ml,
                             const cytnx_int32 &Comm, const cytnx_int32 &Nr);
    void Matmul_internal_b(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &inl,
                           const boost::intrusive_ptr<Storage_base> &inr, const cytnx_int32 &Ml,
                           const cytnx_int32 &Comm, const cytnx_int32 &Nr);
  }  // namespace linalg_internal
}  // namespace cytnx

#endif
