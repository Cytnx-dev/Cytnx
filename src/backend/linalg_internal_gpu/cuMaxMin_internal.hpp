#ifndef CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUMAXMIN_INTERNAL_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUMAXMIN_INTERNAL_H_

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace linalg_internal {

    /// note type can be 'x' or 'n', indicating the return is max or min.

    void cuMaxMin_internal_u64(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &ten,
                               const cytnx_uint64 &Nelem, const char &type);
    void cuMaxMin_internal_i64(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &ten,
                               const cytnx_uint64 &Nelem, const char &type);
    void cuMaxMin_internal_u32(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &ten,
                               const cytnx_uint64 &Nelem, const char &type);
    void cuMaxMin_internal_i32(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &ten,
                               const cytnx_uint64 &Nelem, const char &type);
    void cuMaxMin_internal_u16(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &ten,
                               const cytnx_uint64 &Nelem, const char &type);
    void cuMaxMin_internal_i16(boost::intrusive_ptr<Storage_base> &out,
                               const boost::intrusive_ptr<Storage_base> &ten,
                               const cytnx_uint64 &Nelem, const char &type);

    void cuMaxMin_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                              const boost::intrusive_ptr<Storage_base> &ten,
                              const cytnx_uint64 &Nelem, const char &type);
    void cuMaxMin_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                              const boost::intrusive_ptr<Storage_base> &ten,
                              const cytnx_uint64 &Nelem, const char &type);

    void cuMaxMin_internal_d(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten,
                             const cytnx_uint64 &Nelem, const char &type);
    void cuMaxMin_internal_f(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten,
                             const cytnx_uint64 &Nelem, const char &type);
    void cuMaxMin_internal_b(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten,
                             const cytnx_uint64 &Nelem, const char &type);

  }  // namespace linalg_internal

}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUMAXMIN_INTERNAL_H_
