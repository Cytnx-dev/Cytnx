#ifndef __MaxMin_internal_H__
#define __MaxMin_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "Storage.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace linalg_internal {

    /// note type can be 'd' or 'f', indicating the return is casting as double or float
    void MaxMin_internal_u64(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten,
                             const cytnx_uint64 &Nelem, const char &type);
    void MaxMin_internal_i64(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten,
                             const cytnx_uint64 &Nelem, const char &type);
    void MaxMin_internal_u32(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten,
                             const cytnx_uint64 &Nelem, const char &type);
    void MaxMin_internal_i32(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten,
                             const cytnx_uint64 &Nelem, const char &type);
    void MaxMin_internal_u16(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten,
                             const cytnx_uint64 &Nelem, const char &type);
    void MaxMin_internal_i16(boost::intrusive_ptr<Storage_base> &out,
                             const boost::intrusive_ptr<Storage_base> &ten,
                             const cytnx_uint64 &Nelem, const char &type);
    void MaxMin_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &ten,
                            const cytnx_uint64 &Nelem, const char &type);
    void MaxMin_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                            const boost::intrusive_ptr<Storage_base> &ten,
                            const cytnx_uint64 &Nelem, const char &type);
    void MaxMin_internal_d(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                           const char &type);
    void MaxMin_internal_f(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                           const char &type);
    void MaxMin_internal_b(boost::intrusive_ptr<Storage_base> &out,
                           const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                           const char &type);

  }  // namespace linalg_internal

}  // namespace cytnx

#endif
