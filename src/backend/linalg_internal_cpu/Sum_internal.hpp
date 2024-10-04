#ifndef BACKEND_LINALG_INTERNAL_CPU_SUM_INTERNAL_H_
#define BACKEND_LINALG_INTERNAL_CPU_SUM_INTERNAL_H_

#include "boost/smart_ptr/intrusive_ptr.hpp"

#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace linalg_internal {
    void Sum_internal_u64(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem);
    void Sum_internal_i64(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem);
    void Sum_internal_u32(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem);
    void Sum_internal_i32(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem);
    void Sum_internal_u16(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem);
    void Sum_internal_i16(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem);
    void Sum_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem);
    void Sum_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem);
    void Sum_internal_d(boost::intrusive_ptr<Storage_base> &out,
                        const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem);
    void Sum_internal_f(boost::intrusive_ptr<Storage_base> &out,
                        const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem);
    void Sum_internal_b(boost::intrusive_ptr<Storage_base> &out,
                        const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem);

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // BACKEND_LINALG_INTERNAL_CPU_SUM_INTERNAL_H_
