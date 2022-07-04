#ifndef _H_SetArange_cpu_
#define _H_SetArange_cpu_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "cytnx_error.hpp"
#include "Storage.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace utils_internal {

    // type = 0, start < end , incremental
    // type = 1, start > end , decremental
    void SetArange_cpu_cd(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                          const cytnx_double &end, const cytnx_double &step,
                          const cytnx_uint64 &Nelem);
    void SetArange_cpu_cf(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                          const cytnx_double &end, const cytnx_double &step,
                          const cytnx_uint64 &Nelem);
    void SetArange_cpu_d(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                         const cytnx_double &end, const cytnx_double &step,
                         const cytnx_uint64 &Nelem);
    void SetArange_cpu_f(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                         const cytnx_double &end, const cytnx_double &step,
                         const cytnx_uint64 &Nelem);
    void SetArange_cpu_i64(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                           const cytnx_double &end, const cytnx_double &step,
                           const cytnx_uint64 &Nelem);
    void SetArange_cpu_u64(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                           const cytnx_double &end, const cytnx_double &step,
                           const cytnx_uint64 &Nelem);
    void SetArange_cpu_i32(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                           const cytnx_double &end, const cytnx_double &step,
                           const cytnx_uint64 &Nelem);
    void SetArange_cpu_u32(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                           const cytnx_double &end, const cytnx_double &step,
                           const cytnx_uint64 &Nelem);
    void SetArange_cpu_i16(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                           const cytnx_double &end, const cytnx_double &step,
                           const cytnx_uint64 &Nelem);
    void SetArange_cpu_u16(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                           const cytnx_double &end, const cytnx_double &step,
                           const cytnx_uint64 &Nelem);
    void SetArange_cpu_b(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                         const cytnx_double &end, const cytnx_double &step,
                         const cytnx_uint64 &Nelem);
  }  // namespace utils_internal
}  // namespace cytnx
#endif
