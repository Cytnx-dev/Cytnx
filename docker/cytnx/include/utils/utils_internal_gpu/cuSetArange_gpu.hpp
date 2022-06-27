#ifndef _H_cuSetArange_gpu_
#define _H_cuSetArange_gpu_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "cytnx_error.hpp"
#include "Storage.hpp"

namespace cytnx {
  namespace utils_internal {

    // type = 0, start < end , incremental
    // type = 1, start > end , decremental
    void cuSetArange_gpu_cd(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                            const cytnx_double &end, const cytnx_double &step,
                            const cytnx_uint64 &Nelem);
    void cuSetArange_gpu_cf(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                            const cytnx_double &end, const cytnx_double &step,
                            const cytnx_uint64 &Nelem);
    void cuSetArange_gpu_d(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                           const cytnx_double &end, const cytnx_double &step,
                           const cytnx_uint64 &Nelem);
    void cuSetArange_gpu_f(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                           const cytnx_double &end, const cytnx_double &step,
                           const cytnx_uint64 &Nelem);
    void cuSetArange_gpu_i64(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                             const cytnx_double &end, const cytnx_double &step,
                             const cytnx_uint64 &Nelem);
    void cuSetArange_gpu_u64(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                             const cytnx_double &end, const cytnx_double &step,
                             const cytnx_uint64 &Nelem);
    void cuSetArange_gpu_i32(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                             const cytnx_double &end, const cytnx_double &step,
                             const cytnx_uint64 &Nelem);
    void cuSetArange_gpu_u32(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                             const cytnx_double &end, const cytnx_double &step,
                             const cytnx_uint64 &Nelem);
    void cuSetArange_gpu_i16(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                             const cytnx_double &end, const cytnx_double &step,
                             const cytnx_uint64 &Nelem);
    void cuSetArange_gpu_u16(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                             const cytnx_double &end, const cytnx_double &step,
                             const cytnx_uint64 &Nelem);
    void cuSetArange_gpu_b(boost::intrusive_ptr<Storage_base> &in, const cytnx_double &start,
                           const cytnx_double &end, const cytnx_double &step,
                           const cytnx_uint64 &Nelem);
  }  // namespace utils_internal
}  // namespace cytnx
#endif
