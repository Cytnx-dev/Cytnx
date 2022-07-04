#ifndef _H_cuCast_gpu_
#define _H_cuCast_gpu_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "Storage.hpp"
#include "cytnx_error.hpp"

namespace cytnx {
  namespace utils_internal {

    void cuCast_gpu_cdtcd(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_cdtcf(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);

    void cuCast_gpu_cftcd(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_cftcf(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);

    void cuCast_gpu_dtcd(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& alloc_device);
    void cuCast_gpu_dtcf(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& alloc_device);
    void cuCast_gpu_dtd(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& alloc_device);
    void cuCast_gpu_dtf(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& alloc_device);
    void cuCast_gpu_dti64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_dtu64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_dti32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_dtu32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_dti16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_dtu16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_dtb(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& alloc_device);

    void cuCast_gpu_ftcd(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& alloc_device);
    void cuCast_gpu_ftcf(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& alloc_device);
    void cuCast_gpu_ftd(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& alloc_device);
    void cuCast_gpu_ftf(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& alloc_device);
    void cuCast_gpu_fti64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_ftu64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_fti32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_ftu32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_fti16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_ftu16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_ftb(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& alloc_device);

    void cuCast_gpu_i64tcd(const boost::intrusive_ptr<Storage_base>& in,
                           boost::intrusive_ptr<Storage_base>& out,
                           const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_i64tcf(const boost::intrusive_ptr<Storage_base>& in,
                           boost::intrusive_ptr<Storage_base>& out,
                           const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_i64td(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_i64tf(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_i64ti64(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_i64tu64(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_i64ti32(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_i64tu32(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_i64ti16(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_i64tu16(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_i64tb(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);

    void cuCast_gpu_u64tcd(const boost::intrusive_ptr<Storage_base>& in,
                           boost::intrusive_ptr<Storage_base>& out,
                           const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_u64tcf(const boost::intrusive_ptr<Storage_base>& in,
                           boost::intrusive_ptr<Storage_base>& out,
                           const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_u64td(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_u64tf(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_u64ti64(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_u64tu64(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_u64ti32(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_u64tu32(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_u64ti16(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_u64tu16(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_u64tb(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);

    void cuCast_gpu_i32tcd(const boost::intrusive_ptr<Storage_base>& in,
                           boost::intrusive_ptr<Storage_base>& out,
                           const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_i32tcf(const boost::intrusive_ptr<Storage_base>& in,
                           boost::intrusive_ptr<Storage_base>& out,
                           const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_i32td(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_i32tf(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_i32ti64(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_i32tu64(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_i32ti32(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_i32tu32(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_i32ti16(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_i32tu16(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_i32tb(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);

    void cuCast_gpu_u32tcd(const boost::intrusive_ptr<Storage_base>& in,
                           boost::intrusive_ptr<Storage_base>& out,
                           const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_u32tcf(const boost::intrusive_ptr<Storage_base>& in,
                           boost::intrusive_ptr<Storage_base>& out,
                           const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_u32td(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_u32tf(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_u32ti64(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_u32tu64(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_u32ti32(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_u32tu32(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_u32ti16(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_u32tu16(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_u32tb(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);

    void cuCast_gpu_i16tcd(const boost::intrusive_ptr<Storage_base>& in,
                           boost::intrusive_ptr<Storage_base>& out,
                           const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_i16tcf(const boost::intrusive_ptr<Storage_base>& in,
                           boost::intrusive_ptr<Storage_base>& out,
                           const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_i16td(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_i16tf(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_i16ti64(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_i16tu64(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_i16ti32(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_i16tu32(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_i16ti16(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_i16tu16(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_i16tb(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);

    void cuCast_gpu_u16tcd(const boost::intrusive_ptr<Storage_base>& in,
                           boost::intrusive_ptr<Storage_base>& out,
                           const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_u16tcf(const boost::intrusive_ptr<Storage_base>& in,
                           boost::intrusive_ptr<Storage_base>& out,
                           const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_u16td(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_u16tf(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_u16ti64(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_u16tu64(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_u16ti32(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_u16tu32(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_u16ti16(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_u16tu16(const boost::intrusive_ptr<Storage_base>& in,
                            boost::intrusive_ptr<Storage_base>& out,
                            const unsigned long long& len_in, const int& alloc_device);
    void cuCast_gpu_u16tb(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);

    void cuCast_gpu_btcd(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& alloc_device);
    void cuCast_gpu_btcf(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& alloc_device);
    void cuCast_gpu_btd(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& alloc_device);
    void cuCast_gpu_btf(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& alloc_device);
    void cuCast_gpu_bti64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_btu64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_bti32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_btu32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_bti16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_btu16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& alloc_device);
    void cuCast_gpu_btb(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& alloc_device);

  }  // namespace utils_internal
}  // namespace cytnx

#endif
