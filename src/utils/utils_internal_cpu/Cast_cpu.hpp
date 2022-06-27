#ifndef _H_Cast_cpu_
#define _H_Cast_cpu_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "Storage.hpp"
#include "cytnx_error.hpp"
namespace cytnx {
  namespace utils_internal {

    void Cast_cpu_cdtcd(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_cdtcf(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);

    void Cast_cpu_cftcd(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_cftcf(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);

    void Cast_cpu_dtcd(const boost::intrusive_ptr<Storage_base>& in,
                       boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                       const int& is_alloc);
    void Cast_cpu_dtcf(const boost::intrusive_ptr<Storage_base>& in,
                       boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                       const int& is_alloc);
    void Cast_cpu_dtd(const boost::intrusive_ptr<Storage_base>& in,
                      boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                      const int& is_alloc);
    void Cast_cpu_dtf(const boost::intrusive_ptr<Storage_base>& in,
                      boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                      const int& is_alloc);
    void Cast_cpu_dti64(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_dtu64(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_dti32(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_dtu32(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_dtu16(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_dti16(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_dtb(const boost::intrusive_ptr<Storage_base>& in,
                      boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                      const int& is_alloc);

    void Cast_cpu_ftcd(const boost::intrusive_ptr<Storage_base>& in,
                       boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                       const int& is_alloc);
    void Cast_cpu_ftcf(const boost::intrusive_ptr<Storage_base>& in,
                       boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                       const int& is_alloc);
    void Cast_cpu_ftd(const boost::intrusive_ptr<Storage_base>& in,
                      boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                      const int& is_alloc);
    void Cast_cpu_ftf(const boost::intrusive_ptr<Storage_base>& in,
                      boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                      const int& is_alloc);
    void Cast_cpu_fti64(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_ftu64(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_fti32(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_ftu32(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_ftu16(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_fti16(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_ftb(const boost::intrusive_ptr<Storage_base>& in,
                      boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                      const int& is_alloc);

    void Cast_cpu_i64tcd(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& is_alloc);
    void Cast_cpu_i64tcf(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& is_alloc);
    void Cast_cpu_i64td(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_i64tf(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_i64ti64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_i64tu64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_i64ti32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_i64tu32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_i64tu16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_i64ti16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_i64tb(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);

    void Cast_cpu_u64tcd(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& is_alloc);
    void Cast_cpu_u64tcf(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& is_alloc);
    void Cast_cpu_u64td(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_u64tf(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_u64ti64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_u64tu64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_u64ti32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_u64tu32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_u64tu16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_u64ti16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_u64tb(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);

    void Cast_cpu_i32tcd(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& is_alloc);
    void Cast_cpu_i32tcf(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& is_alloc);
    void Cast_cpu_i32td(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_i32tf(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_i32ti64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_i32tu64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_i32ti32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_i32tu32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_i32tu16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_i32ti16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_i32tb(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);

    void Cast_cpu_u32tcd(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& is_alloc);
    void Cast_cpu_u32tcf(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& is_alloc);
    void Cast_cpu_u32td(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_u32tf(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_u32ti64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_u32tu64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_u32ti32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_u32tu32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_u32tu16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_u32ti16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_u32tb(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);

    void Cast_cpu_i16tcd(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& is_alloc);
    void Cast_cpu_i16tcf(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& is_alloc);
    void Cast_cpu_i16td(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_i16tf(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_i16ti64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_i16tu64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_i16ti32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_i16tu32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_i16tu16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_i16ti16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_i16tb(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);

    void Cast_cpu_u16tcd(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& is_alloc);
    void Cast_cpu_u16tcf(const boost::intrusive_ptr<Storage_base>& in,
                         boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                         const int& is_alloc);
    void Cast_cpu_u16td(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_u16tf(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_u16ti64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_u16tu64(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_u16ti32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_u16tu32(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_u16tu16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_u16ti16(const boost::intrusive_ptr<Storage_base>& in,
                          boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                          const int& is_alloc);
    void Cast_cpu_u16tb(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);

    void Cast_cpu_btcd(const boost::intrusive_ptr<Storage_base>& in,
                       boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                       const int& is_alloc);
    void Cast_cpu_btcf(const boost::intrusive_ptr<Storage_base>& in,
                       boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                       const int& is_alloc);
    void Cast_cpu_btd(const boost::intrusive_ptr<Storage_base>& in,
                      boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                      const int& is_alloc);
    void Cast_cpu_btf(const boost::intrusive_ptr<Storage_base>& in,
                      boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                      const int& is_alloc);
    void Cast_cpu_bti64(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_btu64(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_bti32(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_btu32(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_btu16(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_bti16(const boost::intrusive_ptr<Storage_base>& in,
                        boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                        const int& is_alloc);
    void Cast_cpu_btb(const boost::intrusive_ptr<Storage_base>& in,
                      boost::intrusive_ptr<Storage_base>& out, const unsigned long long& len_in,
                      const int& is_alloc);

  }  // namespace utils_internal

}  // namespace cytnx

#endif
