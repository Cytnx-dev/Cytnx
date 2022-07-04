#ifndef __Mul_internal_H__
#define __Mul_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "Storage.hpp"
#include "Type.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// Mul
    void Mul_internal_cdtcd(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_cdtcf(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_cdtd(boost::intrusive_ptr<Storage_base> &out,
                           boost::intrusive_ptr<Storage_base> &Lin,
                           boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                           const std::vector<cytnx_uint64> &shape,
                           const std::vector<cytnx_uint64> &invmapper_L,
                           const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_cdtf(boost::intrusive_ptr<Storage_base> &out,
                           boost::intrusive_ptr<Storage_base> &Lin,
                           boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                           const std::vector<cytnx_uint64> &shape,
                           const std::vector<cytnx_uint64> &invmapper_L,
                           const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_cdti64(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_cdtu64(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_cdti32(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_cdtu32(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_cdtu16(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_cdti16(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_cdtb(boost::intrusive_ptr<Storage_base> &out,
                           boost::intrusive_ptr<Storage_base> &Lin,
                           boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                           const std::vector<cytnx_uint64> &shape,
                           const std::vector<cytnx_uint64> &invmapper_L,
                           const std::vector<cytnx_uint64> &invmapper_R);

    void Mul_internal_cftcd(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_cftcf(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_cftd(boost::intrusive_ptr<Storage_base> &out,
                           boost::intrusive_ptr<Storage_base> &Lin,
                           boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                           const std::vector<cytnx_uint64> &shape,
                           const std::vector<cytnx_uint64> &invmapper_L,
                           const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_cftf(boost::intrusive_ptr<Storage_base> &out,
                           boost::intrusive_ptr<Storage_base> &Lin,
                           boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                           const std::vector<cytnx_uint64> &shape,
                           const std::vector<cytnx_uint64> &invmapper_L,
                           const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_cfti64(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_cftu64(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_cfti32(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_cftu32(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_cftu16(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_cfti16(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_cftb(boost::intrusive_ptr<Storage_base> &out,
                           boost::intrusive_ptr<Storage_base> &Lin,
                           boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                           const std::vector<cytnx_uint64> &shape,
                           const std::vector<cytnx_uint64> &invmapper_L,
                           const std::vector<cytnx_uint64> &invmapper_R);

    void Mul_internal_dtcd(boost::intrusive_ptr<Storage_base> &out,
                           boost::intrusive_ptr<Storage_base> &Lin,
                           boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                           const std::vector<cytnx_uint64> &shape,
                           const std::vector<cytnx_uint64> &invmapper_L,
                           const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_dtcf(boost::intrusive_ptr<Storage_base> &out,
                           boost::intrusive_ptr<Storage_base> &Lin,
                           boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                           const std::vector<cytnx_uint64> &shape,
                           const std::vector<cytnx_uint64> &invmapper_L,
                           const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_dtd(boost::intrusive_ptr<Storage_base> &out,
                          boost::intrusive_ptr<Storage_base> &Lin,
                          boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                          const std::vector<cytnx_uint64> &shape,
                          const std::vector<cytnx_uint64> &invmapper_L,
                          const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_dtf(boost::intrusive_ptr<Storage_base> &out,
                          boost::intrusive_ptr<Storage_base> &Lin,
                          boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                          const std::vector<cytnx_uint64> &shape,
                          const std::vector<cytnx_uint64> &invmapper_L,
                          const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_dti64(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_dtu64(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_dti32(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_dtu32(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_dtu16(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_dti16(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_dtb(boost::intrusive_ptr<Storage_base> &out,
                          boost::intrusive_ptr<Storage_base> &Lin,
                          boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                          const std::vector<cytnx_uint64> &shape,
                          const std::vector<cytnx_uint64> &invmapper_L,
                          const std::vector<cytnx_uint64> &invmapper_R);

    void Mul_internal_ftcd(boost::intrusive_ptr<Storage_base> &out,
                           boost::intrusive_ptr<Storage_base> &Lin,
                           boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                           const std::vector<cytnx_uint64> &shape,
                           const std::vector<cytnx_uint64> &invmapper_L,
                           const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_ftcf(boost::intrusive_ptr<Storage_base> &out,
                           boost::intrusive_ptr<Storage_base> &Lin,
                           boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                           const std::vector<cytnx_uint64> &shape,
                           const std::vector<cytnx_uint64> &invmapper_L,
                           const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_ftd(boost::intrusive_ptr<Storage_base> &out,
                          boost::intrusive_ptr<Storage_base> &Lin,
                          boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                          const std::vector<cytnx_uint64> &shape,
                          const std::vector<cytnx_uint64> &invmapper_L,
                          const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_ftf(boost::intrusive_ptr<Storage_base> &out,
                          boost::intrusive_ptr<Storage_base> &Lin,
                          boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                          const std::vector<cytnx_uint64> &shape,
                          const std::vector<cytnx_uint64> &invmapper_L,
                          const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_fti64(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_ftu64(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_fti32(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_ftu32(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_ftu16(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_fti16(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_ftb(boost::intrusive_ptr<Storage_base> &out,
                          boost::intrusive_ptr<Storage_base> &Lin,
                          boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                          const std::vector<cytnx_uint64> &shape,
                          const std::vector<cytnx_uint64> &invmapper_L,
                          const std::vector<cytnx_uint64> &invmapper_R);

    void Mul_internal_i64tcd(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i64tcf(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i64td(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i64tf(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i64ti64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i64tu64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i64ti32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i64tu32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i64tu16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i64ti16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i64tb(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);

    void Mul_internal_u64tcd(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u64tcf(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u64td(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u64tf(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u64ti64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u64tu64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u64ti32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u64tu32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u64ti16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u64tu16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u64tb(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);

    void Mul_internal_i32tcd(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i32tcf(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i32td(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i32tf(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i32ti64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i32tu64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i32ti32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i32tu32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i32tu16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i32ti16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i32tb(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);

    void Mul_internal_u32tcd(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u32tcf(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u32td(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u32tf(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u32ti64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u32tu64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u32ti32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u32tu32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u32tu16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u32ti16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u32tb(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);

    void Mul_internal_i16tcd(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i16tcf(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i16td(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i16tf(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i16ti64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i16tu64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i16ti32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i16tu32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i16tu16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i16ti16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_i16tb(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);

    void Mul_internal_u16tcd(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u16tcf(boost::intrusive_ptr<Storage_base> &out,
                             boost::intrusive_ptr<Storage_base> &Lin,
                             boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                             const std::vector<cytnx_uint64> &shape,
                             const std::vector<cytnx_uint64> &invmapper_L,
                             const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u16td(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u16tf(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u16ti64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u16tu64(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u16ti32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u16tu32(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u16tu16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u16ti16(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_u16tb(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);

    void Mul_internal_btcd(boost::intrusive_ptr<Storage_base> &out,
                           boost::intrusive_ptr<Storage_base> &Lin,
                           boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                           const std::vector<cytnx_uint64> &shape,
                           const std::vector<cytnx_uint64> &invmapper_L,
                           const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_btcf(boost::intrusive_ptr<Storage_base> &out,
                           boost::intrusive_ptr<Storage_base> &Lin,
                           boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                           const std::vector<cytnx_uint64> &shape,
                           const std::vector<cytnx_uint64> &invmapper_L,
                           const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_btd(boost::intrusive_ptr<Storage_base> &out,
                          boost::intrusive_ptr<Storage_base> &Lin,
                          boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                          const std::vector<cytnx_uint64> &shape,
                          const std::vector<cytnx_uint64> &invmapper_L,
                          const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_btf(boost::intrusive_ptr<Storage_base> &out,
                          boost::intrusive_ptr<Storage_base> &Lin,
                          boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                          const std::vector<cytnx_uint64> &shape,
                          const std::vector<cytnx_uint64> &invmapper_L,
                          const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_bti64(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_btu64(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_bti32(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_btu32(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_btu16(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_bti16(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R);
    void Mul_internal_btb(boost::intrusive_ptr<Storage_base> &out,
                          boost::intrusive_ptr<Storage_base> &Lin,
                          boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                          const std::vector<cytnx_uint64> &shape,
                          const std::vector<cytnx_uint64> &invmapper_L,
                          const std::vector<cytnx_uint64> &invmapper_R);

  }  // namespace linalg_internal
}  // namespace cytnx

#endif
