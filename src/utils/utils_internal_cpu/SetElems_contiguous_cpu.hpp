#ifndef _H_SetElems_conti_cpu_
#define _H_SetElems_conti_cpu_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include <vector>
#include "Type.hpp"
#include "cytnx_error.hpp"
#include "Storage.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace utils_internal {

    void SetElems_conti_cpu_cdtcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_cdtcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_cdtd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                 const std::vector<cytnx_uint64> &new_offj,
                                 const std::vector<std::vector<cytnx_uint64>> &locators,
                                 const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                 const bool &is_scalar);
    void SetElems_conti_cpu_cdtf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                 const std::vector<cytnx_uint64> &new_offj,
                                 const std::vector<std::vector<cytnx_uint64>> &locators,
                                 const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                 const bool &is_scalar);
    void SetElems_conti_cpu_cdti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar);
    void SetElems_conti_cpu_cdtu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar);
    void SetElems_conti_cpu_cdti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar);
    void SetElems_conti_cpu_cdtu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar);
    void SetElems_conti_cpu_cdti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar);
    void SetElems_conti_cpu_cdtu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar);
    void SetElems_conti_cpu_cdtb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                 const std::vector<cytnx_uint64> &new_offj,
                                 const std::vector<std::vector<cytnx_uint64>> &locators,
                                 const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                 const bool &is_scalar);

    void SetElems_conti_cpu_cftcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_cftcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_cftd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                 const std::vector<cytnx_uint64> &new_offj,
                                 const std::vector<std::vector<cytnx_uint64>> &locators,
                                 const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                 const bool &is_scalar);
    void SetElems_conti_cpu_cftf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                 const std::vector<cytnx_uint64> &new_offj,
                                 const std::vector<std::vector<cytnx_uint64>> &locators,
                                 const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                 const bool &is_scalar);
    void SetElems_conti_cpu_cfti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar);
    void SetElems_conti_cpu_cftu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar);
    void SetElems_conti_cpu_cfti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar);
    void SetElems_conti_cpu_cftu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar);
    void SetElems_conti_cpu_cfti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar);
    void SetElems_conti_cpu_cftu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar);
    void SetElems_conti_cpu_cftb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                 const std::vector<cytnx_uint64> &new_offj,
                                 const std::vector<std::vector<cytnx_uint64>> &locators,
                                 const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                 const bool &is_scalar);

    void SetElems_conti_cpu_dtcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                 const std::vector<cytnx_uint64> &new_offj,
                                 const std::vector<std::vector<cytnx_uint64>> &locators,
                                 const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                 const bool &is_scalar);
    void SetElems_conti_cpu_dtcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                 const std::vector<cytnx_uint64> &new_offj,
                                 const std::vector<std::vector<cytnx_uint64>> &locators,
                                 const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                 const bool &is_scalar);
    void SetElems_conti_cpu_dtd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                const bool &is_scalar);
    void SetElems_conti_cpu_dtf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                const bool &is_scalar);
    void SetElems_conti_cpu_dti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_dtu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_dti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_dtu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_dti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_dtu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_dtb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                const bool &is_scalar);

    void SetElems_conti_cpu_ftcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                 const std::vector<cytnx_uint64> &new_offj,
                                 const std::vector<std::vector<cytnx_uint64>> &locators,
                                 const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                 const bool &is_scalar);
    void SetElems_conti_cpu_ftcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                 const std::vector<cytnx_uint64> &new_offj,
                                 const std::vector<std::vector<cytnx_uint64>> &locators,
                                 const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                 const bool &is_scalar);
    void SetElems_conti_cpu_ftd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                const bool &is_scalar);
    void SetElems_conti_cpu_ftf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                const bool &is_scalar);
    void SetElems_conti_cpu_fti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_ftu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_fti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_ftu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_fti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_ftu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_ftb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                const bool &is_scalar);

    void SetElems_conti_cpu_i64tcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar);
    void SetElems_conti_cpu_i64tcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar);
    void SetElems_conti_cpu_i64td(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_i64tf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_i64ti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_i64tu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_i64ti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_i64tu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_i64ti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_i64tu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_i64tb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);

    void SetElems_conti_cpu_u64tcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar);
    void SetElems_conti_cpu_u64tcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar);
    void SetElems_conti_cpu_u64td(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_u64tf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_u64ti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_u64tu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_u64ti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_u64tu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_u64ti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_u64tu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_u64tb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);

    void SetElems_conti_cpu_i32tcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar);
    void SetElems_conti_cpu_i32tcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar);
    void SetElems_conti_cpu_i32td(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_i32tf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_i32ti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_i32tu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_i32ti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_i32tu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_i32ti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_i32tu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_i32tb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);

    void SetElems_conti_cpu_u32tcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar);
    void SetElems_conti_cpu_u32tcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar);
    void SetElems_conti_cpu_u32td(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_u32tf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_u32ti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_u32tu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_u32ti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_u32tu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_u32ti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_u32tu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_u32tb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);

    void SetElems_conti_cpu_i16tcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar);
    void SetElems_conti_cpu_i16tcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar);
    void SetElems_conti_cpu_i16td(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_i16tf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_i16ti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_i16tu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_i16ti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_i16tu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_i16ti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_i16tu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_i16tb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);

    void SetElems_conti_cpu_u16tcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar);
    void SetElems_conti_cpu_u16tcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                   const std::vector<cytnx_uint64> &new_offj,
                                   const std::vector<std::vector<cytnx_uint64>> &locators,
                                   const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                   const bool &is_scalar);
    void SetElems_conti_cpu_u16td(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_u16tf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_u16ti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_u16tu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_u16ti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_u16tu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_u16ti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_u16tu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                    const std::vector<cytnx_uint64> &new_offj,
                                    const std::vector<std::vector<cytnx_uint64>> &locators,
                                    const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                    const bool &is_scalar);
    void SetElems_conti_cpu_u16tb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);

    void SetElems_conti_cpu_btcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                 const std::vector<cytnx_uint64> &new_offj,
                                 const std::vector<std::vector<cytnx_uint64>> &locators,
                                 const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                 const bool &is_scalar);
    void SetElems_conti_cpu_btcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                 const std::vector<cytnx_uint64> &new_offj,
                                 const std::vector<std::vector<cytnx_uint64>> &locators,
                                 const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                 const bool &is_scalar);
    void SetElems_conti_cpu_btd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                const bool &is_scalar);
    void SetElems_conti_cpu_btf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                const bool &is_scalar);
    void SetElems_conti_cpu_bti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_btu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_bti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_btu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_bti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_btu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                  const std::vector<cytnx_uint64> &new_offj,
                                  const std::vector<std::vector<cytnx_uint64>> &locators,
                                  const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                  const bool &is_scalar);
    void SetElems_conti_cpu_btb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const cytnx_uint64 &Nunit,
                                const bool &is_scalar);

  }  // namespace utils_internal
}  // namespace cytnx
#endif
