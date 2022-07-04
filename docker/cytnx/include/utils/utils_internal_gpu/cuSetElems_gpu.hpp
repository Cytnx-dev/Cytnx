#ifndef _H_cuSetElems_gpu_
#define _H_cuSetElems_gpu_

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

    void cuSetElems_gpu_cdtcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_cdtcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_cdtd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                             const std::vector<cytnx_uint64> &new_offj,
                             const std::vector<std::vector<cytnx_uint64>> &locators,
                             const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_cdtf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                             const std::vector<cytnx_uint64> &new_offj,
                             const std::vector<std::vector<cytnx_uint64>> &locators,
                             const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_cdti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_cdtu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_cdti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_cdtu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_cdtu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_cdti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_cdtb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                             const std::vector<cytnx_uint64> &new_offj,
                             const std::vector<std::vector<cytnx_uint64>> &locators,
                             const cytnx_uint64 &TotalElem, const bool &is_scalar);

    void cuSetElems_gpu_cftcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_cftcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_cftd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                             const std::vector<cytnx_uint64> &new_offj,
                             const std::vector<std::vector<cytnx_uint64>> &locators,
                             const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_cftf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                             const std::vector<cytnx_uint64> &new_offj,
                             const std::vector<std::vector<cytnx_uint64>> &locators,
                             const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_cfti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_cftu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_cfti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_cftu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_cftu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_cfti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_cftb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                             const std::vector<cytnx_uint64> &new_offj,
                             const std::vector<std::vector<cytnx_uint64>> &locators,
                             const cytnx_uint64 &TotalElem, const bool &is_scalar);

    void cuSetElems_gpu_dtcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                             const std::vector<cytnx_uint64> &new_offj,
                             const std::vector<std::vector<cytnx_uint64>> &locators,
                             const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_dtcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                             const std::vector<cytnx_uint64> &new_offj,
                             const std::vector<std::vector<cytnx_uint64>> &locators,
                             const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_dtd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                            const std::vector<cytnx_uint64> &new_offj,
                            const std::vector<std::vector<cytnx_uint64>> &locators,
                            const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_dtf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                            const std::vector<cytnx_uint64> &new_offj,
                            const std::vector<std::vector<cytnx_uint64>> &locators,
                            const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_dti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_dtu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_dti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_dtu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_dtu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_dti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_dtb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                            const std::vector<cytnx_uint64> &new_offj,
                            const std::vector<std::vector<cytnx_uint64>> &locators,
                            const cytnx_uint64 &TotalElem, const bool &is_scalar);

    void cuSetElems_gpu_ftcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                             const std::vector<cytnx_uint64> &new_offj,
                             const std::vector<std::vector<cytnx_uint64>> &locators,
                             const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_ftcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                             const std::vector<cytnx_uint64> &new_offj,
                             const std::vector<std::vector<cytnx_uint64>> &locators,
                             const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_ftd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                            const std::vector<cytnx_uint64> &new_offj,
                            const std::vector<std::vector<cytnx_uint64>> &locators,
                            const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_ftf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                            const std::vector<cytnx_uint64> &new_offj,
                            const std::vector<std::vector<cytnx_uint64>> &locators,
                            const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_fti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_ftu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_fti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_ftu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_ftu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_fti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_ftb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                            const std::vector<cytnx_uint64> &new_offj,
                            const std::vector<std::vector<cytnx_uint64>> &locators,
                            const cytnx_uint64 &TotalElem, const bool &is_scalar);

    void cuSetElems_gpu_i64tcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i64tcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i64td(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i64tf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i64ti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i64tu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i64ti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i64tu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i64tu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i64ti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i64tb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);

    void cuSetElems_gpu_u64tcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u64tcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u64td(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u64tf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u64ti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u64tu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u64ti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u64tu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u64tu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u64ti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u64tb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);

    void cuSetElems_gpu_i32tcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i32tcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i32td(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i32tf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i32ti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i32tu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i32ti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i32tu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i32tu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i32ti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i32tb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);

    void cuSetElems_gpu_u32tcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u32tcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u32td(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u32tf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u32ti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u32tu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u32ti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u32tu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u32tu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u32ti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u32tb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);

    void cuSetElems_gpu_i16tcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i16tcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i16td(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i16tf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i16ti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i16tu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i16ti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i16tu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i16tu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i16ti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_i16tb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);

    void cuSetElems_gpu_u16tcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u16tcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                               const std::vector<cytnx_uint64> &new_offj,
                               const std::vector<std::vector<cytnx_uint64>> &locators,
                               const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u16td(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u16tf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u16ti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u16tu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u16ti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u16tu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u16tu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u16ti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                                const std::vector<cytnx_uint64> &new_offj,
                                const std::vector<std::vector<cytnx_uint64>> &locators,
                                const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_u16tb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);

    void cuSetElems_gpu_btcd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                             const std::vector<cytnx_uint64> &new_offj,
                             const std::vector<std::vector<cytnx_uint64>> &locators,
                             const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_btcf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                             const std::vector<cytnx_uint64> &new_offj,
                             const std::vector<std::vector<cytnx_uint64>> &locators,
                             const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_btd(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                            const std::vector<cytnx_uint64> &new_offj,
                            const std::vector<std::vector<cytnx_uint64>> &locators,
                            const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_btf(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                            const std::vector<cytnx_uint64> &new_offj,
                            const std::vector<std::vector<cytnx_uint64>> &locators,
                            const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_bti64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_btu64(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_bti32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_btu32(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_btu16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_bti16(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                              const std::vector<cytnx_uint64> &new_offj,
                              const std::vector<std::vector<cytnx_uint64>> &locators,
                              const cytnx_uint64 &TotalElem, const bool &is_scalar);
    void cuSetElems_gpu_btb(void *in, void *out, const std::vector<cytnx_uint64> &offj,
                            const std::vector<cytnx_uint64> &new_offj,
                            const std::vector<std::vector<cytnx_uint64>> &locators,
                            const cytnx_uint64 &TotalElem, const bool &is_scalar);

  }  // namespace utils_internal
}  // namespace cytnx
#endif
