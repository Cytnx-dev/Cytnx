#ifndef _H_GetElems_cpu_
#define _H_GetElems_cpu_

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

    void GetElems_cpu_cd(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                         const std::vector<cytnx_uint64> &new_offj,
                         const std::vector<std::vector<cytnx_uint64>> &locators,
                         const cytnx_uint64 &TotalElem);
    void GetElems_cpu_cf(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                         const std::vector<cytnx_uint64> &new_offj,
                         const std::vector<std::vector<cytnx_uint64>> &locators,
                         const cytnx_uint64 &TotalElem);
    void GetElems_cpu_d(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                        const std::vector<cytnx_uint64> &new_offj,
                        const std::vector<std::vector<cytnx_uint64>> &locators,
                        const cytnx_uint64 &TotalElem);
    void GetElems_cpu_f(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                        const std::vector<cytnx_uint64> &new_offj,
                        const std::vector<std::vector<cytnx_uint64>> &locators,
                        const cytnx_uint64 &TotalElem);

    void GetElems_cpu_i64(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                          const std::vector<cytnx_uint64> &new_offj,
                          const std::vector<std::vector<cytnx_uint64>> &locators,
                          const cytnx_uint64 &TotalElem);
    void GetElems_cpu_u64(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                          const std::vector<cytnx_uint64> &new_offj,
                          const std::vector<std::vector<cytnx_uint64>> &locators,
                          const cytnx_uint64 &TotalElem);
    void GetElems_cpu_i32(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                          const std::vector<cytnx_uint64> &new_offj,
                          const std::vector<std::vector<cytnx_uint64>> &locators,
                          const cytnx_uint64 &TotalElem);
    void GetElems_cpu_u32(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                          const std::vector<cytnx_uint64> &new_offj,
                          const std::vector<std::vector<cytnx_uint64>> &locators,
                          const cytnx_uint64 &TotalElem);
    void GetElems_cpu_i16(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                          const std::vector<cytnx_uint64> &new_offj,
                          const std::vector<std::vector<cytnx_uint64>> &locators,
                          const cytnx_uint64 &TotalElem);
    void GetElems_cpu_u16(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                          const std::vector<cytnx_uint64> &new_offj,
                          const std::vector<std::vector<cytnx_uint64>> &locators,
                          const cytnx_uint64 &TotalElem);
    void GetElems_cpu_b(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                        const std::vector<cytnx_uint64> &new_offj,
                        const std::vector<std::vector<cytnx_uint64>> &locators,
                        const cytnx_uint64 &TotalElem);
  }  // namespace utils_internal
}  // namespace cytnx
#endif
