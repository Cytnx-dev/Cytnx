#ifndef CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUGETELEMS_GPU_H_
#define CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUGETELEMS_GPU_H_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include <vector>
#include "Type.hpp"
#include "cytnx_error.hpp"
#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace utils_internal {

    void cuGetElems_gpu_cd(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                           const std::vector<cytnx_uint64> &new_offj,
                           const std::vector<std::vector<cytnx_uint64>> &locators,
                           const cytnx_uint64 &TotalElem);
    void cuGetElems_gpu_cf(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                           const std::vector<cytnx_uint64> &new_offj,
                           const std::vector<std::vector<cytnx_uint64>> &locators,
                           const cytnx_uint64 &TotalElem);
    void cuGetElems_gpu_d(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                          const std::vector<cytnx_uint64> &new_offj,
                          const std::vector<std::vector<cytnx_uint64>> &locators,
                          const cytnx_uint64 &TotalElem);
    void cuGetElems_gpu_f(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                          const std::vector<cytnx_uint64> &new_offj,
                          const std::vector<std::vector<cytnx_uint64>> &locators,
                          const cytnx_uint64 &TotalElem);

    void cuGetElems_gpu_i64(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                            const std::vector<cytnx_uint64> &new_offj,
                            const std::vector<std::vector<cytnx_uint64>> &locators,
                            const cytnx_uint64 &TotalElem);
    void cuGetElems_gpu_u64(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                            const std::vector<cytnx_uint64> &new_offj,
                            const std::vector<std::vector<cytnx_uint64>> &locators,
                            const cytnx_uint64 &TotalElem);
    void cuGetElems_gpu_i32(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                            const std::vector<cytnx_uint64> &new_offj,
                            const std::vector<std::vector<cytnx_uint64>> &locators,
                            const cytnx_uint64 &TotalElem);
    void cuGetElems_gpu_u32(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                            const std::vector<cytnx_uint64> &new_offj,
                            const std::vector<std::vector<cytnx_uint64>> &locators,
                            const cytnx_uint64 &TotalElem);
    void cuGetElems_gpu_i16(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                            const std::vector<cytnx_uint64> &new_offj,
                            const std::vector<std::vector<cytnx_uint64>> &locators,
                            const cytnx_uint64 &TotalElem);
    void cuGetElems_gpu_u16(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                            const std::vector<cytnx_uint64> &new_offj,
                            const std::vector<std::vector<cytnx_uint64>> &locators,
                            const cytnx_uint64 &TotalElem);
    void cuGetElems_gpu_b(void *out, void *in, const std::vector<cytnx_uint64> &offj,
                          const std::vector<cytnx_uint64> &new_offj,
                          const std::vector<std::vector<cytnx_uint64>> &locators,
                          const cytnx_uint64 &TotalElem);
  }  // namespace utils_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUGETELEMS_GPU_H_
