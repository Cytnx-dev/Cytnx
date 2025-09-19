#ifndef CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUTNPERM_GPU_H_
#define CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUTNPERM_GPU_H_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "backend/Storage.hpp"
#include "cytnx_error.hpp"

namespace cytnx {
  namespace utils_internal {

#ifdef UNI_GPU
    boost::intrusive_ptr<Storage_base> cuTNPerm_gpu_cd(
      boost::intrusive_ptr<Storage_base>& in, const std::vector<cytnx_uint64>& old_shape,
      const std::vector<cytnx_uint64>& mapper, const std::vector<cytnx_uint64>& invmapper, );

    boost::intrusive_ptr<Storage_base> cuTNPerm_gpu_cf(
      boost::intrusive_ptr<Storage_base>& in, const std::vector<cytnx_uint64>& old_shape,
      const std::vector<cytnx_uint64>& mapper, const std::vector<cytnx_uint64>& invmapper, );

    boost::intrusive_ptr<Storage_base> cuTNPerm_gpu_d(boost::intrusive_ptr<Storage_base>& in,
                                                      const std::vector<cytnx_uint64>& old_shape,
                                                      const std::vector<cytnx_uint64>& mapper,
                                                      const std::vector<cytnx_uint64>& invmapper, );

    boost::intrusive_ptr<Storage_base> cuTNPerm_gpu_f(boost::intrusive_ptr<Storage_base>& in,
                                                      const std::vector<cytnx_uint64>& old_shape,
                                                      const std::vector<cytnx_uint64>& mapper,
                                                      const std::vector<cytnx_uint64>& invmapper, );

#endif

  }  // namespace utils_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUTNPERM_GPU_H_
