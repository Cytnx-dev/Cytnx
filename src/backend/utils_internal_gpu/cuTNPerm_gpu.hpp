#ifndef _H_cuTNPerm_gpu_
#define _H_cuTNPerm_gpu_

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
      boost::intrusive_ptr<Storage_base> &in, const std::vector<cytnx_uint64> &old_shape,
      const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &invmapper, );

    boost::intrusive_ptr<Storage_base> cuTNPerm_gpu_cf(
      boost::intrusive_ptr<Storage_base> &in, const std::vector<cytnx_uint64> &old_shape,
      const std::vector<cytnx_uint64> &mapper, const std::vector<cytnx_uint64> &invmapper, );

    boost::intrusive_ptr<Storage_base> cuTNPerm_gpu_d(boost::intrusive_ptr<Storage_base> &in,
                                                      const std::vector<cytnx_uint64> &old_shape,
                                                      const std::vector<cytnx_uint64> &mapper,
                                                      const std::vector<cytnx_uint64> &invmapper, );

    boost::intrusive_ptr<Storage_base> cuTNPerm_gpu_f(boost::intrusive_ptr<Storage_base> &in,
                                                      const std::vector<cytnx_uint64> &old_shape,
                                                      const std::vector<cytnx_uint64> &mapper,
                                                      const std::vector<cytnx_uint64> &invmapper, );

#endif

  }  // namespace utils_internal
}  // namespace cytnx
#endif
