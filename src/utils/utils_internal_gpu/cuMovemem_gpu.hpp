#ifndef _H_cuMovemem_gpu_
#define _H_cuMovemem_gpu_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "Storage.hpp"
#include "cytnx_error.hpp"

namespace cytnx {
  namespace utils_internal {
#ifdef UNI_GPU
    boost::intrusive_ptr<Storage_base> cuMovemem_gpu_cd(boost::intrusive_ptr<Storage_base> &in,
                                                        const std::vector<cytnx_uint64> &old_shape,
                                                        const std::vector<cytnx_uint64> &mapper,
                                                        const std::vector<cytnx_uint64> &invmapper,
                                                        const bool is_inplace);

    boost::intrusive_ptr<Storage_base> cuMovemem_gpu_cf(boost::intrusive_ptr<Storage_base> &in,
                                                        const std::vector<cytnx_uint64> &old_shape,
                                                        const std::vector<cytnx_uint64> &mapper,
                                                        const std::vector<cytnx_uint64> &invmapper,
                                                        const bool is_inplace);

    boost::intrusive_ptr<Storage_base> cuMovemem_gpu_d(boost::intrusive_ptr<Storage_base> &in,
                                                       const std::vector<cytnx_uint64> &old_shape,
                                                       const std::vector<cytnx_uint64> &mapper,
                                                       const std::vector<cytnx_uint64> &invmapper,
                                                       const bool is_inplace);

    boost::intrusive_ptr<Storage_base> cuMovemem_gpu_f(boost::intrusive_ptr<Storage_base> &in,
                                                       const std::vector<cytnx_uint64> &old_shape,
                                                       const std::vector<cytnx_uint64> &mapper,
                                                       const std::vector<cytnx_uint64> &invmapper,
                                                       const bool is_inplace);

    boost::intrusive_ptr<Storage_base> cuMovemem_gpu_i64(boost::intrusive_ptr<Storage_base> &in,
                                                         const std::vector<cytnx_uint64> &old_shape,
                                                         const std::vector<cytnx_uint64> &mapper,
                                                         const std::vector<cytnx_uint64> &invmapper,
                                                         const bool is_inplace);

    boost::intrusive_ptr<Storage_base> cuMovemem_gpu_u64(boost::intrusive_ptr<Storage_base> &in,
                                                         const std::vector<cytnx_uint64> &old_shape,
                                                         const std::vector<cytnx_uint64> &mapper,
                                                         const std::vector<cytnx_uint64> &invmapper,
                                                         const bool is_inplace);

    boost::intrusive_ptr<Storage_base> cuMovemem_gpu_i32(boost::intrusive_ptr<Storage_base> &in,
                                                         const std::vector<cytnx_uint64> &old_shape,
                                                         const std::vector<cytnx_uint64> &mapper,
                                                         const std::vector<cytnx_uint64> &invmapper,
                                                         const bool is_inplace);

    boost::intrusive_ptr<Storage_base> cuMovemem_gpu_u32(boost::intrusive_ptr<Storage_base> &in,
                                                         const std::vector<cytnx_uint64> &old_shape,
                                                         const std::vector<cytnx_uint64> &mapper,
                                                         const std::vector<cytnx_uint64> &invmapper,
                                                         const bool is_inplace);
    boost::intrusive_ptr<Storage_base> cuMovemem_gpu_u16(boost::intrusive_ptr<Storage_base> &in,
                                                         const std::vector<cytnx_uint64> &old_shape,
                                                         const std::vector<cytnx_uint64> &mapper,
                                                         const std::vector<cytnx_uint64> &invmapper,
                                                         const bool is_inplace);

    boost::intrusive_ptr<Storage_base> cuMovemem_gpu_i16(boost::intrusive_ptr<Storage_base> &in,
                                                         const std::vector<cytnx_uint64> &old_shape,
                                                         const std::vector<cytnx_uint64> &mapper,
                                                         const std::vector<cytnx_uint64> &invmapper,
                                                         const bool is_inplace);

    boost::intrusive_ptr<Storage_base> cuMovemem_gpu_b(boost::intrusive_ptr<Storage_base> &in,
                                                       const std::vector<cytnx_uint64> &old_shape,
                                                       const std::vector<cytnx_uint64> &mapper,
                                                       const std::vector<cytnx_uint64> &invmapper,
                                                       const bool is_inplace);
#endif

  }  // namespace utils_internal
}  // namespace cytnx
#endif
