#ifndef CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUMOVEMEM_GPU_H_
#define CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUMOVEMEM_GPU_H_

#include <vector>

#include "boost/smart_ptr/intrusive_ptr.hpp"

#include "Type.hpp"

namespace cytnx {
  // TODO: Remove the dependency of Storage.
  class Storage_base;

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
#endif  // CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUMOVEMEM_GPU_H_
