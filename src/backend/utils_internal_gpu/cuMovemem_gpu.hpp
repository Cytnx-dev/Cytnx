#ifndef CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUMOVEMEM_GPU_H_
#define CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUMOVEMEM_GPU_H_

#include <type_traits>
#include <vector>

#include "boost/smart_ptr/intrusive_ptr.hpp"

#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace utils_internal {
#ifdef UNI_GPU
    template <typename DType>
    boost::intrusive_ptr<Storage_base> MoveMemoryGpu(boost::intrusive_ptr<Storage_base> &in,
                                                     const std::vector<cytnx_uint64> &old_shape,
                                                     const std::vector<cytnx_uint64> &mapper,
                                                     const std::vector<cytnx_uint64> &invmapper,
                                                     bool is_inplace);

#endif

  }  // namespace utils_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_UTILS_INTERNAL_GPU_CUMOVEMEM_GPU_H_
