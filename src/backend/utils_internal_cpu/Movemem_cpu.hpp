#ifndef CYTNX_BACKEND_UTILS_INTERNAL_CPU_MOVEMEM_CPU_H_
#define CYTNX_BACKEND_UTILS_INTERNAL_CPU_MOVEMEM_CPU_H_

#include <type_traits>
#include <vector>

#include "boost/smart_ptr/intrusive_ptr.hpp"

#include "backend/Storage.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace utils_internal {

    template <typename T, typename std::enable_if_t<!std::is_integral_v<T>, bool> = true>
    boost::intrusive_ptr<Storage_base> MoveMemoryCpu(boost::intrusive_ptr<Storage_base> &in,
                                                     const std::vector<cytnx_uint64> &old_shape,
                                                     const std::vector<cytnx_uint64> &mapper,
                                                     const std::vector<cytnx_uint64> &invmapper,
                                                     const bool is_inplace);

    template <typename T, typename std::enable_if_t<std::is_integral_v<T>, bool> = true>
    boost::intrusive_ptr<Storage_base> MoveMemoryCpu(boost::intrusive_ptr<Storage_base> &in,
                                                     const std::vector<cytnx_uint64> &old_shape,
                                                     const std::vector<cytnx_uint64> &mapper,
                                                     const std::vector<cytnx_uint64> &invmapper,
                                                     const bool is_inplace);
  }  // namespace utils_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_UTILS_INTERNAL_CPU_MOVEMEM_CPU_H_
