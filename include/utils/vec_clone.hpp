#ifndef CYTNX_UTILS_VEC_CLONE_H_
#define CYTNX_UTILS_VEC_CLONE_H_

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <vector>

#include "Type.hpp"
namespace cytnx {
  namespace internal {
    template <typename, typename = std::void_t<>>
    struct has_clone : std::false_type {};

    template <typename T>
    struct has_clone<T, std::void_t<decltype(std::declval<T>().clone())>> : std::true_type {};
  }  // namespace internal

  template <class T, std::enable_if_t<internal::has_clone<T>::value, bool> = true>
  std::vector<T> vec_clone(const std::vector<T>& in_vec) {
    std::vector<T> out;
    out.reserve(in_vec.size());
    std::transform(in_vec.begin(), in_vec.end(), std::back_inserter(out),
                   [](const T& element) { return element.clone(); });
    return out;
  }

  template <class T,
            std::enable_if_t<std::is_same_v<T, std::string> || std::is_same_v<T, cytnx_uint64>,
                             bool> = true>
  std::vector<T> vec_clone(const std::vector<T>& in_vec,
                           const std::vector<cytnx_uint64>& locators) {
    std::vector<T> out;
    out.reserve(locators.size());
    for (cytnx_uint64 index : locators) {
      cytnx_error_msg(index >= in_vec.size(),
                      "[ERROR] The index [%d] in locators exceeds the bound.\n", index);
      out.push_back(in_vec[index]);
    }
    return out;
  };
}  // namespace cytnx

#endif  // CYTNX_UTILS_VEC_CLONE_H_
