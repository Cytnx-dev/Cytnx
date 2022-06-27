#ifndef __H_vec_clone_
#define __H_vec_clone_

#include <vector>
#include "Type.hpp"
namespace cytnx {

  template <class T>
  std::vector<T> vec_clone(const std::vector<T>& in_vec);

  template <class T>
  std::vector<T> vec_clone(const std::vector<T>& in_vec, const cytnx_uint64& Nelem);

  template <class T>
  std::vector<T> vec_clone(const std::vector<T>& in_vec, const std::vector<cytnx_uint64>& locators);

  template <class T>
  std::vector<T> vec_clone(const std::vector<T>& in_vec, const cytnx_uint64& start,
                           const cytnx_uint64& end);

}  // namespace cytnx
#endif
