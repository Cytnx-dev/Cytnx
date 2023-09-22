#ifndef _H_vec_range_
#define _H_vec_range_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "backend/Storage.hpp"
#include "cytnx_error.hpp"

namespace cytnx {
  std::vector<cytnx_uint64> vec_range(const cytnx_uint64 &len);
  std::vector<cytnx_uint64> vec_range(const cytnx_uint64 &start, const cytnx_uint64 &end);

  template <class T>
  std::vector<T> vec_range(const cytnx_int64 &len);
  template <class T>
  std::vector<T> vec_range(const cytnx_int64 &start, const cytnx_int64 &end);

  void vec_range_(std::vector<cytnx_uint64> &v, const cytnx_uint64 &len);
  void vec_range_(std::vector<cytnx_uint64> &v, const cytnx_uint64 &start, const cytnx_uint64 &end);

  template <class T>
  void vec_range_(std::vector<cytnx_int64> &v, const cytnx_int64 &len);
  template <class T>
  void vec_range_(std::vector<cytnx_int64> &v, const cytnx_int64 &start, const cytnx_int64 &end);

}  // namespace cytnx
#endif
