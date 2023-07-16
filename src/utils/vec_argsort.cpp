#include "utils/vec_argsort.hpp"

namespace cytnx {
  template <class T>
  std::vector<cytnx_uint64> vec_argsort(const std::vector<T>& in) {
    std::vector<cytnx_uint64> v(in.size());
    std::iota(v.begin(), v.end(), 0);
    std::sort(v.begin(), v.end(), [&](cytnx_uint64 i, cytnx_uint64 j) { return in[i] < in[j]; });
    return v;
  }

  template std::vector<cytnx_uint64> vec_argsort(const std::vector<cytnx_uint64>& in);
  template std::vector<cytnx_uint64> vec_argsort(const std::vector<cytnx_uint32>& in);
  template std::vector<cytnx_uint64> vec_argsort(const std::vector<cytnx_uint16>& in);
  template std::vector<cytnx_uint64> vec_argsort(const std::vector<cytnx_int64>& in);
  template std::vector<cytnx_uint64> vec_argsort(const std::vector<cytnx_int32>& in);
  template std::vector<cytnx_uint64> vec_argsort(const std::vector<cytnx_int16>& in);
  template std::vector<cytnx_uint64> vec_argsort(const std::vector<cytnx_double>& in);
  template std::vector<cytnx_uint64> vec_argsort(const std::vector<cytnx_float>& in);
  template std::vector<cytnx_uint64> vec_argsort(const std::vector<cytnx_bool>& in);

  template std::vector<cytnx_uint64> vec_argsort(const std::vector<std::vector<cytnx_uint64>>& in);
  template std::vector<cytnx_uint64> vec_argsort(const std::vector<std::vector<cytnx_int64>>& in);

}  // namespace cytnx
