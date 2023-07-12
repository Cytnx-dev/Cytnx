#include "utils/vec_sort.hpp"
#include "utils/vec_map.hpp"
#include <algorithm>
#include <numeric>
namespace cytnx {
  template <class T>
  std::vector<cytnx_uint64> vec_sort(std::vector<T>& in, const bool& return_order) {
    if (return_order) {
      std::vector<cytnx_uint64> v(in.size());
      std::iota(v.begin(), v.end(), 0);
      std::sort(v.begin(), v.end(),
                [&in](cytnx_uint64 i, cytnx_uint64 j) { return in[i] < in[j]; });

      std::vector<T> cpy_in = in;

      transform(v.begin(), v.end(), in.begin(), [&](std::size_t i) { return cpy_in[i]; });

      return v;
    } else {
      std::sort(in.begin(), in.end());
      return std::vector<cytnx_uint64>();
    }
  }

  template std::vector<cytnx_uint64> vec_sort(std::vector<cytnx_uint64>& in,
                                              const bool& return_order);
  template std::vector<cytnx_uint64> vec_sort(std::vector<cytnx_uint32>& in,
                                              const bool& return_order);
  template std::vector<cytnx_uint64> vec_sort(std::vector<cytnx_uint16>& in,
                                              const bool& return_order);
  template std::vector<cytnx_uint64> vec_sort(std::vector<cytnx_int64>& in,
                                              const bool& return_order);
  template std::vector<cytnx_uint64> vec_sort(std::vector<cytnx_int32>& in,
                                              const bool& return_order);
  template std::vector<cytnx_uint64> vec_sort(std::vector<cytnx_int16>& in,
                                              const bool& return_order);
  template std::vector<cytnx_uint64> vec_sort(std::vector<cytnx_double>& in,
                                              const bool& return_order);
  template std::vector<cytnx_uint64> vec_sort(std::vector<cytnx_float>& in,
                                              const bool& return_order);
  template std::vector<cytnx_uint64> vec_sort(std::vector<cytnx_bool>& in,
                                              const bool& return_order);
  template std::vector<cytnx_uint64> vec_sort(std::vector<std::vector<cytnx_uint64>>& in,
                                              const bool& return_order);
  template std::vector<cytnx_uint64> vec_sort(std::vector<std::vector<cytnx_int64>>& in,
                                              const bool& return_order);

}  // namespace cytnx
