#include "utils/vec_unique.hpp"
#include "Type.hpp"
#include <algorithm>
#include <vector>
#include <cstring>
namespace cytnx {

  template <class T>
  bool myfunction(T &i, T &j) {
    return i == j;
  };

  template <class T>
  std::vector<T> vec_unique(const std::vector<T> &in) {
    if (in.size() == 0)
      return std::vector<T>();
    else if (in.size() == 1)
      return in;

    std::vector<T> new_vec(in.size());
    memcpy(&new_vec[0], &in[0], sizeof(T) * in.size());

    typename std::vector<T>::iterator it;
    std::sort(new_vec.begin(), new_vec.end());

    it = std::unique(new_vec.begin(), new_vec.begin() + new_vec.size());
    new_vec.resize(std::distance(new_vec.begin(), it));
    return new_vec;
  };

  template <>
  std::vector<bool> vec_unique(const std::vector<bool> &in) {
    if (in.size() == 0)
      return std::vector<bool>();
    else if (in.size() == 1)
      return in;

    std::vector<bool> new_vec = in;

    std::vector<bool>::iterator it;
    std::sort(new_vec.begin(), new_vec.end());

    it = std::unique(new_vec.begin(), new_vec.begin() + new_vec.size());
    new_vec.resize(std::distance(new_vec.begin(), it));
    return new_vec;
  };

  // template std::vector<cytnx_complex128> vec_unique(const std::vector<cytnx_complex128> &);
  // template std::vector<cytnx_complex64> vec_unique(const std::vector<cytnx_complex64> &);
  template std::vector<cytnx_double> vec_unique(const std::vector<cytnx_double> &);
  template std::vector<cytnx_float> vec_unique(const std::vector<cytnx_float> &);
  template std::vector<cytnx_int64> vec_unique(const std::vector<cytnx_int64> &);
  template std::vector<cytnx_uint64> vec_unique(const std::vector<cytnx_uint64> &);
  template std::vector<cytnx_int32> vec_unique(const std::vector<cytnx_int32> &);
  template std::vector<cytnx_uint32> vec_unique(const std::vector<cytnx_uint32> &);
  template std::vector<cytnx_int16> vec_unique(const std::vector<cytnx_int16> &);
  template std::vector<cytnx_uint16> vec_unique(const std::vector<cytnx_uint16> &);
  // template std::vector<cytnx_bool> vec_unique(const std::vector<cytnx_bool> &);
}  // namespace cytnx
