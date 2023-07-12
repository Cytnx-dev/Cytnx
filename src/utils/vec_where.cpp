#include "utils/vec_where.hpp"
#include "cytnx_error.hpp"
#include <algorithm>
#include <vector>
#include <cstring>
namespace cytnx {

  template <class T>
  std::vector<cytnx_uint64> vec_argwhere(std::vector<T> const &v, const T &target) {
    std::vector<cytnx_uint64> indices;
    auto it = v.begin();
    while ((it = std::find_if(it, v.end(), [&](T const &e) { return e == target; })) != v.end()) {
      indices.push_back(std::distance(v.begin(), it));
      it++;
    }
    return indices;
  }

  template <class T>
  cytnx_uint64 vec_where(const std::vector<T> &in, const T &key) {
    typename std::vector<T>::const_iterator it = std::find(in.begin(), in.end(), key);
    cytnx_error_msg(it == in.end(), "[ERROR] no element indicate as [key] is found.%s", "\n");
    return std::distance(in.begin(), it);
  }

  // template cytnx_uint64 vec_where(const std::vector<cytnx_complex128> &,const cytnx_complex128
  // &); template cytnx_uint64 vec_where(const std::vector<cytnx_complex64> &,const cytnx_complex64
  // &);
  template cytnx_uint64 vec_where(const std::vector<cytnx_double> &, const cytnx_double &);
  template cytnx_uint64 vec_where(const std::vector<cytnx_float> &, const cytnx_float &);
  template cytnx_uint64 vec_where(const std::vector<cytnx_int64> &, const cytnx_int64 &);
  template cytnx_uint64 vec_where(const std::vector<cytnx_uint64> &, const cytnx_uint64 &);
  template cytnx_uint64 vec_where(const std::vector<cytnx_int32> &, const cytnx_int32 &);
  template cytnx_uint64 vec_where(const std::vector<cytnx_uint32> &, const cytnx_uint32 &);
  template cytnx_uint64 vec_where(const std::vector<cytnx_int16> &, const cytnx_int16 &);
  template cytnx_uint64 vec_where(const std::vector<cytnx_uint16> &, const cytnx_uint16 &);
  template cytnx_uint64 vec_where(const std::vector<cytnx_bool> &, const cytnx_bool &);
  template cytnx_uint64 vec_where(const std::vector<std::string> &, const std::string &);
  template cytnx_uint64 vec_where(const std::vector<std::vector<cytnx_int64>> &,
                                  const std::vector<cytnx_int64> &);
  template cytnx_uint64 vec_where(const std::vector<std::vector<cytnx_uint64>> &,
                                  const std::vector<cytnx_uint64> &);

  template std::vector<cytnx_uint64> vec_argwhere(std::vector<cytnx_double> const &v,
                                                  const cytnx_double &target);
  template std::vector<cytnx_uint64> vec_argwhere(std::vector<cytnx_float> const &v,
                                                  const cytnx_float &target);
  template std::vector<cytnx_uint64> vec_argwhere(std::vector<cytnx_int64> const &v,
                                                  const cytnx_int64 &target);
  template std::vector<cytnx_uint64> vec_argwhere(std::vector<cytnx_uint64> const &v,
                                                  const cytnx_uint64 &target);
  template std::vector<cytnx_uint64> vec_argwhere(std::vector<cytnx_int32> const &v,
                                                  const cytnx_int32 &target);
  template std::vector<cytnx_uint64> vec_argwhere(std::vector<cytnx_uint32> const &v,
                                                  const cytnx_uint32 &target);
  template std::vector<cytnx_uint64> vec_argwhere(std::vector<cytnx_int16> const &v,
                                                  const cytnx_int16 &target);
  template std::vector<cytnx_uint64> vec_argwhere(std::vector<cytnx_uint16> const &v,
                                                  const cytnx_uint16 &target);
  template std::vector<cytnx_uint64> vec_argwhere(std::vector<cytnx_bool> const &v,
                                                  const cytnx_bool &target);
  template std::vector<cytnx_uint64> vec_argwhere(std::vector<std::string> const &v,
                                                  const std::string &target);
  template std::vector<cytnx_uint64> vec_argwhere(std::vector<std::vector<cytnx_int64>> const &v,
                                                  const std::vector<cytnx_int64> &target);
  template std::vector<cytnx_uint64> vec_argwhere(std::vector<std::vector<cytnx_uint64>> const &v,
                                                  const std::vector<cytnx_uint64> &target);

}  // namespace cytnx
