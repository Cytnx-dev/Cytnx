#include "utils/vec_where.hpp"
#include "Bond.hpp"
#include <algorithm>
#include <vector>
#include <cstring>
namespace cytnx {

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

}  // namespace cytnx
