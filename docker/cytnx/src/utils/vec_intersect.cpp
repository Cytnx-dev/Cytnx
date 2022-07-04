#include "utils/vec_intersect.hpp"
#include "utils/utils_internal_interface.hpp"
#include "Bond.hpp"
#include <algorithm>
#include <vector>
#include <cstring>
namespace cytnx {

  std::vector<std::vector<cytnx_int64>> vec2d_intersect(
    const std::vector<std::vector<cytnx_int64>> &inL,
    const std::vector<std::vector<cytnx_int64>> &inR, const bool &sorted_L, const bool &sorted_R) {
    std::vector<std::vector<cytnx_int64>> out;
    std::vector<std::vector<cytnx_int64>> v1 = inL;
    std::vector<std::vector<cytnx_int64>> v2 = inR;
    if (!sorted_L) std::sort(v1.begin(), v1.end(), utils_internal::_fx_compare_vec_inc);
    if (!sorted_R) std::sort(v2.begin(), v2.end(), utils_internal::_fx_compare_vec_inc);

    std::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(out));
    return out;
  }

  template <class T>
  std::vector<T> vec_intersect(const std::vector<T> &inL, const std::vector<T> &inR) {
    std::vector<T> out;
    std::vector<T> v1 = inL;  // copy
    std::vector<T> v2 = inR;  // copy
    std::sort(v1.begin(), v1.end());
    std::sort(v2.begin(), v2.end());

    std::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(out));
    return out;
  }

  template <class T>
  void vec_intersect_(std::vector<T> &out, const std::vector<T> &inL, const std::vector<T> &inR,
                      std::vector<cytnx_uint64> &indices_v1,
                      std::vector<cytnx_uint64> &indices_v2) {
    out.clear();
    indices_v1.clear();
    indices_v2.clear();
    std::vector<T> v1 = inL;
    std::vector<T> v2 = inR;  // copy
    std::sort(v1.begin(), v1.end());
    std::sort(v2.begin(), v2.end());

    std::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(out));
    for (cytnx_uint64 i = 0; i < out.size(); i++) {
      indices_v1.push_back(std::distance(inL.begin(), std::find(inL.begin(), inL.end(), out[i])));
      indices_v2.push_back(std::distance(inR.begin(), std::find(inR.begin(), inR.end(), out[i])));
    }
  }

  template <class T>
  void vec_intersect_(std::vector<T> &out, const std::vector<T> &inL, const std::vector<T> &inR) {
    out.clear();
    std::vector<T> v1 = inL;
    std::vector<T> v2 = inR;  // copy
    std::sort(v1.begin(), v1.end());
    std::sort(v2.begin(), v2.end());

    std::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(out));
  }

  // template std::vector<cytnx_complex128> vec_intersect(const std::vector<cytnx_complex128>
  // &,const std::vector<cytnx_complex128> &); template std::vector<cytnx_complex64>
  // vec_intersect(const std::vector<cytnx_complex64> &,const std::vector<cytnx_complex64> &);
  template std::vector<cytnx_double> vec_intersect(const std::vector<cytnx_double> &,
                                                   const std::vector<cytnx_double> &);
  template std::vector<cytnx_float> vec_intersect(const std::vector<cytnx_float> &,
                                                  const std::vector<cytnx_float> &);
  template std::vector<cytnx_int64> vec_intersect(const std::vector<cytnx_int64> &,
                                                  const std::vector<cytnx_int64> &);
  template std::vector<cytnx_uint64> vec_intersect(const std::vector<cytnx_uint64> &,
                                                   const std::vector<cytnx_uint64> &);
  template std::vector<cytnx_int32> vec_intersect(const std::vector<cytnx_int32> &,
                                                  const std::vector<cytnx_int32> &);
  template std::vector<cytnx_uint32> vec_intersect(const std::vector<cytnx_uint32> &,
                                                   const std::vector<cytnx_uint32> &);
  template std::vector<cytnx_int16> vec_intersect(const std::vector<cytnx_int16> &,
                                                  const std::vector<cytnx_int16> &);
  template std::vector<cytnx_uint16> vec_intersect(const std::vector<cytnx_uint16> &,
                                                   const std::vector<cytnx_uint16> &);
  template std::vector<cytnx_bool> vec_intersect(const std::vector<cytnx_bool> &,
                                                 const std::vector<cytnx_bool> &);

  // template void vec_intersect_(std::vector<cytnx_complex128> &out, const
  // std::vector<cytnx_complex128> &,const std::vector<cytnx_complex128> &); template void
  // vec_intersect_(std::vector<cytnx_complex64> &out,const std::vector<cytnx_complex64> &,const
  // std::vector<cytnx_complex64> &);
  template void vec_intersect_(std::vector<cytnx_double> &out, const std::vector<cytnx_double> &,
                               const std::vector<cytnx_double> &, std::vector<cytnx_uint64> &,
                               std::vector<cytnx_uint64> &);
  template void vec_intersect_(std::vector<cytnx_float> &out, const std::vector<cytnx_float> &,
                               const std::vector<cytnx_float> &, std::vector<cytnx_uint64> &,
                               std::vector<cytnx_uint64> &);
  template void vec_intersect_(std::vector<cytnx_int64> &out, const std::vector<cytnx_int64> &,
                               const std::vector<cytnx_int64> &, std::vector<cytnx_uint64> &,
                               std::vector<cytnx_uint64> &);
  template void vec_intersect_(std::vector<cytnx_uint64> &out, const std::vector<cytnx_uint64> &,
                               const std::vector<cytnx_uint64> &, std::vector<cytnx_uint64> &,
                               std::vector<cytnx_uint64> &);
  template void vec_intersect_(std::vector<cytnx_int32> &out, const std::vector<cytnx_int32> &,
                               const std::vector<cytnx_int32> &, std::vector<cytnx_uint64> &,
                               std::vector<cytnx_uint64> &);
  template void vec_intersect_(std::vector<cytnx_uint32> &out, const std::vector<cytnx_uint32> &,
                               const std::vector<cytnx_uint32> &, std::vector<cytnx_uint64> &,
                               std::vector<cytnx_uint64> &);
  template void vec_intersect_(std::vector<cytnx_int16> &out, const std::vector<cytnx_int16> &,
                               const std::vector<cytnx_int16> &, std::vector<cytnx_uint64> &,
                               std::vector<cytnx_uint64> &);
  template void vec_intersect_(std::vector<cytnx_uint16> &out, const std::vector<cytnx_uint16> &,
                               const std::vector<cytnx_uint16> &, std::vector<cytnx_uint64> &,
                               std::vector<cytnx_uint64> &);
  template void vec_intersect_(std::vector<cytnx_bool> &out, const std::vector<cytnx_bool> &,
                               const std::vector<cytnx_bool> &, std::vector<cytnx_uint64> &,
                               std::vector<cytnx_uint64> &);
  // template void vec_intersect_(std::vector<cytnx_complex128> &out, const
  // std::vector<cytnx_complex128> &,const std::vector<cytnx_complex128> &); template void
  // vec_intersect_(std::vector<cytnx_complex64> &out,const std::vector<cytnx_complex64> &,const
  // std::vector<cytnx_complex64> &);
  template void vec_intersect_(std::vector<cytnx_double> &out, const std::vector<cytnx_double> &,
                               const std::vector<cytnx_double> &);
  template void vec_intersect_(std::vector<cytnx_float> &out, const std::vector<cytnx_float> &,
                               const std::vector<cytnx_float> &);
  template void vec_intersect_(std::vector<cytnx_int64> &out, const std::vector<cytnx_int64> &,
                               const std::vector<cytnx_int64> &);
  template void vec_intersect_(std::vector<cytnx_uint64> &out, const std::vector<cytnx_uint64> &,
                               const std::vector<cytnx_uint64> &);
  template void vec_intersect_(std::vector<cytnx_int32> &out, const std::vector<cytnx_int32> &,
                               const std::vector<cytnx_int32> &);
  template void vec_intersect_(std::vector<cytnx_uint32> &out, const std::vector<cytnx_uint32> &,
                               const std::vector<cytnx_uint32> &);
  template void vec_intersect_(std::vector<cytnx_int16> &out, const std::vector<cytnx_int16> &,
                               const std::vector<cytnx_int16> &);
  template void vec_intersect_(std::vector<cytnx_uint16> &out, const std::vector<cytnx_uint16> &,
                               const std::vector<cytnx_uint16> &);
  template void vec_intersect_(std::vector<cytnx_bool> &out, const std::vector<cytnx_bool> &,
                               const std::vector<cytnx_bool> &);

}  // namespace cytnx
