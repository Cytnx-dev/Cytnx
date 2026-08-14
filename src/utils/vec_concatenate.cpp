#include "utils/vec_concatenate.hpp"

#include <string>
#include <vector>

#include "Type.hpp"
#include "cytnx_error.hpp"

namespace cytnx {

  template <class T>
  std::vector<T> vec_concatenate(const std::vector<T> &inL, const std::vector<T> &inR) {
    std::vector<T> out;
    out.reserve(inL.size() + inR.size());
    out.insert(out.end(), inL.begin(), inL.end());
    out.insert(out.end(), inR.begin(), inR.end());
    return out;
  }

  template <class T>
  void vec_concatenate_(std::vector<T> &out, const std::vector<T> &inL, const std::vector<T> &inR) {
    cytnx_error_msg(&out == &inL or &out == &inR,
                    "[ERROR][vec_concatenate_][Fromfile] You cannot store the result of "
                    "vec_concatenate_ to inL or inR.%s",
                    "\n");
    out.clear();
    out.reserve(inL.size() + inR.size());
    out.insert(out.end(), inL.begin(), inL.end());
    out.insert(out.end(), inR.begin(), inR.end());
  }

  template std::vector<cytnx_complex128> vec_concatenate(const std::vector<cytnx_complex128> &,
                                                         const std::vector<cytnx_complex128> &);
  template std::vector<cytnx_complex64> vec_concatenate(const std::vector<cytnx_complex64> &,
                                                        const std::vector<cytnx_complex64> &);
  template std::vector<cytnx_double> vec_concatenate(const std::vector<cytnx_double> &,
                                                     const std::vector<cytnx_double> &);
  template std::vector<cytnx_float> vec_concatenate(const std::vector<cytnx_float> &,
                                                    const std::vector<cytnx_float> &);
  template std::vector<cytnx_int64> vec_concatenate(const std::vector<cytnx_int64> &,
                                                    const std::vector<cytnx_int64> &);
  template std::vector<cytnx_uint64> vec_concatenate(const std::vector<cytnx_uint64> &,
                                                     const std::vector<cytnx_uint64> &);
  template std::vector<cytnx_int32> vec_concatenate(const std::vector<cytnx_int32> &,
                                                    const std::vector<cytnx_int32> &);
  template std::vector<cytnx_uint32> vec_concatenate(const std::vector<cytnx_uint32> &,
                                                     const std::vector<cytnx_uint32> &);
  template std::vector<cytnx_int16> vec_concatenate(const std::vector<cytnx_int16> &,
                                                    const std::vector<cytnx_int16> &);
  template std::vector<cytnx_uint16> vec_concatenate(const std::vector<cytnx_uint16> &,
                                                     const std::vector<cytnx_uint16> &);
  template std::vector<std::string> vec_concatenate(const std::vector<std::string> &,
                                                    const std::vector<std::string> &);
  template std::vector<cytnx_bool> vec_concatenate(const std::vector<cytnx_bool> &,
                                                   const std::vector<cytnx_bool> &);

  template void vec_concatenate_(std::vector<cytnx_complex128> &out,
                                 const std::vector<cytnx_complex128> &,
                                 const std::vector<cytnx_complex128> &);
  template void vec_concatenate_(std::vector<cytnx_complex64> &out,
                                 const std::vector<cytnx_complex64> &,
                                 const std::vector<cytnx_complex64> &);
  template void vec_concatenate_(std::vector<cytnx_double> &out, const std::vector<cytnx_double> &,
                                 const std::vector<cytnx_double> &);
  template void vec_concatenate_(std::vector<cytnx_float> &out, const std::vector<cytnx_float> &,
                                 const std::vector<cytnx_float> &);
  template void vec_concatenate_(std::vector<cytnx_int64> &out, const std::vector<cytnx_int64> &,
                                 const std::vector<cytnx_int64> &);
  template void vec_concatenate_(std::vector<cytnx_uint64> &out, const std::vector<cytnx_uint64> &,
                                 const std::vector<cytnx_uint64> &);
  template void vec_concatenate_(std::vector<cytnx_int32> &out, const std::vector<cytnx_int32> &,
                                 const std::vector<cytnx_int32> &);
  template void vec_concatenate_(std::vector<cytnx_uint32> &out, const std::vector<cytnx_uint32> &,
                                 const std::vector<cytnx_uint32> &);
  template void vec_concatenate_(std::vector<cytnx_uint16> &out, const std::vector<cytnx_uint16> &,
                                 const std::vector<cytnx_uint16> &);
  template void vec_concatenate_(std::vector<cytnx_int16> &out, const std::vector<cytnx_int16> &,
                                 const std::vector<cytnx_int16> &);
  template void vec_concatenate_(std::vector<std::string> &out, const std::vector<std::string> &,
                                 const std::vector<std::string> &);
  template void vec_concatenate_(std::vector<cytnx_bool> &out, const std::vector<cytnx_bool> &,
                                 const std::vector<cytnx_bool> &);
}  // namespace cytnx
