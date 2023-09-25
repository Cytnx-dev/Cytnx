#include "utils/vec_erase.hpp"
#include "utils/vec_print.hpp"
#include "cytnx_error.hpp"
#include <algorithm>
#include <vector>
#include <cstring>

#ifdef BACKEND_TORCH
  #include "Bond.hpp"
#else

  #include "Bond.hpp"
  #include "Tensor.hpp"

#endif

namespace cytnx {

  bool _largeTosmall(const cytnx_uint64 &i, const cytnx_uint64 &j) { return (i > j); }

  template <class T>
  std::vector<T> vec_erase(const std::vector<T> &in, const std::vector<cytnx_uint64> &eraseper) {
    std::vector<T> out = in;
    std::vector<cytnx_uint64> idxs = eraseper;
    std::sort(idxs.begin(), idxs.end(), _largeTosmall);

    for (cytnx_uint64 i = 0; i < idxs.size(); i++) {
      cytnx_error_msg(idxs[i] >= in.size(),
                      "[ERROR][vec_erase] eraseper exceed the size of in vector%s", "\n");
      out.erase(out.begin() + idxs[i]);
    }
    return out;
  }

  template <class T>
  void vec_erase_(std::vector<T> &in, const std::vector<cytnx_uint64> &eraseper) {
    std::vector<cytnx_uint64> idxs = eraseper;
    std::sort(idxs.begin(), idxs.end(), _largeTosmall);

    for (cytnx_uint64 i = 0; i < idxs.size(); i++) {
      cytnx_error_msg(idxs[i] >= in.size(),
                      "[ERROR][vec_erase] eraseper exceed the size of in vector%s", "\n");
      in.erase(in.begin() + idxs[i]);
    }
  }

  template std::vector<cytnx_complex128> vec_erase(const std::vector<cytnx_complex128> &,
                                                   const std::vector<cytnx_uint64> &);
  template std::vector<cytnx_complex64> vec_erase(const std::vector<cytnx_complex64> &,
                                                  const std::vector<cytnx_uint64> &);
  template std::vector<cytnx_double> vec_erase(const std::vector<cytnx_double> &,
                                               const std::vector<cytnx_uint64> &);
  template std::vector<cytnx_float> vec_erase(const std::vector<cytnx_float> &,
                                              const std::vector<cytnx_uint64> &);
  template std::vector<cytnx_int64> vec_erase(const std::vector<cytnx_int64> &,
                                              const std::vector<cytnx_uint64> &);
  template std::vector<cytnx_uint64> vec_erase(const std::vector<cytnx_uint64> &,
                                               const std::vector<cytnx_uint64> &);
  template std::vector<cytnx_int32> vec_erase(const std::vector<cytnx_int32> &,
                                              const std::vector<cytnx_uint64> &);
  template std::vector<cytnx_uint32> vec_erase(const std::vector<cytnx_uint32> &,
                                               const std::vector<cytnx_uint64> &);
  template std::vector<cytnx_int16> vec_erase(const std::vector<cytnx_int16> &,
                                              const std::vector<cytnx_uint64> &);
  template std::vector<cytnx_uint16> vec_erase(const std::vector<cytnx_uint16> &,
                                               const std::vector<cytnx_uint64> &);
  template std::vector<cytnx_bool> vec_erase(const std::vector<cytnx_bool> &,
                                             const std::vector<cytnx_uint64> &);
  template std::vector<std::string> vec_erase(const std::vector<std::string> &,
                                              const std::vector<cytnx_uint64> &);

  template std::vector<std::vector<cytnx_uint64>> vec_erase(
    const std::vector<std::vector<cytnx_uint64>> &, const std::vector<cytnx_uint64> &);
  template std::vector<std::vector<cytnx_int64>> vec_erase(
    const std::vector<std::vector<cytnx_int64>> &, const std::vector<cytnx_uint64> &);

  template void vec_erase_(std::vector<cytnx_complex128> &, const std::vector<cytnx_uint64> &);
  template void vec_erase_(std::vector<cytnx_complex64> &, const std::vector<cytnx_uint64> &);
  template void vec_erase_(std::vector<cytnx_double> &, const std::vector<cytnx_uint64> &);
  template void vec_erase_(std::vector<cytnx_float> &, const std::vector<cytnx_uint64> &);
  template void vec_erase_(std::vector<cytnx_int64> &, const std::vector<cytnx_uint64> &);
  template void vec_erase_(std::vector<cytnx_uint64> &, const std::vector<cytnx_uint64> &);
  template void vec_erase_(std::vector<cytnx_int32> &, const std::vector<cytnx_uint64> &);
  template void vec_erase_(std::vector<cytnx_uint32> &, const std::vector<cytnx_uint64> &);
  template void vec_erase_(std::vector<cytnx_int16> &, const std::vector<cytnx_uint64> &);
  template void vec_erase_(std::vector<cytnx_uint16> &, const std::vector<cytnx_uint64> &);
  template void vec_erase_(std::vector<cytnx_bool> &, const std::vector<cytnx_uint64> &);
  template void vec_erase_(std::vector<std::string> &, const std::vector<cytnx_uint64> &);

#ifdef BACKEND_TORCH
#else

  template std::vector<Tensor> vec_erase(const std::vector<Tensor> &,
                                         const std::vector<cytnx_uint64> &);
  template void vec_erase_(std::vector<Tensor> &, const std::vector<cytnx_uint64> &);

#endif

  template std::vector<Bond> vec_erase(const std::vector<Bond> &,
                                       const std::vector<cytnx_uint64> &);
  template void vec_erase_(std::vector<Bond> &, const std::vector<cytnx_uint64> &);

  template void vec_erase_(std::vector<std::vector<cytnx_uint64>> &,
                           const std::vector<cytnx_uint64> &);

  template void vec_erase_(std::vector<std::vector<cytnx_int64>> &,
                           const std::vector<cytnx_uint64> &);

}  // namespace cytnx
