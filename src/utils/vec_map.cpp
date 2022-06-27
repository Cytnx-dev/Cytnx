#include "utils/vec_map.hpp"
#include <cytnx_error.hpp>
#include <algorithm>
#include <vector>
#include <cstring>
#include "Bond.hpp"
#include "Accessor.hpp"
namespace cytnx {

  template <class T>
  std::vector<T> vec_map(const std::vector<T> &in, const std::vector<cytnx_uint64> &mapper) {
    cytnx_error_msg(in.size() != mapper.size(),
                    "[ERROR][vec_map] in and mapper does not have same size.%s", "\n");
    std::vector<T> out(in.size());
    for (cytnx_uint64 i = 0; i < in.size(); i++) {
      cytnx_error_msg(mapper[i] >= in.size(),
                      "[ERROR][vec_map] mapper exceed the size of in vector%s", "\n");
      out[i] = in[mapper[i]];
    }
    return out;
  }

  template std::vector<cytnx_complex128> vec_map(const std::vector<cytnx_complex128> &,
                                                 const std::vector<cytnx_uint64> &);
  template std::vector<cytnx_complex64> vec_map(const std::vector<cytnx_complex64> &,
                                                const std::vector<cytnx_uint64> &);
  template std::vector<cytnx_double> vec_map(const std::vector<cytnx_double> &,
                                             const std::vector<cytnx_uint64> &);
  template std::vector<cytnx_float> vec_map(const std::vector<cytnx_float> &,
                                            const std::vector<cytnx_uint64> &);
  template std::vector<cytnx_int64> vec_map(const std::vector<cytnx_int64> &,
                                            const std::vector<cytnx_uint64> &);
  template std::vector<cytnx_uint64> vec_map(const std::vector<cytnx_uint64> &,
                                             const std::vector<cytnx_uint64> &);
  template std::vector<cytnx_int32> vec_map(const std::vector<cytnx_int32> &,
                                            const std::vector<cytnx_uint64> &);
  template std::vector<cytnx_uint32> vec_map(const std::vector<cytnx_uint32> &,
                                             const std::vector<cytnx_uint64> &);
  template std::vector<cytnx_int16> vec_map(const std::vector<cytnx_int16> &,
                                            const std::vector<cytnx_uint64> &);
  template std::vector<cytnx_uint16> vec_map(const std::vector<cytnx_uint16> &,
                                             const std::vector<cytnx_uint64> &);
  template std::vector<cytnx_bool> vec_map(const std::vector<cytnx_bool> &,
                                           const std::vector<cytnx_uint64> &);
  template std::vector<Bond> vec_map(const std::vector<Bond> &, const std::vector<cytnx_uint64> &);
  template std::vector<Accessor> vec_map(const std::vector<Accessor> &,
                                         const std::vector<cytnx_uint64> &);
}  // namespace cytnx
