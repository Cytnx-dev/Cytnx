#include "utils/vec_range.hpp"
#ifdef UNI_OMP
  #include <omp.h>
#endif

using namespace std;
namespace cytnx {

  vector<cytnx_uint64> vec_range(const cytnx_uint64 &start, const cytnx_uint64 &end) {
    cytnx_error_msg(end < start, "[ERROR] cannot have end < start%s", "\n");
    vector<cytnx_uint64> out(end - start);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
    for (cytnx_uint64 i = 0; i < end - start; i++) {
      out[i] = start + i;
    }
    return out;
  }

  vector<cytnx_uint64> vec_range(const cytnx_uint64 &len) {
    vector<cytnx_uint64> out(len);

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
    for (cytnx_uint64 i = 0; i < len; i++) {
      out[i] = i;
    }
    return out;
  }

  template <>
  vector<cytnx_int64> vec_range<cytnx_int64>(const cytnx_int64 &len) {
    vector<cytnx_int64> out(len);
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
    for (cytnx_int64 i = 0; i < len; i++) {
      out[i] = i;
    }
    return out;
  }

  template <>
  vector<cytnx_int64> vec_range<cytnx_int64>(const cytnx_int64 &start, const cytnx_int64 &end) {
    cytnx_error_msg(end < start, "[ERROR] cannot have end < start%s", "\n");
    vector<cytnx_int64> out(end - start);
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
    for (cytnx_int64 i = 0; i < end - start; i++) {
      out[i] = start + i;
    }
    return out;
  }

  template <>
  vector<std::string> vec_range<std::string>(const cytnx_int64 &len) {
    vector<std::string> out(len);
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
    for (cytnx_int64 i = 0; i < len; i++) {
      out[i] = to_string(i);
    }
    return out;
  }

  template <>
  vector<std::string> vec_range<std::string>(const cytnx_int64 &start, const cytnx_int64 &end) {
    cytnx_error_msg(end < start, "[ERROR] cannot have end < start%s", "\n");
    vector<std::string> out(end - start);
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
    for (cytnx_int64 i = 0; i < end - start; i++) {
      out[i] = to_string(start + i);
    }
    return out;
  }

  void vec_range_(vector<cytnx_uint64> &v, const cytnx_uint64 &start, const cytnx_uint64 &end) {
    cytnx_error_msg(end < start, "[ERROR] cannot have end < start%s", "\n");
    // v.resize(end - start);
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
    for (cytnx_uint64 i = 0; i < end - start; i++) {
      v[i] = start + i;
    }
  }

  void vec_range_(vector<cytnx_uint64> &v, const cytnx_uint64 &len) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
    for (cytnx_uint64 i = 0; i < len; i++) {
      v[i] = i;
    }
  }

  template <>
  void vec_range_<cytnx_int64>(vector<cytnx_int64> &v, const cytnx_int64 &len) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
    for (cytnx_int64 i = 0; i < len; i++) {
      v[i] = i;
    }
  }

  template <>
  void vec_range_<cytnx_int64>(vector<cytnx_int64> &v, const cytnx_int64 &start,
                               const cytnx_int64 &end) {
    cytnx_error_msg(end < start, "[ERROR] cannot have end < start%s", "\n");
    // v.resize(end - start);
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
    for (cytnx_int64 i = 0; i < end - start; i++) {
      v[i] = start + i;
    }
  }

}  // namespace cytnx
