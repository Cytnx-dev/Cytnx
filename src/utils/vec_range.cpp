#include "utils/vec_range.hpp"

namespace cytnx {

  std::vector<cytnx_uint64> vec_range(const cytnx_uint64 &start, const cytnx_uint64 &end) {
    cytnx_error_msg(end < start, "[ERROR] Cannot have end < start%s", "\n");
    std::vector<cytnx_uint64> out(end - start);

    for (cytnx_uint64 i = 0; i < end - start; i++) {
      out[i] = start + i;
    }
    return out;
  }

  std::vector<cytnx_uint64> vec_range(const cytnx_uint64 &len) {
    std::vector<cytnx_uint64> out(len);

    for (cytnx_uint64 i = 0; i < len; i++) {
      out[i] = i;
    }
    return out;
  }

  template <>
  std::vector<cytnx_int64> vec_range<cytnx_int64>(const cytnx_int64 &len) {
    std::vector<cytnx_int64> out(len);
    for (cytnx_int64 i = 0; i < len; i++) {
      out[i] = i;
    }
    return out;
  }

  template <>
  std::vector<cytnx_int64> vec_range<cytnx_int64>(const cytnx_int64 &start,
                                                  const cytnx_int64 &end) {
    cytnx_error_msg(end < start, "[ERROR] Cannot have end < start%s", "\n");
    std::vector<cytnx_int64> out(end - start);
    for (cytnx_int64 i = 0; i < end - start; i++) {
      out[i] = start + i;
    }
    return out;
  }

  template <>
  std::vector<std::string> vec_range<std::string>(const cytnx_int64 &len) {
    std::vector<std::string> out(len);
    for (cytnx_int64 i = 0; i < len; i++) {
      out[i] = std::to_string(i);
    }
    return out;
  }

  template <>
  std::vector<std::string> vec_range<std::string>(const cytnx_int64 &start,
                                                  const cytnx_int64 &end) {
    cytnx_error_msg(end < start, "[ERROR] Cannot have end < start%s", "\n");
    std::vector<std::string> out(end - start);
    for (cytnx_int64 i = 0; i < end - start; i++) {
      out[i] = std::to_string(start + i);
    }
    return out;
  }

  void vec_range_(std::vector<cytnx_uint64> &v, const cytnx_uint64 &start,
                  const cytnx_uint64 &end) {
    cytnx_error_msg(end < start, "[ERROR] Cannot have end < start%s", "\n");
    // v.resize(end - start);
    for (cytnx_uint64 i = 0; i < end - start; i++) {
      v[i] = start + i;
    }
  }

  void vec_range_(std::vector<cytnx_uint64> &v, const cytnx_uint64 &len) {
    for (cytnx_uint64 i = 0; i < len; i++) {
      v[i] = i;
    }
  }

  template <>
  void vec_range_<cytnx_int64>(std::vector<cytnx_int64> &v, const cytnx_int64 &len) {
    for (cytnx_int64 i = 0; i < len; i++) {
      v[i] = i;
    }
  }

  template <>
  void vec_range_<cytnx_int64>(std::vector<cytnx_int64> &v, const cytnx_int64 &start,
                               const cytnx_int64 &end) {
    cytnx_error_msg(end < start, "[ERROR] Cannot have end < start%s", "\n");
    // v.resize(end - start);
    for (cytnx_int64 i = 0; i < end - start; i++) {
      v[i] = start + i;
    }
  }

}  // namespace cytnx
