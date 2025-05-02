#include "Range_cpu.hpp"

using namespace std;
namespace cytnx {
  namespace utils_internal {

    vector<cytnx_uint64> range_cpu(const cytnx_uint64 &start, const cytnx_uint64 &end) {
      cytnx_error_msg(end < start, "[ERROR] cannot have end < start%s", "\n");
      vector<cytnx_uint64> out(end - start);

      for (cytnx_uint64 i = 0; i < end - start; i++) {
        out[i] = start + i;
      }
      return out;
    }

    vector<cytnx_uint64> range_cpu(const cytnx_uint64 &len) {
      vector<cytnx_uint64> out(len);

      for (cytnx_uint64 i = 0; i < len; i++) {
        out[i] = i;
      }
      return out;
    }

    template <>
    vector<cytnx_int64> range_cpu<cytnx_int64>(const cytnx_int64 &len) {
      vector<cytnx_int64> out(len);
      for (cytnx_int64 i = 0; i < len; i++) {
        out[i] = i;
      }
      return out;
    }

    template <>
    vector<cytnx_int64> range_cpu<cytnx_int64>(const cytnx_int64 &start, const cytnx_int64 &end) {
      cytnx_error_msg(end < start, "[ERROR] cannot have end < start%s", "\n");
      vector<cytnx_int64> out(end - start);
      for (cytnx_int64 i = 0; i < end - start; i++) {
        out[i] = start + i;
      }
      return out;
    }

  }  // namespace utils_internal
}  // namespace cytnx
