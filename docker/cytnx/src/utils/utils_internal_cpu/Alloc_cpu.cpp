#include "utils/utils_internal_cpu/Alloc_cpu.hpp"

using namespace std;

namespace cytnx {
  namespace utils_internal {
    void *Calloc_cpu(const cytnx_uint64 &N, const cytnx_uint64 &perelem_bytes) {
      return calloc(N, perelem_bytes);
    }
    void *Malloc_cpu(const cytnx_uint64 &bytes) { return malloc(bytes); }
  }  // namespace utils_internal
}  // namespace cytnx
