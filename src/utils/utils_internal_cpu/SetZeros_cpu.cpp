#include "SetZeros_cpu.hpp"

using namespace std;

namespace cytnx {
  namespace utils_internal {
    void SetZeros(void* c_ptr, const cytnx_uint64& bytes) { memset(c_ptr, 0, bytes); }
  }  // namespace utils_internal
}  // namespace cytnx
