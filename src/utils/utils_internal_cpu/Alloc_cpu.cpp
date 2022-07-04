#include "utils/utils_internal_cpu/Alloc_cpu.hpp"

using namespace std;

namespace cytnx {
  namespace utils_internal {
    void* Calloc_cpu(const cytnx_uint64& N, const cytnx_uint64& perelem_bytes) {
      void* tmp = calloc(N, perelem_bytes);
      cytnx_error_msg(((tmp == NULL) && (N > 0)), "[ERROR][calloc] Memory allocation failed.%s",
                      "\n");
      return tmp;
    }
    void* Malloc_cpu(const cytnx_uint64& bytes) {
      void* tmp = malloc(bytes);
      cytnx_error_msg(((tmp == NULL) && (bytes > 0)), "[ERROR][malloc] Memory allocation failed.%s",
                      "\n");
      return tmp;
    }
  }  // namespace utils_internal
}  // namespace cytnx
