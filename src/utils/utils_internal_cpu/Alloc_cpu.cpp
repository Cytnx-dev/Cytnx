#include "utils/utils_internal_cpu/Alloc_cpu.hpp"

using namespace std;

namespace tor10{
    namespace utils_internal{
        void* Calloc_cpu(const tor10_uint64 &N, const tor10_uint64 &perelem_bytes){
            return calloc(N,perelem_bytes);
        }
        void* Malloc_cpu(const tor10_uint64 &bytes){
            return malloc(bytes);
        }
    }
}
