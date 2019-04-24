#ifndef _H_Alloc_cpu_
#define _H_Alloc_cpu_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "tor10_error.hpp"

namespace tor10{
    namespace utils_internal{
        
        void* Calloc_cpu(const tor10_uint64&N, const tor10_uint64 &perelem_bytes);
        void* Malloc_cpu(const tor10_uint64&bytes);

    }
}
#endif
