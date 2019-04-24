#ifndef _H_cuAlloc_gpu_
#define _H_cuAlloc_gpu_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "tor10_error.hpp"
namespace tor10{
    namespace utils_internal{

    #ifdef UNI_GPU    
        //void* Calloc_gpu(const tor10_uint64&N, const tor10_uint64 &perelem_bytes);
        void* cuMalloc_gpu(const tor10_uint64 &bytes);
    #endif
    }
}
#endif
