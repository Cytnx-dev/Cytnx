#ifndef _H_Range_cpu_
#define _H_Range_cpu_

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <climits>
#include "Type.hpp"
#include "Storage.hpp"
#include "tor10_error.hpp"

namespace tor10{
    namespace utils_internal{
        std::vector<tor10_uint64> range_cpu(const tor10_uint64 &len);
    }
}
#endif
