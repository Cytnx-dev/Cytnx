#ifndef _H_utils_internal_
#define _H_utils_internal_

#include "utils_internal_cpu/Cast_cpu.hpp"
#include "utils_internal_cpu/Movemem_cpu.hpp"
#include "utils_internal_cpu/Alloc_cpu.hpp"
#include "utils_internal_cpu/SetZeros_cpu.hpp"
#include "utils_internal_cpu/Range_cpu.hpp"
#include "complex_arithmic.hpp"

#ifdef UNI_GPU
    #include "utils_internal_gpu/cuCast_gpu.hpp"
    #include "utils_internal_gpu/cuAlloc_gpu.hpp"
    #include "utils_internal_gpu/cuMovemem_gpu.hpp"
    #include "utils_internal_gpu/cuSetZeros_gpu.hpp"
#endif


#endif
