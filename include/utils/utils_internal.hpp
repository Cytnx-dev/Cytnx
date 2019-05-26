#ifndef _H_utils_internal_
#define _H_utils_internal_

#include "utils_internal_cpu/Cast_cpu.hpp"
#include "utils_internal_cpu/Movemem_cpu.hpp"
#include "utils_internal_cpu/Alloc_cpu.hpp"
#include "utils_internal_cpu/Range_cpu.hpp"
#include "complex_arithmic.hpp"

#ifdef UNI_GPU
    #include "utils_internal_gpu/cuCast_gpu.hpp"
    #include "utils_internal_gpu/cuAlloc_gpu.hpp"
    #include "utils_internal_gpu/cuMovemem_gpu.hpp"
#endif


/// Helper function to print vector with ODT:
#include <vector>
#include <iostream>
namespace cytnx{
    template<typename T>
    std::ostream& operator<<(std::ostream& os, const std::vector<T> &vec){
        os << "Vector Print:\n";
        os << "Total Elements:" << vec.size() << std::endl;
        os << "[";
        unsigned long long NBin = vec.size()/10;
        if(vec.size()%10) NBin++;
        for(unsigned long long i=0;i<NBin;i++){
            for(int j=0;j<10;j++){
                if(i*10+j>=vec.size()) break;
                os << vec[i*10+j] << " ";
            }
            if(i==NBin-1) os << "]";
            os << std::endl;
        }
        return os;
    }

}

#endif
