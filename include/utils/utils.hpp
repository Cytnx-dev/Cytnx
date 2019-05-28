#ifndef __utils_H_
#define __utils_H_

#include "complex_arithmic.hpp"
#include "is.hpp"

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

    Tensor zeros(const cytnx_uint64 &Nelem, const unsigned int &dtype=cytnxtype.Double, const int &device=cytnxdevice.cpu);


}
#endif
