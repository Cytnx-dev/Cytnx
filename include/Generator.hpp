#ifndef __Generator_H_
#define __Generator_H_

#include "Tensor.hpp"
#include "Type.hpp"
#include <vector>

namespace cytnx{

    Tensor arange(const cytnx_uint64 &Nelem, const unsigned int &dtype=cytnxtype.Double, const int &device = cytnxdevice.cpu);
    Tensor arange(const std::vector<cytnx_uint64> &size, const unsigned int &dtype=cytnxtype.Double);
    Tensor arange(const std::initalizer_list<cytnx_uint64> &size, const unsigned int &dtype=cytnxtype.Double, const int &device = cytnxdevice.cpu);


}


#endif
