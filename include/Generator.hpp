#ifndef _Generator_H_
#define _Generator_H_
#include "Type.hpp"
#include "Device.hpp"
#include "cytnx_error.hpp"
#include "Tensor.hpp"
#include <vector>
namespace cytnx{

    Tensor zeros(const cytnx_uint64 &Nelem, const unsigned int &dtype=cytnxtype.Double, const int &device=cytnxdevice.cpu);
    Tensor zeros(const std::vector<cytnx_uint64> &Nelem, const unsigned int &dtype=cytnxtype.Double, const int &device=cytnxdevice.cpu);
    Tensor zeros(const std::initializer_list<cytnx_uint64> &Nelem, const unsigned int &dtype=cytnxtype.Double, const int &device=cytnxdevice.cpu);
}

#endif
