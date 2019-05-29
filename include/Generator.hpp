#ifndef _Generator_H_
#define _Generator_H_
#include "Type.hpp"
#include "Device.hpp"
#include "cytnx_error.hpp"
#include "Tensor.hpp"
#include <vector>
#include <initializer_list>
namespace cytnx{

    
    Tensor zeros(const cytnx_uint64 &Nelem, const unsigned int &dtype=Type.Double, const int &device=Device.cpu);
    Tensor zeros(const std::vector<cytnx_uint64> &Nelem, const unsigned int &dtype=Type.Double, const int &device=Device.cpu);
    Tensor zeros(const std::initializer_list<cytnx_uint64> &Nelem, const unsigned int &dtype=Type.Double, const int &device=Device.cpu);
    
    Tensor ones(const cytnx_uint64 &Nelem, const unsigned int &dtype=Type.Double, const int &device=Device.cpu);
    Tensor ones(const std::vector<cytnx_uint64> &Nelem, const unsigned int &dtype=Type.Double, const int &device=Device.cpu);
    Tensor ones(const std::initializer_list<cytnx_uint64> &Nelem, const unsigned int &dtype=Type.Double, const int &device=Device.cpu);

    Tensor arange(const cytnx_int64 &Nelem, const unsigned int &dtype=Type.Double, const int &device=Device.cpu);
    Tensor arange(const cytnx_double &start, const cytnx_double &end, const cytnx_double &step, const unsigned int &dtype=Type.Double, const int &device=Device.cpu);
 
}

#endif
