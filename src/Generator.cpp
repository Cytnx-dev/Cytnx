#include "Generator.hpp"


namespace cytnx{


    Tensor zeros(const cytnx_uint64 &Nelem, const unsigned int &dtype, const int &device){
        Tensor out({Nelem},dtype,device);
        return out;
    }
    Tensor zeros(const std::vector<cytnx_uint64> &Nelem, const unsigned int &dtype, const int &device){
        Tensor out(Nelem,dtype,device);
        return out;
    }
    Tensor zeros(const std::initializer_list<cytnx_uint64> &Nelem, const unsigned int &dtype, const int &device){
        Tensor out(Nelem,dtype,device);
        return out;
    }


}
