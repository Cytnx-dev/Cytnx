#include "Generator.hpp"


namespace cytnx{

    // degraded from initializer_list? 
    Tensor zeros(const cytnx_uint64 &Nelem, const unsigned int &dtype, const int &device){
        Tensor out({Nelem},dtype,device); // the default
        out._impl->storage().set_zeros(); 
        return out;
    }
    Tensor zeros(const std::vector<cytnx_uint64> &Nelem, const unsigned int &dtype, const int &device){
        Tensor out(Nelem,dtype,device);
        out._impl->storage().set_zeros(); 
        return out;
    }
    Tensor zeros(const std::initializer_list<cytnx_uint64> &Nelem, const unsigned int &dtype, const int &device){
        Tensor out(Nelem,dtype,device);
        out._impl->storage().set_zeros(); 
        return out;
    }


}
