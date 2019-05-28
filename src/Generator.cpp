#include "Generator.hpp"


namespace cytnx{


    Tensor arange(const cytnx_uint64 &Nelem, const unsigned int &dtype=cytnxtype.Double, const int &device = cytnxdevice.cpu){
        cytnx_error_msg(Nelem==0,"[ERROR] %s","[ararnge] Nelem must > 0\n");

        Tensor out({Nelem},dtype, device);
        #ifdef UNI_OMP
        #endif
        for(cytnx_uint64 i=0;i<Nelem;i++){
            
        }
        return out;
    }
    Tensor arange(const cytnx_uint64 &min, const cytnx_uint64 &max, const unsigned int &dtype=cytnxtype.Double, const int &device = cytnxdevice.cpu)



    Tensor ones(const cytnx_uint64 &Nelem, const unsigned int &dtype, const int &device)
    Tensor ones(const std::vector<cytnx_uint64> &size, const unsigned int &dtype);
    Tensor ones(const std::initalizer_list<cytnx_uint64> &size, const unsigned int &dtype, const int &device);

}
