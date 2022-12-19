#include "Concate_internal.hpp"
#include "algo/algo_internal_interface.hpp"
#include <iostream>
#include <algorithm>

#ifdef UNI_OMP
  #include <omp.h>
#endif

namespace cytnx {

  namespace algo_internal {

    void Concate_internal(char *out_ptr, std::vector<void*> &ins, const std::vector<cytnx_uint64> &lens, const cytnx_uint64 &ElemSize) {

        // require:
        // 1. Data type of out, all [ins] to be the same
        // 2. out is properly allocated! 
        // 3. size is deref from out.type!
        // 4. checking bool type!!

        //cytnx_uint64 ElemSize = Type.typeSize(out->dtype);
        cytnx_uint64 offs = 0;
        //char *out_ptr = (char*)out->Mem;

        for(cytnx_int32 i=0;i<ins.size();i++){
            memcpy(out_ptr + offs,ins[i],ElemSize*lens[i]);
            offs += ElemSize*lens[i];
            //std::cout << ElemSize*lens[i] << std::endl;
        }
        
    }

  }  // namespace algo_internal

}  // namespace cytnx
