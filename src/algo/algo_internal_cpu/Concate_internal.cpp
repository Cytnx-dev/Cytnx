#include "Concate_internal.hpp"
#include "algo/algo_internal_interface.hpp"
#include <iostream>
#include <algorithm>

#ifdef UNI_OMP
  #include <omp.h>
#endif

namespace cytnx {

  namespace algo_internal {

    void Concate_internal(boost::intrusive_ptr<Storage_base> &out, std::vector<boost::intrusive_ptr<Storage_base> > &ins) {

        // require:
        // 1. Data type of out, all [ins] to be the same
        // 2. out is properly allocated! 
        // 3. size is deref from out.type!
        // 4. checking bool type!!

        cytnx_uint64 ElemSize = Type.typeSize(out->dtype);
        cytnx_uint64 offs = 0;
        char *out_ptr = (char*)out->Mem;

        for(cytnx_int32 i=0;i<ins.size();i++){
            memcpy(out_ptr + offs,ins[i]->Mem,ElemSize*ins[i]->len);
            offs += ElemSize*ins[i]->len;
        }
        
    }

  }  // namespace algo_internal

}  // namespace cytnx
