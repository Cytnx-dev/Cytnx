#include "utils/utils_internal_cpu/Range_cpu.hpp"
#ifdef UNI_OMP
#include <omp.h>
#endif

using namespace std;
namespace cytnx{
    namespace utils_internal{
        vector<cytnx_uint64> range_cpu(const cytnx_uint64 &len){
            vector<cytnx_uint64> out(len);

            #ifdef UNI_OMP
            #pragma omp parallel for schedule(dynamic)
            #endif
            for(cytnx_uint64 i=0;i<len;i++){
                out[i] = i;
            }
            return out;
        }

    }//namespace utils_internal
}//namespace cytnx
