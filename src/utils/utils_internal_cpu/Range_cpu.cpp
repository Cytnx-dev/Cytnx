#include "utils/utils_internal_cpu/Range_cpu.hpp"
#ifdef UNI_OMP
#include <omp.h>
#endif

using namespace std;
namespace tor10{
    namespace utils_internal{
        vector<tor10_uint64> range_cpu(const tor10_uint64 &len){
            vector<tor10_uint64> out(len);
            #ifdef UNI_OMP
            #pragma omp parallel for
            #endif
            for(tor10_uint64 i=0;i<len;i++){
                out[i] = i;
            }
            return out;
        }

    }//namespace utils_internal
}//namespace tor10
