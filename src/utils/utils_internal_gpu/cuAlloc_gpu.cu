#include "utils/utils_internal_gpu/cuAlloc_gpu.hpp"

using namespace std;

namespace tor10{
    namespace utils_internal{
    #ifdef UNI_GPU
        //void* Calloc_cpu(const tor10_uint64 &N, const tor10_uint64 &perelem_bytes){
        //    return calloc(M,perelem_bytes);
        //}
        void* cuMalloc_gpu(const tor10_uint64 &bytes){
            void* ptr;
            checkCudaErrors(cudaMallocManaged(&ptr,bytes));
            return ptr;
        }
    #endif
    }//utils_internal
}//tor10
