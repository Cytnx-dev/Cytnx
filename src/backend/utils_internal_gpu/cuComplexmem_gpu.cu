#include "cuComplexmem_gpu.hpp"
#include "cuAlloc_gpu.hpp"
#include "backend/Storage.hpp"
#ifdef UNI_OMP
  #include <omp.h>
#endif

using namespace std;

namespace cytnx {
  namespace utils_internal {
#ifdef UNI_GPU

    void cuComplexmem_gpu_cdtd(void *out, void *in, const cytnx_uint64 &Nelem,
                               const bool get_real) {
      double *ddes = (double *)out;
      double *dsrc = (double *)in;  // we cast into double, so the Memcpy2D can get elem by stride.

      if (get_real) {
        cudaMemcpy2D(ddes, 1 * sizeof(cytnx_double), dsrc, 2 * sizeof(cytnx_double),
                     sizeof(cytnx_double), Nelem, cudaMemcpyDeviceToDevice);
      } else {
        cudaMemcpy2D(ddes, 1 * sizeof(cytnx_double), dsrc + 1, 2 * sizeof(cytnx_double),
                     sizeof(cytnx_double), Nelem, cudaMemcpyDeviceToDevice);
      }
    }

    void cuComplexmem_gpu_cftf(void *out, void *in, const cytnx_uint64 &Nelem,
                               const bool get_real) {
      float *ddes = (float *)out;
      float *dsrc = (float *)in;  // we cast into double, so the Memcpy2D can get elem by stride.

      if (get_real) {
        cudaMemcpy2D(ddes, 1 * sizeof(cytnx_float), dsrc, 2 * sizeof(cytnx_float),
                     sizeof(cytnx_float), Nelem, cudaMemcpyDeviceToDevice);
      } else {
        cudaMemcpy2D(ddes, 1 * sizeof(cytnx_float), dsrc + 1, 2 * sizeof(cytnx_float),
                     sizeof(cytnx_float), Nelem, cudaMemcpyDeviceToDevice);
      }
    }

#endif  // UNI_GPU
  }  // namespace utils_internal
}  // namespace cytnx
