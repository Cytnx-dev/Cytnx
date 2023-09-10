#include "cuFill_gpu.hpp"
#include "backend/Storage.hpp"
#ifdef UNI_OMP
  #include <omp.h>
#endif

using namespace std;
namespace cytnx {
  namespace utils_internal {

    template <class T3>
    __global__ void cuFill_kernel(T3* des, T3 val, cytnx_uint64 Nelem) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        des[blockIdx.x * blockDim.x + threadIdx.x] = val;
      }
    }

    //========================================================================
    void cuFill_gpu_cd(void* in, void* val, const cytnx_uint64& Nelem) {
      cuDoubleComplex* ptr = (cuDoubleComplex*)in;
      cuDoubleComplex _val = *((cuDoubleComplex*)val);

      cytnx_uint64 NBlocks = Nelem / 512;
      if (Nelem % 512) NBlocks += 1;
      cuFill_kernel<<<NBlocks, 512>>>(ptr, _val, Nelem);
    }

    void cuFill_gpu_cf(void* in, void* val, const cytnx_uint64& Nelem) {
      cuFloatComplex* ptr = (cuFloatComplex*)in;
      cuFloatComplex _val = *((cuFloatComplex*)val);

      cytnx_uint64 NBlocks = Nelem / 512;
      if (Nelem % 512) NBlocks += 1;
      cuFill_kernel<<<NBlocks, 512>>>(ptr, _val, Nelem);
    }

    void cuFill_gpu_d(void* in, void* val, const cytnx_uint64& Nelem) {
      cytnx_double* ptr = (cytnx_double*)in;
      cytnx_double _val = *((cytnx_double*)val);

      cytnx_uint64 NBlocks = Nelem / 512;
      if (Nelem % 512) NBlocks += 1;
      cuFill_kernel<<<NBlocks, 512>>>(ptr, _val, Nelem);
    }

    void cuFill_gpu_f(void* in, void* val, const cytnx_uint64& Nelem) {
      cytnx_float* ptr = (cytnx_float*)in;
      cytnx_float _val = *((cytnx_float*)val);

      cytnx_uint64 NBlocks = Nelem / 512;
      if (Nelem % 512) NBlocks += 1;
      cuFill_kernel<<<NBlocks, 512>>>(ptr, _val, Nelem);
    }

    void cuFill_gpu_i64(void* in, void* val, const cytnx_uint64& Nelem) {
      cytnx_int64* ptr = (cytnx_int64*)in;
      cytnx_int64 _val = *((cytnx_int64*)val);

      cytnx_uint64 NBlocks = Nelem / 512;
      if (Nelem % 512) NBlocks += 1;
      cuFill_kernel<<<NBlocks, 512>>>(ptr, _val, Nelem);
    }

    void cuFill_gpu_u64(void* in, void* val, const cytnx_uint64& Nelem) {
      cytnx_uint64* ptr = (cytnx_uint64*)in;
      cytnx_uint64 _val = *((cytnx_uint64*)val);

      cytnx_uint64 NBlocks = Nelem / 512;
      if (Nelem % 512) NBlocks += 1;
      cuFill_kernel<<<NBlocks, 512>>>(ptr, _val, Nelem);
    }

    void cuFill_gpu_i32(void* in, void* val, const cytnx_uint64& Nelem) {
      cytnx_int32* ptr = (cytnx_int32*)in;
      cytnx_int32 _val = *((cytnx_int32*)val);

      cytnx_uint64 NBlocks = Nelem / 512;
      if (Nelem % 512) NBlocks += 1;
      cuFill_kernel<<<NBlocks, 512>>>(ptr, _val, Nelem);
    }

    void cuFill_gpu_u32(void* in, void* val, const cytnx_uint64& Nelem) {
      cytnx_uint32* ptr = (cytnx_uint32*)in;
      cytnx_uint32 _val = *((cytnx_uint32*)val);

      cytnx_uint64 NBlocks = Nelem / 512;
      if (Nelem % 512) NBlocks += 1;
      cuFill_kernel<<<NBlocks, 512>>>(ptr, _val, Nelem);
    }

    void cuFill_gpu_i16(void* in, void* val, const cytnx_uint64& Nelem) {
      cytnx_int16* ptr = (cytnx_int16*)in;
      cytnx_int16 _val = *((cytnx_int16*)val);

      cytnx_uint64 NBlocks = Nelem / 512;
      if (Nelem % 512) NBlocks += 1;
      cuFill_kernel<<<NBlocks, 512>>>(ptr, _val, Nelem);
    }

    void cuFill_gpu_u16(void* in, void* val, const cytnx_uint64& Nelem) {
      cytnx_uint16* ptr = (cytnx_uint16*)in;
      cytnx_uint16 _val = *((cytnx_uint16*)val);

      cytnx_uint64 NBlocks = Nelem / 512;
      if (Nelem % 512) NBlocks += 1;
      cuFill_kernel<<<NBlocks, 512>>>(ptr, _val, Nelem);
    }
    void cuFill_gpu_b(void* in, void* val, const cytnx_uint64& Nelem) {
      cytnx_bool* ptr = (cytnx_bool*)in;
      cytnx_bool _val = *((cytnx_bool*)val);

      cytnx_uint64 NBlocks = Nelem / 512;
      if (Nelem % 512) NBlocks += 1;
      cuFill_kernel<<<NBlocks, 512>>>(ptr, _val, Nelem);
    }
  }  // namespace utils_internal
}  // namespace cytnx
