#include "backend/utils_internal_gpu/cuFill_gpu.hpp"

#include <complex>

#include "cuda/std/complex"

#include "Type.hpp"

namespace cytnx {
  namespace utils_internal {

    template <class CudaDType>
    __global__ void FillGpuKernel(CudaDType* first, CudaDType value, cytnx_uint64 count) {
      if (blockIdx.x * blockDim.x + threadIdx.x < count) {
        first[blockIdx.x * blockDim.x + threadIdx.x] = value;
      }
    }

    template <typename DType>
    struct ToCudaDType {
      typedef DType type;
    };

    template <typename DType>
    struct ToCudaDType<std::complex<DType>> {
      typedef cuda::std::complex<DType> type;
    };

    template <typename DType>
    void FillGpu(void* first, const DType& value, cytnx_uint64 count) {
      using CudaDType = typename ToCudaDType<DType>::type;

      CudaDType* typed_first = reinterpret_cast<CudaDType*>(first);
      cytnx_uint64 block_count = (count + 511) / 512;
      FillGpuKernel<<<block_count, 512>>>(typed_first, static_cast<CudaDType>(value), count);
    }

    template void FillGpu<cytnx_complex128>(void*, const cytnx_complex128&, cytnx_uint64);
    template void FillGpu<cytnx_complex64>(void*, const cytnx_complex64&, cytnx_uint64);
    template void FillGpu<cytnx_double>(void*, const cytnx_double&, cytnx_uint64);
    template void FillGpu<cytnx_float>(void*, const cytnx_float&, cytnx_uint64);
    template void FillGpu<cytnx_uint64>(void*, const cytnx_uint64&, cytnx_uint64);
    template void FillGpu<cytnx_int64>(void*, const cytnx_int64&, cytnx_uint64);
    template void FillGpu<cytnx_uint32>(void*, const cytnx_uint32&, cytnx_uint64);
    template void FillGpu<cytnx_int32>(void*, const cytnx_int32&, cytnx_uint64);
    template void FillGpu<cytnx_uint16>(void*, const cytnx_uint16&, cytnx_uint64);
    template void FillGpu<cytnx_int16>(void*, const cytnx_int16&, cytnx_uint64);
    template void FillGpu<cytnx_bool>(void*, const cytnx_bool&, cytnx_uint64);

  }  // namespace utils_internal
}  // namespace cytnx
