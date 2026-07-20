#include <cuda/std/complex>

#include "cuConj_inplace_internal.hpp"
#include "backend/utils_internal_interface.hpp"

// #1003 step 11: typed GPU dispatch for the in-place elementwise Conj, replacing the legacy
// lii.cuConj_inplace_ii[dtype] lookup table and its CUDA C complex kernels. The front end only
// calls this for a complex dtype (real Conj is a no-op), so only ComplexDouble/ComplexFloat are
// handled. Uses cuda::std::conj on the cuda::std::complex GPU scalar types.

namespace cytnx {

  namespace linalg_internal {

    namespace {

      template <typename T>
      __global__ void cuConj_inplace_dispatch_kernel(T *data, const cytnx_uint64 Nelem) {
        const cytnx_uint64 idx = static_cast<cytnx_uint64>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx < Nelem) data[idx] = cuda::std::conj(data[idx]);
      }

      template <typename T>
      void cuConj_inplace_dispatch_typed(boost::intrusive_ptr<Storage_base> &inout,
                                         cytnx_uint64 Nelem) {
        T *_data = reinterpret_cast<T *>(inout->data());
        const cytnx_uint32 NBlocks = (Nelem + 511) / 512;
        cuConj_inplace_dispatch_kernel<<<NBlocks, 512>>>(_data, Nelem);
      }

    }  // namespace

    void cuConj_inplace_dispatch(boost::intrusive_ptr<Storage_base> &inout, cytnx_uint64 Nelem) {
      if (Nelem == 0) return;
      switch (inout->dtype()) {
        case Type.ComplexDouble:
          cuConj_inplace_dispatch_typed<cytnx_cuda_complex128>(inout, Nelem);
          break;
        case Type.ComplexFloat:
          cuConj_inplace_dispatch_typed<cytnx_cuda_complex64>(inout, Nelem);
          break;
        default:
          cytnx_error_msg(true, "[cuConj_inplace_dispatch] unsupported dtype (complex only): %d%s",
                          inout->dtype(), "\n");
      }
    }

  }  // namespace linalg_internal
}  // namespace cytnx
