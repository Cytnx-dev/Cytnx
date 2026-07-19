#include <cuda/std/complex>

#include "cuExp_internal.hpp"
#include "backend/utils_internal_interface.hpp"

// #1003 step 11: typed GPU dispatch for elementwise Exp, replacing the legacy lii.cuExp_ii[dtype]
// lookup table and its CUDA C complex kernels. Exp is dtype-preserving: the caller pre-casts the
// input to a floating/complex dispatch dtype (integer/bool inputs are promoted to Double first) and
// passes it as both `out` and `in`, so this only ever sees ComplexDouble/ComplexFloat/Double/Float.
// Complex uses cuda::std::exp on the cuda::std::complex GPU scalar types.

namespace cytnx {

  namespace linalg_internal {

    namespace {

      template <typename T>
      __device__ inline T CuExpValue(const T &x) {
        return cuda::std::exp(x);  // T -> T for real (double/float) and complex
      }

      template <typename T>
      __global__ void cuExp_dispatch_kernel(T *out, const T *in, const cytnx_uint64 Nelem) {
        const cytnx_uint64 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < Nelem) out[idx] = CuExpValue(in[idx]);
      }

      template <typename T>
      void cuExp_dispatch_typed(boost::intrusive_ptr<Storage_base> &out,
                                const boost::intrusive_ptr<Storage_base> &in,
                                const cytnx_uint64 &Nelem) {
        T *_out = reinterpret_cast<T *>(out->data());
        const T *_in = reinterpret_cast<const T *>(in->data());
        cytnx_uint32 NBlocks = Nelem / 512;
        if (Nelem % 512) NBlocks += 1;
        cuExp_dispatch_kernel<<<NBlocks, 512>>>(_out, _in, Nelem);
      }

    }  // namespace

    void cuExp_dispatch(boost::intrusive_ptr<Storage_base> &out,
                        const boost::intrusive_ptr<Storage_base> &in, const cytnx_uint64 &Nelem) {
      if (Nelem == 0) return;
      switch (in->dtype()) {
        case Type.ComplexDouble:
          cuExp_dispatch_typed<cytnx_cuda_complex128>(out, in, Nelem);
          break;
        case Type.ComplexFloat:
          cuExp_dispatch_typed<cytnx_cuda_complex64>(out, in, Nelem);
          break;
        case Type.Double:
          cuExp_dispatch_typed<cytnx_double>(out, in, Nelem);
          break;
        case Type.Float:
          cuExp_dispatch_typed<cytnx_float>(out, in, Nelem);
          break;
        default:
          cytnx_error_msg(true, "[cuExp_dispatch] unsupported dtype: %d%s", in->dtype(), "\n");
      }
    }

  }  // namespace linalg_internal
}  // namespace cytnx
