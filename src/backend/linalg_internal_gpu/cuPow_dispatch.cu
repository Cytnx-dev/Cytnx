#include <cuda/std/complex>
#include <type_traits>

#include "cuPow_internal.hpp"
#include "backend/utils_internal_interface.hpp"

// #1003 step 11: typed GPU dispatch for elementwise Pow(., p), replacing the legacy
// lii.cuPow_ii[dtype] lookup table and its CUDA C complex kernels. Pow is dtype-preserving: the
// caller pre-casts the input to a floating/complex dispatch dtype (integer/bool inputs are promoted
// to Double first) and passes it as both `out` and `in`, so this only ever sees
// ComplexDouble/ComplexFloat/Double/Float. Complex uses cuda::std::pow on the cuda::std::complex
// GPU scalar types.

namespace cytnx {

  namespace linalg_internal {

    namespace {

      template <typename T>
      __device__ inline T CuPowValue(const T &x, const double p) {
        if constexpr (std::is_same_v<T, cytnx_cuda_complex128>) {
          return cuda::std::pow(x, p);
        } else if constexpr (std::is_same_v<T, cytnx_cuda_complex64>) {
          return cuda::std::pow(x, static_cast<float>(p));
        } else {
          return static_cast<T>(cuda::std::pow(x, static_cast<T>(p)));  // real double/float
        }
      }

      template <typename T>
      __global__ void cuPow_dispatch_kernel(T *out, const T *in, const cytnx_uint64 Nelem,
                                            const double p) {
        const cytnx_uint64 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < Nelem) out[idx] = CuPowValue(in[idx], p);
      }

      template <typename T>
      void cuPow_dispatch_typed(boost::intrusive_ptr<Storage_base> &out,
                                const boost::intrusive_ptr<Storage_base> &in,
                                const cytnx_uint64 &Nelem, const double &p) {
        T *_out = reinterpret_cast<T *>(out->data());
        const T *_in = reinterpret_cast<const T *>(in->data());
        cytnx_uint32 NBlocks = Nelem / 512;
        if (Nelem % 512) NBlocks += 1;
        cuPow_dispatch_kernel<<<NBlocks, 512>>>(_out, _in, Nelem, p);
      }

    }  // namespace

    void cuPow_dispatch(boost::intrusive_ptr<Storage_base> &out,
                        const boost::intrusive_ptr<Storage_base> &in, const cytnx_uint64 &Nelem,
                        const double &p) {
      if (Nelem == 0) return;
      switch (in->dtype()) {
        case Type.ComplexDouble:
          cuPow_dispatch_typed<cytnx_cuda_complex128>(out, in, Nelem, p);
          break;
        case Type.ComplexFloat:
          cuPow_dispatch_typed<cytnx_cuda_complex64>(out, in, Nelem, p);
          break;
        case Type.Double:
          cuPow_dispatch_typed<cytnx_double>(out, in, Nelem, p);
          break;
        case Type.Float:
          cuPow_dispatch_typed<cytnx_float>(out, in, Nelem, p);
          break;
        default:
          cytnx_error_msg(true, "[cuPow_dispatch] unsupported dtype: %d%s", in->dtype(), "\n");
      }
    }

  }  // namespace linalg_internal
}  // namespace cytnx
