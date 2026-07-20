#include <cuda/std/cmath>
#include <cuda/std/complex>
#include <cuda/std/cstdlib>
#include <type_traits>

#include "cuAbs_internal.hpp"
#include "backend/utils_internal_interface.hpp"

// #1003 step 11: typed GPU dispatch for the elementwise Abs, replacing the legacy
// lii.cuAbs_ii[dtype] lookup table and its cuDoubleComplex / cuCabs kernels. The operation-specific
// output dtype rule Abs(complex) -> real (ComplexDouble -> Double, ComplexFloat -> Float; every
// other dtype maps to itself) is encoded in AbsOutput below; the caller allocates `out` with that
// dtype and this dispatch verifies it. Complex magnitude uses cuda::std::abs on the
// cuda::std::complex GPU scalar types.

namespace cytnx {

  namespace linalg_internal {

    namespace {

      template <typename TIn>
      struct AbsOutput {
        using type = TIn;
      };
      template <>
      struct AbsOutput<cytnx_cuda_complex128> {
        using type = cytnx_double;
      };
      template <>
      struct AbsOutput<cytnx_cuda_complex64> {
        using type = cytnx_float;
      };

      // Abs magnitude. cuda::std::abs is correct for every non-unsigned type: complex -> real
      // magnitude, signed integer -> |x|, and floating -> |x| with the right sign for -0.0/NaN
      // (unlike `x < 0 ? -x : x`, since -0.0 < 0 is false). Only the unsigned integer / bool
      // types, for which abs is the identity, take the fallback.
      template <typename TIn>
      __device__ inline auto cu_abs_value(const TIn &x) {
        if constexpr (std::is_unsigned_v<TIn>) {
          return x;
        } else {
          return cuda::std::abs(x);
        }
      }

      template <typename TOut, typename TIn>
      __global__ void cuAbs_dispatch_kernel(TOut *out, const TIn *in, const cytnx_uint64 Nelem) {
        const cytnx_uint64 idx = static_cast<cytnx_uint64>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx < Nelem) out[idx] = static_cast<TOut>(cu_abs_value(in[idx]));
      }

      template <typename TIn>
      void cuAbs_dispatch_typed(boost::intrusive_ptr<Storage_base> &out,
                                const boost::intrusive_ptr<Storage_base> &in, cytnx_uint64 Nelem) {
        using TOut = typename AbsOutput<TIn>::type;
        cytnx_error_msg(out->dtype() != Type_class::cy_typeid_gpu_v<TOut>,
                        "[cuAbs_dispatch] output dtype mismatch. got=%d expected=%d%s",
                        out->dtype(), Type_class::cy_typeid_gpu_v<TOut>, "\n");

        TOut *_out = reinterpret_cast<TOut *>(out->data());
        const TIn *_in = reinterpret_cast<const TIn *>(in->data());

        const cytnx_uint32 NBlocks = (Nelem + 511) / 512;
        cuAbs_dispatch_kernel<<<NBlocks, 512>>>(_out, _in, Nelem);
      }

    }  // namespace

    void cuAbs_dispatch(boost::intrusive_ptr<Storage_base> &out,
                        const boost::intrusive_ptr<Storage_base> &in, cytnx_uint64 Nelem) {
      if (Nelem == 0) return;
      switch (in->dtype()) {
        case Type.ComplexDouble:
          cuAbs_dispatch_typed<cytnx_cuda_complex128>(out, in, Nelem);
          break;
        case Type.ComplexFloat:
          cuAbs_dispatch_typed<cytnx_cuda_complex64>(out, in, Nelem);
          break;
        case Type.Double:
          cuAbs_dispatch_typed<cytnx_double>(out, in, Nelem);
          break;
        case Type.Float:
          cuAbs_dispatch_typed<cytnx_float>(out, in, Nelem);
          break;
        case Type.Int64:
          cuAbs_dispatch_typed<cytnx_int64>(out, in, Nelem);
          break;
        case Type.Uint64:
          cuAbs_dispatch_typed<cytnx_uint64>(out, in, Nelem);
          break;
        case Type.Int32:
          cuAbs_dispatch_typed<cytnx_int32>(out, in, Nelem);
          break;
        case Type.Uint32:
          cuAbs_dispatch_typed<cytnx_uint32>(out, in, Nelem);
          break;
        case Type.Int16:
          cuAbs_dispatch_typed<cytnx_int16>(out, in, Nelem);
          break;
        case Type.Uint16:
          cuAbs_dispatch_typed<cytnx_uint16>(out, in, Nelem);
          break;
        case Type.Bool:
          cuAbs_dispatch_typed<cytnx_bool>(out, in, Nelem);
          break;
        default:
          cytnx_error_msg(true, "[cuAbs_dispatch] unsupported dtype: %d%s", in->dtype(), "\n");
      }
    }

  }  // namespace linalg_internal
}  // namespace cytnx
