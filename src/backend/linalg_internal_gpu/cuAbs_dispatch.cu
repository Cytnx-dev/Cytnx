#include <cuda/std/complex>
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

      template <typename TIn>
      __device__ inline auto CuAbsValue(const TIn &x) {
        if constexpr (std::is_same_v<TIn, cytnx_cuda_complex128> ||
                      std::is_same_v<TIn, cytnx_cuda_complex64>) {
          return cuda::std::abs(x);  // magnitude: complex<T> -> T
        } else if constexpr (std::is_signed_v<TIn>) {
          return x < TIn(0) ? static_cast<TIn>(-x) : x;  // fabs / abs for signed real & integer
        } else {
          return x;  // unsigned / bool: abs is the identity
        }
      }

      template <typename TOut, typename TIn>
      __global__ void cuAbs_dispatch_kernel(TOut *out, const TIn *in, const cytnx_uint64 Nelem) {
        const cytnx_uint64 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < Nelem) out[idx] = static_cast<TOut>(CuAbsValue(in[idx]));
      }

      template <typename TIn>
      void cuAbs_dispatch_typed(boost::intrusive_ptr<Storage_base> &out,
                                const boost::intrusive_ptr<Storage_base> &in,
                                const cytnx_uint64 &Nelem) {
        using TOut = typename AbsOutput<TIn>::type;
        cytnx_error_msg(out->dtype() != Type_class::cy_typeid_gpu_v<TOut>,
                        "[cuAbs_dispatch] output dtype mismatch. got=%d expected=%d%s",
                        out->dtype(), Type_class::cy_typeid_gpu_v<TOut>, "\n");

        TOut *_out = reinterpret_cast<TOut *>(out->data());
        const TIn *_in = reinterpret_cast<const TIn *>(in->data());

        cytnx_uint32 NBlocks = Nelem / 512;
        if (Nelem % 512) NBlocks += 1;
        cuAbs_dispatch_kernel<<<NBlocks, 512>>>(_out, _in, Nelem);
      }

    }  // namespace

    void cuAbs_dispatch(boost::intrusive_ptr<Storage_base> &out,
                        const boost::intrusive_ptr<Storage_base> &in, const cytnx_uint64 &Nelem) {
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
