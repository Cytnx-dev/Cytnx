#include <cuda.h>
#include <device_launch_parameters.h>
#include <variant>
#include "cuOuter_internal.hpp"
#include "cuGer_internal.hpp"
#include "backend/utils_internal_interface.hpp"

namespace cytnx {

  namespace linalg_internal {

    //====================================================================

    template <class T1, class T2, class T3>
    __global__ void cuOuter_kernel(T1 *out, const T2 *val, const cytnx_uint64 Nelem,
                                   const cytnx_uint64 OffL, const T3 *ptr) {
      if (blockIdx.x * blockDim.x + threadIdx.x < Nelem) {
        // Promote both operands to the output type T1 before multiplying:
        // cuda::std::complex has no mixed-type operator* (e.g.
        // complex<float> * double is ill-formed), and casting to T1 first
        // yields the same value the promoted output dtype expects.
        out[blockIdx.x * blockDim.x + threadIdx.x] =
          T1(val[cytnx_uint64((blockIdx.x * blockDim.x + threadIdx.x) / OffL)]) *
          T1(ptr[(blockIdx.x * blockDim.x + threadIdx.x) % OffL]);
      }
      __syncthreads();
    }

    template <class T1, class T2, class T3>
    void cuOuter_general(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &Lin,
                         const boost::intrusive_ptr<Storage_base> &Rin, const cytnx_uint64 &j1,
                         const cytnx_uint64 &j2) {
      T1 *_out = (T1 *)out->data();
      T2 *_Lin = (T2 *)Lin->data();
      T3 *_Rin = (T3 *)Rin->data();

      cytnx_uint64 Nelem = Lin->size() * Rin->size();
      cytnx_uint32 NBlocks = Nelem / 512;
      if (Nelem % 512) NBlocks += 1;

      cuOuter_kernel<<<NBlocks, 512>>>(_out, _Lin, Nelem, j2, _Rin);
    }

    //=====================================================================

    // Typed GPU dispatch for Outer, replacing the legacy 12x12 cuOuter_ii
    // dtype-pair function table (#1003). Outer.cpp promotes both operands to the
    // output dtype before calling in, so out/Lin/Rin all share out->dtype(): we
    // dispatch on that single type via as_storage_variant. Floating/complex
    // dtypes use the cuBLAS GER path (A = 1 * x * y^T); integer and bool dtypes
    // use the custom cuOuter_kernel. (This mirrors what the old diagonal table
    // entries did.)
    void cuOuter_dispatch(boost::intrusive_ptr<Storage_base> &out,
                          boost::intrusive_ptr<Storage_base> &Lin,
                          boost::intrusive_ptr<Storage_base> &Rin, const cytnx_uint64 &j1,
                          const cytnx_uint64 &j2) {
      std::visit(
        [&](auto out_impl) {
          using T = storage_value_t<decltype(out_impl)>;
          if constexpr (std::is_same_v<T, cytnx_complex128>) {
            cuGer_internal_cd(out, Lin, Rin, Scalar(1, Type.ComplexDouble));
          } else if constexpr (std::is_same_v<T, cytnx_complex64>) {
            cuGer_internal_cf(out, Lin, Rin, Scalar(1, Type.ComplexFloat));
          } else if constexpr (std::is_same_v<T, cytnx_double>) {
            cuGer_internal_d(out, Lin, Rin, Scalar(1, Type.Double));
          } else if constexpr (std::is_same_v<T, cytnx_float>) {
            cuGer_internal_f(out, Lin, Rin, Scalar(1, Type.Float));
          } else {
            cuOuter_general<T, T, T>(out, Lin, Rin, j1, j2);
          }
        },
        as_storage_variant(out));
    }

  }  // namespace linalg_internal
}  // namespace cytnx
