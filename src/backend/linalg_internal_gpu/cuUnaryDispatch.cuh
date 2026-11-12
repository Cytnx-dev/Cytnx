#ifndef CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUUNARYDISPATCH_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUUNARYDISPATCH_H_

#include <cuda/std/cmath>
#include <cuda/std/complex>
#include <cuda/std/cstdlib>

#include <cstddef>
#include <type_traits>
#include <variant>

#include "Type.hpp"
#include "backend/Storage.hpp"
#include "cuTypeCvt.hpp"
#include "utils/complex_arithmetic.hpp"

// One shared elementwise-unary GPU framework for Abs / Exp / Pow / Conj (#1003 step 11).
//
// Before this, each of the four operations carried its own copy of the linear kernel, the launch
// configuration, the typed launch helper and an eleven-case dtype switch. They now share
// unary_kernel + launch below and differ only in an operation functor, a supported-dtype
// predicate, and an output-type rule.
//
// Dispatch runs over the ordinary Cytnx value types via as_storage_variant() / storage_cast<T>
// (Storage.hpp: the type-erased Storage_base::data() must not gain new callers), and the
// CUDA-native complex representation is confined to the kernel-launch boundary through to_cuda_t,
// matching cuArithmeticDispatch.cuh / cuiArithmeticDispatch.cuh (#1013).

namespace cytnx {
  namespace linalg_internal {
    namespace gpu_unary {

      constexpr unsigned int kThreadsPerBlock = 512;

      template <typename TOut, typename TIn, typename Op>
      __global__ void unary_kernel(TOut *out, const TIn *in, std::size_t size, Op op) {
        // 64-bit throughout: a 32-bit blockIdx.x * blockDim.x wraps past 2^32 elements.
        const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx < size) out[idx] = op(in[idx]);
      }

      // `out` and `in` may alias: an in-place operation passes the same buffer for both.
      template <typename TOut, typename TIn, typename Op>
      void launch(TOut *out, const TIn *in, std::size_t size, Op op) {
        if (size == 0) return;
        const unsigned int num_blocks =
          static_cast<unsigned int>((size + kThreadsPerBlock - 1) / kThreadsPerBlock);
        unary_kernel<<<num_blocks, kThreadsPerBlock>>>(out, in, size, op);
        CYTNX_CHECK_CUDA_LAUNCH(unary_kernel);
      }

      // ---- operation functors -------------------------------------------------------------
      //
      // Each returns exactly the operation's output type for the given input type, so the store in
      // unary_kernel is a same-type assignment. A TOut/TIn mismatch is then a compile error rather
      // than a silent conversion absorbed by a blanket static_cast at the store.

      struct AbsOp {
        template <typename TIn>
        __device__ auto operator()(TIn x) const {
          if constexpr (std::is_unsigned_v<TIn>) {
            return x;  // unsigned integer / bool: abs is the identity
          } else if constexpr (is_complex_v<TIn>) {
            return cuda::std::abs(x);  // complex<T> -> T, the magnitude
          } else {
            // Signed integer and floating. cuda::std::abs has no width-preserving overload for the
            // narrow integer types (the argument promotes to int), so the result narrows back.
            // Floating goes through cuda::std::abs rather than `x < 0 ? -x : x`, which returns
            // -0.0 for -0.0 because -0.0 < 0 is false.
            return static_cast<TIn>(cuda::std::abs(x));
          }
        }
      };

      struct ExpOp {
        template <typename TIn>
        __device__ TIn operator()(TIn x) const {
          return cuda::std::exp(x);
        }
      };

      struct ConjOp {
        template <typename TIn>
        __device__ TIn operator()(TIn x) const {
          return cuda::std::conj(x);
        }
      };

      struct PowOp {
        double p;

        template <typename TIn>
        __device__ TIn operator()(TIn x) const {
          if constexpr (std::is_same_v<TIn, cytnx_cuda_complex128>) {
            return cuda::std::pow(x, p);
          } else if constexpr (std::is_same_v<TIn, cytnx_cuda_complex64>) {
            // Keep the double exponent: compute in complex<double> and narrow only the result,
            // matching the CPU Pow_internal_cf path std::pow(complex<float>, double). Casting p to
            // float first would drop precision for exponents not representable as float.
            return static_cast<TIn>(cuda::std::pow(static_cast<cytnx_cuda_complex128>(x), p));
          } else {
            // Real. The CPU counterparts are pow(double, double) and powf(float, float), so the
            // float path narrows the exponent there too -- match it rather than widening.
            return static_cast<TIn>(cuda::std::pow(x, static_cast<TIn>(p)));
          }
        }
      };

      // ---- per-operation dtype and output-type rules --------------------------------------
      //
      // The rules are stated on the ordinary Cytnx value types; to_cuda_t is applied afterwards,
      // at the kernel-launch boundary only.

      // Abs accepts every dtype. Abs(complex) is real, everything else maps to itself --
      // complex_value_type_t is exactly that trait (#1092).
      template <typename T>
      struct AbsTraits {
        static constexpr bool supported = true;
        using output_t = internal::complex_value_type_t<T>;
      };

      // Exp and Pow are dtype-preserving. The front end pre-casts integer/bool input to Double, so
      // only the floating and complex dtypes ever reach the kernel.
      template <typename T>
      struct FloatingTraits {
        static constexpr bool supported =
          std::is_floating_point_v<T> || is_complex_floating_point_v<T>;
        using output_t = T;
      };

      // Conj is only dispatched for a complex dtype; real Conj is a no-op handled by the caller.
      template <typename T>
      struct ComplexTraits {
        static constexpr bool supported = is_complex_floating_point_v<T>;
        using output_t = T;
      };

    }  // namespace gpu_unary

    // Out-of-place dispatch. `out` must already be allocated with the operation's output dtype;
    // storage_cast<TOut> below is the single point that enforces it -- a mismatch throws there
    // rather than being assumed.
    template <template <typename> class Traits, typename Op>
    void cuUnaryDispatch(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &in, cytnx_uint64 Nelem, Op op,
                         const char *caller) {
      if (Nelem == 0) return;
      std::visit(
        [&](auto in_impl) {
          using TIn = storage_value_t<decltype(in_impl)>;
          if constexpr (!Traits<TIn>::supported) {
            cytnx_error_msg(true, "[%s] unsupported dtype: %d\n", caller, in->dtype());
          } else {
            using TOut = typename Traits<TIn>::output_t;
            gpu_unary::launch(reinterpret_cast<to_cuda_t<TOut> *>(storage_cast<TOut>(out)->data()),
                              reinterpret_cast<const to_cuda_t<TIn> *>(in_impl->data()),
                              static_cast<std::size_t>(Nelem), op);
          }
        },
        as_storage_variant(in));
    }

    // In-place dispatch: one buffer is both input and output, so the output type is the input type
    // by construction.
    template <template <typename> class Traits, typename Op>
    void cuUnaryDispatchInplace(boost::intrusive_ptr<Storage_base> &inout, cytnx_uint64 Nelem,
                                Op op, const char *caller) {
      if (Nelem == 0) return;
      std::visit(
        [&](auto impl) {
          using T = storage_value_t<decltype(impl)>;
          if constexpr (!Traits<T>::supported) {
            cytnx_error_msg(true, "[%s] unsupported dtype: %d\n", caller, inout->dtype());
          } else {
            auto *data = reinterpret_cast<to_cuda_t<T> *>(impl->data());
            gpu_unary::launch(data, data, static_cast<std::size_t>(Nelem), op);
          }
        },
        as_storage_variant(inout));
    }

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUUNARYDISPATCH_H_
