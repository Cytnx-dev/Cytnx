#ifndef CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUIARITHMETICDISPATCH_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUIARITHMETICDISPATCH_H_

#include <type_traits>
#include <variant>
#include <vector>

#include "Type.hpp"
#include "Tensor.hpp"
#include "backend/Storage.hpp"
#include "backend/utils_internal_interface.hpp"
#include "cuTypeCvt.hpp"

// Shared typed GPU dispatch for *in-place* elementwise binary arithmetic
// (Add/Sub/Mul/Div), mirroring the CPU design in src/linalg/iArithmetic_visit.hpp:
// promote Lt's storage to the output dtype via storage_as_type_or_replace (a
// genuine-tensor RHS promotes like the out-of-place op; a python weak-scalar RHS
// preserves the LHS dtype, #980), then run the op writing into that (possibly
// aliased) buffer at Lt's PHYSICAL layout. The output dtype rule matches the
// out-of-place path: Div (3) is true division, others use type_promote (#941).
// std::visit over the ordinary Cytnx value types (as_storage_variant() +
// type_promote_t) with the CUDA-native representation confined to the kernel
// boundary via to_cuda_t (#1013).
//
// op_code: 0=Add, 1=Mul, 2=Sub, 3=Div (true division).

namespace cytnx {
  namespace linalg_internal {

    namespace gpu_iarith {

      // Output storage dtype per op_code (host value-type level). Mirrors
      // linalg::detail::InplaceOutputType_t: Div (3) is true division
      // (make_floating_point_t<type_promote_t<TL,TR>>), everything else uses plain
      // type_promote_t<TL,TR>.
      template <char op_code, typename TL, typename TR>
      struct InplaceOutputType {
        using type = Type_class::type_promote_t<TL, TR>;
      };
      template <typename TL, typename TR>
      struct InplaceOutputType<3, TL, TR> {
        using type = Type_class::make_floating_point_t<Type_class::type_promote_t<TL, TR>>;
      };
      template <char op_code, typename TL, typename TR>
      using InplaceOutputType_t = typename InplaceOutputType<op_code, TL, TR>::type;

      // Apply op_code in the output/compute type TO. The only way to reach a real TO
      // with a complex operand is a real LHS + complex *weak-scalar* RHS (TO recomputed
      // as TL, see DispatchInplaceArithmeticGPU); that case is rejected at the host
      // dispatch (iAdd/iSub/iMul/iDiv guard real (op)= complex-scalar), so this branch
      // is unreachable at runtime. A genuine complex tensor RHS promotes TO to complex,
      // so TO is never real there. The branch still needs to be well-formed because the
      // visit instantiates it -- a static_cast<real>(complex) would otherwise be
      // ill-formed. Note it silently returns TO{}, it does NOT throw: correctness of the
      // rejection depends entirely on the host guard above.
      template <char op_code, typename TO, typename TL, typename TR>
      __device__ inline TO ApplyInplaceGpuArithOp(TL lhs, TR rhs) {
        if constexpr (!is_complex_v<TO> && (is_complex_v<TL> || is_complex_v<TR>)) {
          return TO{};
        } else if constexpr (op_code == 0) {
          return static_cast<TO>(static_cast<TO>(lhs) + static_cast<TO>(rhs));
        } else if constexpr (op_code == 1) {
          return static_cast<TO>(static_cast<TO>(lhs) * static_cast<TO>(rhs));
        } else if constexpr (op_code == 2) {
          return static_cast<TO>(static_cast<TO>(lhs) - static_cast<TO>(rhs));
        } else {
          static_assert(op_code == 3, "gpu_iarith: op_code must be 0..3");
          return static_cast<TO>(static_cast<TO>(lhs) / static_cast<TO>(rhs));
        }
      }

      // Scalar RHS: out[i] = op(lhs[i], rhs). rhs is read host-side and passed by
      // value (a broadcast scalar stays host-resident, #988), matching the
      // out-of-place rconst kernel.
      template <char op_code, typename TO, typename TL, typename TR>
      __global__ void iscalar_kernel(TO *out, const TL *lhs, const cytnx_uint64 n, const TR rhs) {
        const cytnx_uint64 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = ApplyInplaceGpuArithOp<op_code, TO>(lhs[idx], rhs);
      }

      // Contiguous tensor RHS: out[i] = op(lhs[i], rhs[i]).
      template <char op_code, typename TO, typename TL, typename TR>
      __global__ void itn_kernel(TO *out, const TL *lhs, const cytnx_uint64 n, const TR *rhs) {
        const cytnx_uint64 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = ApplyInplaceGpuArithOp<op_code, TO>(lhs[idx], rhs[idx]);
      }

      // Non-contiguous tensor RHS. In-place semantics: the result lands at lhs's
      // PHYSICAL position Lidx (out either aliases lhs's buffer, or is a fresh
      // promoted buffer that Lt's unchanged meta/invmapper addresses the same way)
      // -- NOT at the logical linear index idx, unlike the out-of-place kernel.
      template <char op_code, typename TO, typename TL, typename TR>
      __global__ void itn_nonconti_kernel(TO *out, const TL *lhs, const cytnx_uint64 n,
                                          const TR *rhs, const cytnx_uint64 *accu_shape,
                                          const cytnx_uint64 *old_accu_shapeL,
                                          const cytnx_uint64 *old_accu_shapeR,
                                          const cytnx_uint64 *invmapper_L,
                                          const cytnx_uint64 *invmapper_R,
                                          const cytnx_uint64 shapesize) {
        extern __shared__ cytnx_uint64 tmpv[];

        const cytnx_uint64 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
          cytnx_uint64 tmp = idx;
          const cytnx_uint64 offset = threadIdx.x * shapesize;
          cytnx_uint64 Lidx = 0, Ridx = 0;

          for (cytnx_uint64 j = 0; j < shapesize; j++) {
            tmpv[offset + j] = tmp / accu_shape[j];
            tmp = tmp % accu_shape[j];
          }
          for (cytnx_uint64 j = 0; j < shapesize; j++) {
            Lidx += tmpv[offset + invmapper_L[j]] * old_accu_shapeL[j];
            Ridx += tmpv[offset + invmapper_R[j]] * old_accu_shapeR[j];
          }
          out[Lidx] = ApplyInplaceGpuArithOp<op_code, TO>(lhs[Lidx], rhs[Ridx]);
        }
      }

      // Launch the in-place kernels for one concrete (TO, TL, TR) instantiation.
      // out is the (possibly-replaced) promoted LHS buffer; lhs is the ORIGINAL LHS
      // buffer (captured before replacement, may alias out when TO == TL); rhs is
      // the RHS buffer. All are CUDA-native kernel pointers.
      template <char op_code, typename TO, typename TL, typename TR>
      void inplace_launch(TO *out, const TL *lhs, const TR *rhs, const cytnx_uint64 len,
                          bool rhs_is_scalar, const std::vector<cytnx_uint64> &shape,
                          const std::vector<cytnx_uint64> &invmapper_L,
                          const std::vector<cytnx_uint64> &invmapper_R) {
        if (len == 0) return;

        cytnx_uint32 NBlocks = len / 512;
        if (len % 512) NBlocks += 1;

        if (rhs_is_scalar) {
          iscalar_kernel<op_code><<<NBlocks, 512>>>(out, lhs, len, rhs[0]);
        } else if (shape.size() == 0) {
          itn_kernel<op_code><<<NBlocks, 512>>>(out, lhs, len, rhs);
        } else {
          cytnx_uint64 *m_accu_shape = reinterpret_cast<cytnx_uint64 *>(
            utils_internal::cuCalloc_gpu(shape.size(), sizeof(cytnx_uint64)));
          cytnx_uint64 *m_old_accu_shapeL = reinterpret_cast<cytnx_uint64 *>(
            utils_internal::cuCalloc_gpu(shape.size(), sizeof(cytnx_uint64)));
          cytnx_uint64 *m_old_accu_shapeR = reinterpret_cast<cytnx_uint64 *>(
            utils_internal::cuCalloc_gpu(shape.size(), sizeof(cytnx_uint64)));
          cytnx_uint64 *m_invmapper_L = reinterpret_cast<cytnx_uint64 *>(
            utils_internal::cuMalloc_gpu(invmapper_L.size() * sizeof(cytnx_uint64)));
          cytnx_uint64 *m_invmapper_R = reinterpret_cast<cytnx_uint64 *>(
            utils_internal::cuMalloc_gpu(invmapper_R.size() * sizeof(cytnx_uint64)));

          checkCudaErrors(cudaMemcpy(m_invmapper_L, &invmapper_L[0],
                                     sizeof(cytnx_uint64) * invmapper_L.size(),
                                     cudaMemcpyHostToDevice));
          checkCudaErrors(cudaMemcpy(m_invmapper_R, &invmapper_R[0],
                                     sizeof(cytnx_uint64) * invmapper_R.size(),
                                     cudaMemcpyHostToDevice));

          cytnx_uint64 tmp1 = 1, tmp2 = 1, tmp3 = 1;
          for (cytnx_uint64 i = 0; i < shape.size(); i++) {
            m_accu_shape[shape.size() - 1 - i] = tmp1;
            tmp1 *= shape[shape.size() - 1 - i];

            m_old_accu_shapeL[shape.size() - 1 - i] = tmp2;
            tmp2 *= shape[invmapper_L[shape.size() - 1 - i]];

            m_old_accu_shapeR[shape.size() - 1 - i] = tmp3;
            tmp3 *= shape[invmapper_R[shape.size() - 1 - i]];
          }

          itn_nonconti_kernel<op_code><<<NBlocks, 512, 512 * shape.size() * sizeof(cytnx_uint64)>>>(
            out, lhs, len, rhs, m_accu_shape, m_old_accu_shapeL, m_old_accu_shapeR, m_invmapper_L,
            m_invmapper_R, shape.size());

          checkCudaErrors(cudaFree(m_accu_shape));
          checkCudaErrors(cudaFree(m_old_accu_shapeL));
          checkCudaErrors(cudaFree(m_old_accu_shapeR));
          checkCudaErrors(cudaFree(m_invmapper_L));
          checkCudaErrors(cudaFree(m_invmapper_R));
        }
      }

    }  // namespace gpu_iarith

    // In-place typed GPU dispatch. Promotes Lt's storage to the output dtype and
    // runs op_code in place. rhs_is_weak_scalar signals a python-scalar RHS (numpy
    // weak-scalar semantics, #980/#1015): it preserves the LHS dtype (RHS treated
    // as TL) rather than promoting; a genuine tensor RHS (incl. a rank-0 tensor)
    // promotes (#941), so a real LHS + complex tensor RHS becomes complex here. Only
    // the real-LHS + complex-weak-scalar case is rejected by the caller.
    template <char op_code>
    void DispatchInplaceArithmeticGPU(Tensor &Lt, const Tensor &Rt, bool rhs_is_weak_scalar,
                                      const std::vector<cytnx_uint64> &shape,
                                      const std::vector<cytnx_uint64> &invmapper_L,
                                      const std::vector<cytnx_uint64> &invmapper_R) {
      const cytnx_uint64 len = Lt._impl->storage()._impl->size();
      const bool rhs_is_scalar = (Rt.size() == 1);
      const int device = Lt.device();
      checkCudaErrors(cudaSetDevice(device));

      std::visit(
        [&](auto lhs_impl, auto rhs_impl) {
          using TL = storage_value_t<decltype(lhs_impl)>;
          using TR = storage_value_t<decltype(rhs_impl)>;
          // Weak scalar: treat the RHS dtype as TL so Add/Sub/Mul keep TL (#980)
          // while Div still follows #941 true division (make_floating_point(TL)).
          // storage_as_type_or_replace<TO> may replace Lt's storage; lhs_impl was
          // captured by value above as an owning pointer to the ORIGINAL buffer, so
          // lhs_impl->data() stays valid for the kernel below.
          if (rhs_is_weak_scalar) {
            using TO = gpu_iarith::InplaceOutputType_t<op_code, TL, TL>;
            auto out_impl = storage_as_type_or_replace<TO>(Lt._impl->storage(), len, device);
            gpu_iarith::inplace_launch<op_code, to_cuda_t<TO>, to_cuda_t<TL>, to_cuda_t<TR>>(
              reinterpret_cast<to_cuda_t<TO> *>(out_impl->data()),
              reinterpret_cast<const to_cuda_t<TL> *>(lhs_impl->data()),
              reinterpret_cast<const to_cuda_t<TR> *>(rhs_impl->data()), len,
              /*rhs_is_scalar=*/true, shape, invmapper_L, invmapper_R);
          } else {
            using TO = gpu_iarith::InplaceOutputType_t<op_code, TL, TR>;
            auto out_impl = storage_as_type_or_replace<TO>(Lt._impl->storage(), len, device);
            gpu_iarith::inplace_launch<op_code, to_cuda_t<TO>, to_cuda_t<TL>, to_cuda_t<TR>>(
              reinterpret_cast<to_cuda_t<TO> *>(out_impl->data()),
              reinterpret_cast<const to_cuda_t<TL> *>(lhs_impl->data()),
              reinterpret_cast<const to_cuda_t<TR> *>(rhs_impl->data()), len, rhs_is_scalar, shape,
              invmapper_L, invmapper_R);
          }
        },
        Lt._impl->storage().as_storage_variant(), Rt._impl->storage().as_storage_variant());
    }

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUIARITHMETICDISPATCH_H_
