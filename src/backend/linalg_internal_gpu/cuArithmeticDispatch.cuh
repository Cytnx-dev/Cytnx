#ifndef CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUARITHMETICDISPATCH_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUARITHMETICDISPATCH_H_

#include <variant>
#include <vector>

#include "Type.hpp"
#include "backend/Storage.hpp"
#include "backend/utils_internal_interface.hpp"
#include "cuTypeCvt.hpp"

// Shared typed GPU dispatch for elementwise binary arithmetic (Add/Sub/Mul/Div),
// mirroring the CPU design in src/linalg/iArithmetic_visit.hpp: dispatch with
// std::visit over the ordinary Cytnx value types (as_storage_variant() +
// type_promote_t -- the dtype indices are device-independent) and confine the
// CUDA-native complex representation to the kernel-launch boundary via to_cuda_t.
// This replaces the legacy per-dtype-pair function tables and the parallel
// type_promote_gpu_t / cy_typeid_gpu_v promotion hierarchy (#1013).
//
// op_code: 0=Add, 1=Mul, 2=Sub, 3=Div (true division), 4=Mod. TO/TL/TR here are
// the CUDA-native kernel types (to_cuda_t of the Cytnx value types).

namespace cytnx {
  namespace linalg_internal {

    namespace gpu_arith {

      template <char op_code, typename TO, typename TL, typename TR>
      __device__ inline TO ApplyGpuArithOp(TL lhs, TR rhs) {
        if constexpr (op_code == 0) {
          return static_cast<TO>(lhs) + static_cast<TO>(rhs);
        } else if constexpr (op_code == 1) {
          return static_cast<TO>(lhs) * static_cast<TO>(rhs);
        } else if constexpr (op_code == 2) {
          return static_cast<TO>(lhs) - static_cast<TO>(rhs);
        } else if constexpr (op_code == 3) {
          return static_cast<TO>(lhs) / static_cast<TO>(rhs);
        } else {
          static_assert(op_code == 4, "gpu_arith: op_code must be 0..4");
          // Mod (#941): integral -> a % b, floating -> fmod/fmodf, mirroring the CPU
          // ModOp<T>. Complex modulo is rejected at the host dispatch (cuMod_dispatch),
          // so the complex branch is unreachable and only needs to be well-formed.
          if constexpr (is_complex_v<TO>) {
            return TO{};
          } else if constexpr (std::is_same_v<TO, cytnx_double>) {
            return fmod(static_cast<TO>(lhs), static_cast<TO>(rhs));
          } else if constexpr (std::is_same_v<TO, cytnx_float>) {
            return fmodf(static_cast<TO>(lhs), static_cast<TO>(rhs));
          } else {
            return static_cast<TO>(lhs) % static_cast<TO>(rhs);
          }
        }
      }

      template <char op_code, typename TO, typename TL, typename TR>
      __global__ void constconst_kernel(TO *out, const TL lhs, const cytnx_uint64 n, const TR rhs) {
        const cytnx_uint64 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = ApplyGpuArithOp<op_code, TO>(lhs, rhs);
      }

      template <char op_code, typename TO, typename TL, typename TR>
      __global__ void lconst_kernel(TO *out, const TL lhs, const cytnx_uint64 n, const TR *rhs) {
        const cytnx_uint64 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = ApplyGpuArithOp<op_code, TO>(lhs, rhs[idx]);
      }

      template <char op_code, typename TO, typename TL, typename TR>
      __global__ void rconst_kernel(TO *out, const TL *lhs, const cytnx_uint64 n, const TR rhs) {
        const cytnx_uint64 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = ApplyGpuArithOp<op_code, TO>(lhs[idx], rhs);
      }

      template <char op_code, typename TO, typename TL, typename TR>
      __global__ void tn_kernel(TO *out, const TL *lhs, const cytnx_uint64 n, const TR *rhs) {
        const cytnx_uint64 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = ApplyGpuArithOp<op_code, TO>(lhs[idx], rhs[idx]);
      }

      template <char op_code, typename TO, typename TL, typename TR>
      __global__ void tn_kernel_nonconti(TO *out, const TL *lhs, const cytnx_uint64 n,
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
          out[idx] = ApplyGpuArithOp<op_code, TO>(lhs[Lidx], rhs[Ridx]);
        }
      }

      // Launch the elementwise kernels for one concrete (TO, TL, TR) instantiation.
      // out/Lin/Rin are erased Storage_base holding CUDA-resident buffers; their raw
      // data() pointers are reinterpret_cast to the CUDA-native kernel types (valid:
      // std::complex<T> and cuda::std::complex<T> are layout-compatible). The output
      // storage dtype is validated by the caller.
      template <char op_code, typename TO, typename TL, typename TR>
      void launch(boost::intrusive_ptr<Storage_base> &out, boost::intrusive_ptr<Storage_base> &Lin,
                  boost::intrusive_ptr<Storage_base> &Rin, const cytnx_uint64 len,
                  const std::vector<cytnx_uint64> &shape,
                  const std::vector<cytnx_uint64> &invmapper_L,
                  const std::vector<cytnx_uint64> &invmapper_R) {
        TO *_out = reinterpret_cast<TO *>(out->data());
        const TL *_Lin = reinterpret_cast<const TL *>(Lin->data());
        const TR *_Rin = reinterpret_cast<const TR *>(Rin->data());

        cytnx_uint32 NBlocks = len / 512;
        if (len % 512) NBlocks += 1;

        if (Lin->size() == 1 and Rin->size() == 1) {
          constconst_kernel<op_code><<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
        } else if (Lin->size() == 1) {
          lconst_kernel<op_code><<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
        } else if (Rin->size() == 1) {
          rconst_kernel<op_code><<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
        } else {
          if (shape.size() == 0) {
            tn_kernel<op_code><<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
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

            tn_kernel_nonconti<op_code>
              <<<NBlocks, 512, 512 * shape.size() * sizeof(cytnx_uint64)>>>(
                _out, _Lin, len, _Rin, m_accu_shape, m_old_accu_shapeL, m_old_accu_shapeR,
                m_invmapper_L, m_invmapper_R, shape.size());

            checkCudaErrors(cudaFree(m_accu_shape));
            checkCudaErrors(cudaFree(m_old_accu_shapeL));
            checkCudaErrors(cudaFree(m_old_accu_shapeR));
            checkCudaErrors(cudaFree(m_invmapper_L));
            checkCudaErrors(cudaFree(m_invmapper_R));
          }
        }
      }

      // Output value type for op_code, at the Cytnx (host) value-type level:
      // Div (3) is true division -> make_floating_point_t<type_promote_t<TL,TR>>;
      // Add/Sub/Mul use plain type_promote_t<TL,TR> (#941).
      template <char op_code, typename TL, typename TR>
      struct OutputType {
        using type = Type_class::type_promote_t<TL, TR>;
      };
      template <typename TL, typename TR>
      struct OutputType<3, TL, TR> {
        using type = Type_class::make_floating_point_t<Type_class::type_promote_t<TL, TR>>;
      };
      template <char op_code, typename TL, typename TR>
      using OutputType_t = typename OutputType<op_code, TL, TR>::type;

      // Runtime output dtype for op_code (matches OutputType_t at the enum level).
      inline unsigned int output_dtype(char op_code, unsigned int lhs_dtype,
                                       unsigned int rhs_dtype) {
        const unsigned int promoted = Type.type_promote(lhs_dtype, rhs_dtype);
        return op_code == 3 ? Type_class::make_floating_point_dtype(promoted) : promoted;
      }

    }  // namespace gpu_arith

    // Out-of-place typed GPU dispatch: `out` is pre-allocated by the caller with the
    // dtype gpu_arith::output_dtype(op_code, Lin, Rin). Validates that, then visits.
    template <char op_code>
    void cuArithmeticDispatchGPU(boost::intrusive_ptr<Storage_base> &out,
                                 boost::intrusive_ptr<Storage_base> &Lin,
                                 boost::intrusive_ptr<Storage_base> &Rin, const cytnx_uint64 len,
                                 const std::vector<cytnx_uint64> &shape,
                                 const std::vector<cytnx_uint64> &invmapper_L,
                                 const std::vector<cytnx_uint64> &invmapper_R) {
      const unsigned int expected_dtype =
        gpu_arith::output_dtype(op_code, Lin->dtype(), Rin->dtype());
      cytnx_error_msg(out->dtype() != expected_dtype,
                      "[cuArithmeticDispatchGPU] output dtype mismatch. got=%d expected=%d%s",
                      out->dtype(), expected_dtype, "\n");
      if (len == 0) return;

      std::visit(
        [&](auto lhs_impl, auto rhs_impl) {
          using TLc = storage_value_t<decltype(lhs_impl)>;
          using TRc = storage_value_t<decltype(rhs_impl)>;
          using TOc = gpu_arith::OutputType_t<op_code, TLc, TRc>;
          gpu_arith::launch<op_code, to_cuda_t<TOc>, to_cuda_t<TLc>, to_cuda_t<TRc>>(
            out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
        },
        as_storage_variant(Lin), as_storage_variant(Rin));
    }

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUARITHMETICDISPATCH_H_
