#ifndef CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUARITHMETICDISPATCH_H_
#define CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUARITHMETICDISPATCH_H_

#include <variant>
#include <vector>

#include "Type.hpp"
#include "backend/Storage.hpp"
#include "backend/utils_internal_interface.hpp"
#include "cuNonContigLayout.cuh"
#include "cuTypeCvt.hpp"

// Shared typed GPU dispatch for elementwise binary arithmetic (Add/Sub/Mul/Div),
// mirroring the CPU design in src/linalg/iArithmetic_visit.hpp: dispatch over
// the ordinary Cytnx value types (type_promote_t -- the dtype indices are
// device-independent) and confine the
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
                                         const TR *rhs,
                                         const gpu_layout::GpuNonContigLayout layout) {
        const cytnx_uint64 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
          cytnx_uint64 Lidx, Ridx;
          gpu_layout::compute_gpu_non_contig_indices(idx, layout, Lidx, Ridx);
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
            const gpu_layout::GpuNonContigLayout layout =
              gpu_layout::make_gpu_non_contig_layout(shape, invmapper_L, invmapper_R);
            tn_kernel_nonconti<op_code><<<NBlocks, 512>>>(_out, _Lin, len, _Rin, layout);
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

      // CUDA's Windows front end crashes while lowering the two-variant std::visit
      // previously used here (11 x 11 alternatives plus five kernels per pair).
      // Dispatching directly from the dtype ids preserves the same exhaustive
      // Type_list-derived instantiations without forcing cudafe++ through MSVC's
      // large std::visit dispatch table.
      template <char op_code, typename TLc, std::size_t RIndex = 1>
      void dispatch_rhs(boost::intrusive_ptr<Storage_base> &out,
                        boost::intrusive_ptr<Storage_base> &Lin,
                        boost::intrusive_ptr<Storage_base> &Rin, const cytnx_uint64 len,
                        const std::vector<cytnx_uint64> &shape,
                        const std::vector<cytnx_uint64> &invmapper_L,
                        const std::vector<cytnx_uint64> &invmapper_R) {
        using TRc = std::variant_alternative_t<RIndex, Type_list>;
        if (Rin->dtype() == static_cast<int>(RIndex)) {
          (void)storage_cast<TLc>(Lin);
          (void)storage_cast<TRc>(Rin);
          using TOc = OutputType_t<op_code, TLc, TRc>;
          launch<op_code, to_cuda_t<TOc>, to_cuda_t<TLc>, to_cuda_t<TRc>>(out, Lin, Rin, len, shape,
                                                                          invmapper_L, invmapper_R);
        } else if constexpr (RIndex + 1 < std::variant_size_v<Type_list>) {
          dispatch_rhs<op_code, TLc, RIndex + 1>(out, Lin, Rin, len, shape, invmapper_L,
                                                 invmapper_R);
        } else {
          cytnx_error_msg(true, "[cuArithmeticDispatchGPU] invalid RHS dtype %d%s", Rin->dtype(),
                          "\n");
        }
      }

      template <char op_code, std::size_t LIndex = 1>
      void dispatch_lhs(boost::intrusive_ptr<Storage_base> &out,
                        boost::intrusive_ptr<Storage_base> &Lin,
                        boost::intrusive_ptr<Storage_base> &Rin, const cytnx_uint64 len,
                        const std::vector<cytnx_uint64> &shape,
                        const std::vector<cytnx_uint64> &invmapper_L,
                        const std::vector<cytnx_uint64> &invmapper_R) {
        using TLc = std::variant_alternative_t<LIndex, Type_list>;
        if (Lin->dtype() == static_cast<int>(LIndex)) {
          dispatch_rhs<op_code, TLc>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
        } else if constexpr (LIndex + 1 < std::variant_size_v<Type_list>) {
          dispatch_lhs<op_code, LIndex + 1>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
        } else {
          cytnx_error_msg(true, "[cuArithmeticDispatchGPU] invalid LHS dtype %d%s", Lin->dtype(),
                          "\n");
        }
      }

    }  // namespace gpu_arith

    // Out-of-place typed GPU dispatch: `out` is pre-allocated by the caller with the
    // dtype gpu_arith::output_dtype(op_code, Lin, Rin). Validates that, then dispatches.
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

      gpu_arith::dispatch_lhs<op_code>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
    }

  }  // namespace linalg_internal
}  // namespace cytnx

#endif  // CYTNX_BACKEND_LINALG_INTERNAL_GPU_CUARITHMETICDISPATCH_H_
