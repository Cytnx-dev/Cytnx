#include "cuCpr_internal.hpp"

#include <variant>

#include "backend/Storage.hpp"
#include "backend/utils_internal_interface.hpp"
#include "cuNonContigLayout.cuh"
#include "cuTypeCvt.hpp"

// Typed GPU dispatch for out-of-place comparison (==). Unlike the arithmetic ops
// in cuArithmeticDispatch.cuh, the output storage type is always Bool while the
// operands are compared in the promoted type TO, so Cpr keeps its own Bool-output
// kernels. The *dispatch* still follows the #1013 design: dispatch over the
// ordinary Cytnx value types (type_promote_t) and map to
// the CUDA-native representation only at the kernel boundary via to_cuda_t --
// replacing the legacy type_promote_gpu_t / per-dtype switch dispatch.

namespace cytnx {

  namespace linalg_internal {

    namespace {

      // Compare two operands after promoting both to the common comparison type
      // TO. Casting to TO first avoids mixed-type operator issues (e.g.
      // cuda::std::complex<float> == double is ill-formed) and matches the CPU Cpr
      // semantics: an integer/real operand is compared against a complex operand by
      // widening it to the complex type (imaginary part 0), exactly as the old
      // make_cuDoubleComplex(v, 0) path did.
      template <typename TO, typename TL, typename TR>
      __device__ inline cytnx_bool CuCprDispatchOp(const TL &lhs, const TR &rhs) {
        return TO(lhs) == TO(rhs);
      }

      template <typename TO, typename TL, typename TR>
      __global__ void cuCpr_dispatch_constconst_kernel(cytnx_bool *out, const TL lhs,
                                                       const cytnx_uint64 n, const TR rhs) {
        const cytnx_uint64 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = CuCprDispatchOp<TO>(lhs, rhs);
      }

      template <typename TO, typename TL, typename TR>
      __global__ void cuCpr_dispatch_lconst_kernel(cytnx_bool *out, const TL lhs,
                                                   const cytnx_uint64 n, const TR *rhs) {
        const cytnx_uint64 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = CuCprDispatchOp<TO>(lhs, rhs[idx]);
      }

      template <typename TO, typename TL, typename TR>
      __global__ void cuCpr_dispatch_rconst_kernel(cytnx_bool *out, const TL *lhs,
                                                   const cytnx_uint64 n, const TR rhs) {
        const cytnx_uint64 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = CuCprDispatchOp<TO>(lhs[idx], rhs);
      }

      template <typename TO, typename TL, typename TR>
      __global__ void cuCpr_dispatch_tn_kernel(cytnx_bool *out, const TL *lhs, const cytnx_uint64 n,
                                               const TR *rhs) {
        const cytnx_uint64 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = CuCprDispatchOp<TO>(lhs[idx], rhs[idx]);
      }

      template <typename TO, typename TL, typename TR>
      __global__ void cuCpr_dispatch_tn_kernel_nonconti(
        cytnx_bool *out, const TL *lhs, const cytnx_uint64 n, const TR *rhs,
        const gpu_layout::GpuNonContigLayout layout) {
        const cytnx_uint64 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
          cytnx_uint64 Lidx, Ridx;
          gpu_layout::compute_gpu_non_contig_indices(idx, layout, Lidx, Ridx);
          out[idx] = CuCprDispatchOp<TO>(lhs[Lidx], rhs[Ridx]);
        }
      }

      // Launch the Bool-output comparison kernels for one concrete (TO, TL, TR)
      // instantiation. TO is the CUDA-native comparison (compute) type; the output
      // storage is Bool. out/Lin/Rin are erased Storage_base holding CUDA-resident
      // buffers; their raw data() pointers are reinterpret_cast to the kernel types.
      template <typename TO, typename TL, typename TR>
      void cpr_launch(boost::intrusive_ptr<Storage_base> &out,
                      boost::intrusive_ptr<Storage_base> &Lin,
                      boost::intrusive_ptr<Storage_base> &Rin, const cytnx_uint64 len,
                      const std::vector<cytnx_uint64> &shape,
                      const std::vector<cytnx_uint64> &invmapper_L,
                      const std::vector<cytnx_uint64> &invmapper_R) {
        cytnx_bool *_out = reinterpret_cast<cytnx_bool *>(out->data());
        const TL *_Lin = reinterpret_cast<const TL *>(Lin->data());
        const TR *_Rin = reinterpret_cast<const TR *>(Rin->data());

        cytnx_uint32 NBlocks = len / 512;
        if (len % 512) NBlocks += 1;

        if (Lin->size() == 1 and Rin->size() == 1) {
          cuCpr_dispatch_constconst_kernel<TO><<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
        } else if (Lin->size() == 1) {
          cuCpr_dispatch_lconst_kernel<TO><<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
        } else if (Rin->size() == 1) {
          cuCpr_dispatch_rconst_kernel<TO><<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
        } else {
          if (shape.size() == 0) {
            cuCpr_dispatch_tn_kernel<TO><<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
          } else {
            const gpu_layout::GpuNonContigLayout layout =
              gpu_layout::make_gpu_non_contig_layout(shape, invmapper_L, invmapper_R);
            cuCpr_dispatch_tn_kernel_nonconti<TO><<<NBlocks, 512>>>(_out, _Lin, len, _Rin, layout);
          }
        }
      }

      // The two-variant std::visit previously used here crashes cudafe++ on
      // Windows for the same reason as the arithmetic dispatcher: 11 x 11
      // alternatives, each with five kernel instantiations. Walk Type_list by
      // dtype id instead, retaining the checked erased-to-typed boundary.
      template <typename TLc, std::size_t RIndex = 1>
      void cpr_dispatch_rhs(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const cytnx_uint64 len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
        using TRc = std::variant_alternative_t<RIndex, Type_list>;
        if (Rin->dtype() == static_cast<int>(RIndex)) {
          (void)storage_cast<TLc>(Lin);
          (void)storage_cast<TRc>(Rin);
          using TOc = Type_class::type_promote_t<TLc, TRc>;
          cpr_launch<to_cuda_t<TOc>, to_cuda_t<TLc>, to_cuda_t<TRc>>(out, Lin, Rin, len, shape,
                                                                     invmapper_L, invmapper_R);
        } else if constexpr (RIndex + 1 < std::variant_size_v<Type_list>) {
          cpr_dispatch_rhs<TLc, RIndex + 1>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
        } else {
          cytnx_error_msg(true, "[cuCpr_dispatch] invalid RHS dtype %d%s", Rin->dtype(), "\n");
        }
      }

      template <std::size_t LIndex = 1>
      void cpr_dispatch_lhs(boost::intrusive_ptr<Storage_base> &out,
                            boost::intrusive_ptr<Storage_base> &Lin,
                            boost::intrusive_ptr<Storage_base> &Rin, const cytnx_uint64 len,
                            const std::vector<cytnx_uint64> &shape,
                            const std::vector<cytnx_uint64> &invmapper_L,
                            const std::vector<cytnx_uint64> &invmapper_R) {
        using TLc = std::variant_alternative_t<LIndex, Type_list>;
        if (Lin->dtype() == static_cast<int>(LIndex)) {
          cpr_dispatch_rhs<TLc>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
        } else if constexpr (LIndex + 1 < std::variant_size_v<Type_list>) {
          cpr_dispatch_lhs<LIndex + 1>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
        } else {
          cytnx_error_msg(true, "[cuCpr_dispatch] invalid LHS dtype %d%s", Lin->dtype(), "\n");
        }
      }

    }  // namespace

    void cuCpr_dispatch(boost::intrusive_ptr<Storage_base> &out,
                        boost::intrusive_ptr<Storage_base> &Lin,
                        boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                        const std::vector<cytnx_uint64> &shape,
                        const std::vector<cytnx_uint64> &invmapper_L,
                        const std::vector<cytnx_uint64> &invmapper_R) {
      cytnx_error_msg(out->dtype() != Type.Bool,
                      "[cuCpr_dispatch] output dtype must be Bool, got=%d%s", out->dtype(), "\n");
      if (len == 0) return;

      cpr_dispatch_lhs(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
    }

  }  // namespace linalg_internal
}  // namespace cytnx
