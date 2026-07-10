#include "cuCpr_internal.hpp"
#include "backend/utils_internal_interface.hpp"

namespace cytnx {

  namespace linalg_internal {

    namespace {

      // Compare two operands after promoting both to the common comparison type
      // TO = type_promote_gpu(TL, TR). Casting to TO first avoids mixed-type
      // operator issues (e.g. cuda::std::complex<float> == double is
      // ill-formed) and matches the CPU Cpr semantics: an integer/real operand
      // is compared against a complex operand by widening it to the complex
      // type (imaginary part 0), exactly as the old make_cuDoubleComplex(v, 0)
      // path did.
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
        const cytnx_uint64 *accu_shape, const cytnx_uint64 *old_accu_shapeL,
        const cytnx_uint64 *old_accu_shapeR, const cytnx_uint64 *invmapper_L,
        const cytnx_uint64 *invmapper_R, const cytnx_uint64 shapesize) {
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
          out[idx] = CuCprDispatchOp<TO>(lhs[Lidx], rhs[Ridx]);
        }
      }

      template <typename TL, typename TR>
      void cuCpr_dispatch_typed(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
        // Comparison result is always Bool; TO is only the common type the two
        // operands are compared in.
        using TO = Type_class::type_promote_gpu_t<TL, TR>;

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

            cuCpr_dispatch_tn_kernel_nonconti<TO>
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

      template <typename TL>
      void cuCpr_dispatch_rhs(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
        switch (Rin->dtype()) {
          case Type.ComplexDouble:
            cuCpr_dispatch_typed<TL, cytnx_cuda_complex128>(out, Lin, Rin, len, shape, invmapper_L,
                                                            invmapper_R);
            break;
          case Type.ComplexFloat:
            cuCpr_dispatch_typed<TL, cytnx_cuda_complex64>(out, Lin, Rin, len, shape, invmapper_L,
                                                           invmapper_R);
            break;
          case Type.Double:
            cuCpr_dispatch_typed<TL, cytnx_double>(out, Lin, Rin, len, shape, invmapper_L,
                                                   invmapper_R);
            break;
          case Type.Float:
            cuCpr_dispatch_typed<TL, cytnx_float>(out, Lin, Rin, len, shape, invmapper_L,
                                                  invmapper_R);
            break;
          case Type.Int64:
            cuCpr_dispatch_typed<TL, cytnx_int64>(out, Lin, Rin, len, shape, invmapper_L,
                                                  invmapper_R);
            break;
          case Type.Uint64:
            cuCpr_dispatch_typed<TL, cytnx_uint64>(out, Lin, Rin, len, shape, invmapper_L,
                                                   invmapper_R);
            break;
          case Type.Int32:
            cuCpr_dispatch_typed<TL, cytnx_int32>(out, Lin, Rin, len, shape, invmapper_L,
                                                  invmapper_R);
            break;
          case Type.Uint32:
            cuCpr_dispatch_typed<TL, cytnx_uint32>(out, Lin, Rin, len, shape, invmapper_L,
                                                   invmapper_R);
            break;
          case Type.Int16:
            cuCpr_dispatch_typed<TL, cytnx_int16>(out, Lin, Rin, len, shape, invmapper_L,
                                                  invmapper_R);
            break;
          case Type.Uint16:
            cuCpr_dispatch_typed<TL, cytnx_uint16>(out, Lin, Rin, len, shape, invmapper_L,
                                                   invmapper_R);
            break;
          case Type.Bool:
            cuCpr_dispatch_typed<TL, cytnx_bool>(out, Lin, Rin, len, shape, invmapper_L,
                                                 invmapper_R);
            break;
          default:
            cytnx_error_msg(true, "[cuCpr_dispatch] unsupported rhs dtype: %d%s", Rin->dtype(),
                            "\n");
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

      switch (Lin->dtype()) {
        case Type.ComplexDouble:
          cuCpr_dispatch_rhs<cytnx_cuda_complex128>(out, Lin, Rin, len, shape, invmapper_L,
                                                    invmapper_R);
          break;
        case Type.ComplexFloat:
          cuCpr_dispatch_rhs<cytnx_cuda_complex64>(out, Lin, Rin, len, shape, invmapper_L,
                                                   invmapper_R);
          break;
        case Type.Double:
          cuCpr_dispatch_rhs<cytnx_double>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        case Type.Float:
          cuCpr_dispatch_rhs<cytnx_float>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        case Type.Int64:
          cuCpr_dispatch_rhs<cytnx_int64>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        case Type.Uint64:
          cuCpr_dispatch_rhs<cytnx_uint64>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        case Type.Int32:
          cuCpr_dispatch_rhs<cytnx_int32>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        case Type.Uint32:
          cuCpr_dispatch_rhs<cytnx_uint32>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        case Type.Int16:
          cuCpr_dispatch_rhs<cytnx_int16>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        case Type.Uint16:
          cuCpr_dispatch_rhs<cytnx_uint16>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        case Type.Bool:
          cuCpr_dispatch_rhs<cytnx_bool>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        default:
          cytnx_error_msg(true, "[cuCpr_dispatch] unsupported lhs dtype: %d%s", Lin->dtype(), "\n");
      }
    }

  }  // namespace linalg_internal
}  // namespace cytnx
