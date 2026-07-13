#include "cuMod_internal.hpp"
#include "backend/utils_internal_interface.hpp"

#include <type_traits>

namespace cytnx {

  namespace linalg_internal {

    namespace {

      // Modulo in the promoted type TO = type_promote_gpu(TL, TR), mirroring the
      // CPU ModOp<T>: integral -> a % b, floating -> fmod(a, b). Complex is not
      // a valid modulo type and is rejected at the host dispatch below, so the
      // complex branch here is unreachable and only needs to be well-formed.
      // Casting both operands to TO first avoids mixed-type operator issues (a
      // cuda::std::complex never reaches this in practice, and integer/real
      // promotion follows type_promote).
      template <typename TO, typename TL, typename TR>
      __device__ inline TO CuModDispatchOp(const TL &lhs, const TR &rhs) {
        if constexpr (is_complex_v<TO>) {
          return TO{};  // unreachable: complex modulo rejected at host dispatch
        } else if constexpr (std::is_same_v<TO, cytnx_double>) {
          return fmod(TO(lhs), TO(rhs));
        } else if constexpr (std::is_same_v<TO, cytnx_float>) {
          return fmodf(TO(lhs), TO(rhs));
        } else {
          return TO(lhs) % TO(rhs);
        }
      }

      template <typename TO, typename TL, typename TR>
      __global__ void cuMod_dispatch_constconst_kernel(TO *out, const TL lhs, const cytnx_uint64 n,
                                                       const TR rhs) {
        const cytnx_uint64 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = CuModDispatchOp<TO>(lhs, rhs);
      }

      template <typename TO, typename TL, typename TR>
      __global__ void cuMod_dispatch_lconst_kernel(TO *out, const TL lhs, const cytnx_uint64 n,
                                                   const TR *rhs) {
        const cytnx_uint64 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = CuModDispatchOp<TO>(lhs, rhs[idx]);
      }

      template <typename TO, typename TL, typename TR>
      __global__ void cuMod_dispatch_rconst_kernel(TO *out, const TL *lhs, const cytnx_uint64 n,
                                                   const TR rhs) {
        const cytnx_uint64 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = CuModDispatchOp<TO>(lhs[idx], rhs);
      }

      template <typename TO, typename TL, typename TR>
      __global__ void cuMod_dispatch_tn_kernel(TO *out, const TL *lhs, const cytnx_uint64 n,
                                               const TR *rhs) {
        const cytnx_uint64 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = CuModDispatchOp<TO>(lhs[idx], rhs[idx]);
      }

      template <typename TO, typename TL, typename TR>
      __global__ void cuMod_dispatch_tn_kernel_nonconti(
        TO *out, const TL *lhs, const cytnx_uint64 n, const TR *rhs, const cytnx_uint64 *accu_shape,
        const cytnx_uint64 *old_accu_shapeL, const cytnx_uint64 *old_accu_shapeR,
        const cytnx_uint64 *invmapper_L, const cytnx_uint64 *invmapper_R,
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
          out[idx] = CuModDispatchOp<TO>(lhs[Lidx], rhs[Ridx]);
        }
      }

      template <typename TL, typename TR>
      void cuMod_dispatch_typed(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
        using TO = Type_class::type_promote_gpu_t<TL, TR>;
        cytnx_error_msg(out->dtype() != Type_class::cy_typeid_gpu_v<TO>,
                        "[cuMod_dispatch] output dtype mismatch. got=%d expected=%d%s",
                        out->dtype(), Type_class::cy_typeid_gpu_v<TO>, "\n");

        TO *_out = reinterpret_cast<TO *>(out->data());
        const TL *_Lin = reinterpret_cast<const TL *>(Lin->data());
        const TR *_Rin = reinterpret_cast<const TR *>(Rin->data());

        cytnx_uint32 NBlocks = len / 512;
        if (len % 512) NBlocks += 1;

        if (Lin->size() == 1 and Rin->size() == 1) {
          cuMod_dispatch_constconst_kernel<TO><<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
        } else if (Lin->size() == 1) {
          cuMod_dispatch_lconst_kernel<TO><<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
        } else if (Rin->size() == 1) {
          cuMod_dispatch_rconst_kernel<TO><<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
        } else {
          if (shape.size() == 0) {
            cuMod_dispatch_tn_kernel<TO><<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
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

            cuMod_dispatch_tn_kernel_nonconti<TO>
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
      void cuMod_dispatch_rhs(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
        switch (Rin->dtype()) {
          case Type.Double:
            cuMod_dispatch_typed<TL, cytnx_double>(out, Lin, Rin, len, shape, invmapper_L,
                                                   invmapper_R);
            break;
          case Type.Float:
            cuMod_dispatch_typed<TL, cytnx_float>(out, Lin, Rin, len, shape, invmapper_L,
                                                  invmapper_R);
            break;
          case Type.Int64:
            cuMod_dispatch_typed<TL, cytnx_int64>(out, Lin, Rin, len, shape, invmapper_L,
                                                  invmapper_R);
            break;
          case Type.Uint64:
            cuMod_dispatch_typed<TL, cytnx_uint64>(out, Lin, Rin, len, shape, invmapper_L,
                                                   invmapper_R);
            break;
          case Type.Int32:
            cuMod_dispatch_typed<TL, cytnx_int32>(out, Lin, Rin, len, shape, invmapper_L,
                                                  invmapper_R);
            break;
          case Type.Uint32:
            cuMod_dispatch_typed<TL, cytnx_uint32>(out, Lin, Rin, len, shape, invmapper_L,
                                                   invmapper_R);
            break;
          case Type.Int16:
            cuMod_dispatch_typed<TL, cytnx_int16>(out, Lin, Rin, len, shape, invmapper_L,
                                                  invmapper_R);
            break;
          case Type.Uint16:
            cuMod_dispatch_typed<TL, cytnx_uint16>(out, Lin, Rin, len, shape, invmapper_L,
                                                   invmapper_R);
            break;
          case Type.Bool:
            cuMod_dispatch_typed<TL, cytnx_bool>(out, Lin, Rin, len, shape, invmapper_L,
                                                 invmapper_R);
            break;
          default:
            cytnx_error_msg(true, "[cuMod_dispatch] unsupported rhs dtype: %d%s", Rin->dtype(),
                            "\n");
        }
      }

    }  // namespace

    void cuMod_dispatch(boost::intrusive_ptr<Storage_base> &out,
                        boost::intrusive_ptr<Storage_base> &Lin,
                        boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                        const std::vector<cytnx_uint64> &shape,
                        const std::vector<cytnx_uint64> &invmapper_L,
                        const std::vector<cytnx_uint64> &invmapper_R) {
      // Modulo is undefined for complex operands (matches the CPU ModOp and the
      // old cuMod_internal_* host functions, which all error out for complex).
      cytnx_error_msg(Type.is_complex(Lin->dtype()) || Type.is_complex(Rin->dtype()),
                      "[cuMod] Cannot mod complex numbers%s", "\n");

      const unsigned int expected_dtype = Type.type_promote(Lin->dtype(), Rin->dtype());
      cytnx_error_msg(out->dtype() != expected_dtype,
                      "[cuMod_dispatch] output dtype mismatch. got=%d expected=%d%s", out->dtype(),
                      expected_dtype, "\n");
      if (len == 0) return;

      switch (Lin->dtype()) {
        case Type.Double:
          cuMod_dispatch_rhs<cytnx_double>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        case Type.Float:
          cuMod_dispatch_rhs<cytnx_float>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        case Type.Int64:
          cuMod_dispatch_rhs<cytnx_int64>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        case Type.Uint64:
          cuMod_dispatch_rhs<cytnx_uint64>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        case Type.Int32:
          cuMod_dispatch_rhs<cytnx_int32>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        case Type.Uint32:
          cuMod_dispatch_rhs<cytnx_uint32>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        case Type.Int16:
          cuMod_dispatch_rhs<cytnx_int16>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        case Type.Uint16:
          cuMod_dispatch_rhs<cytnx_uint16>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        case Type.Bool:
          cuMod_dispatch_rhs<cytnx_bool>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        default:
          cytnx_error_msg(true, "[cuMod_dispatch] unsupported lhs dtype: %d%s", Lin->dtype(), "\n");
      }
    }

  }  // namespace linalg_internal
}  // namespace cytnx
