#include "cuMul_internal.hpp"
#include "backend/utils_internal_interface.hpp"

namespace cytnx {

  namespace linalg_internal {

    namespace {

      template <typename T>
      __device__ inline cuDoubleComplex CuToComplexDouble(const T &v) {
        return make_cuDoubleComplex(static_cast<cytnx_double>(v), 0.0);
      }

      __device__ inline cuDoubleComplex CuToComplexDouble(const cuDoubleComplex &v) { return v; }

      __device__ inline cuDoubleComplex CuToComplexDouble(const cuComplex &v) {
        return cuComplexFloatToDouble(v);
      }

      template <typename T>
      __device__ inline cuComplex CuToComplexFloat(const T &v) {
        return make_cuFloatComplex(static_cast<cytnx_float>(v), 0.0f);
      }

      __device__ inline cuComplex CuToComplexFloat(const cuComplex &v) { return v; }

      __device__ inline cuComplex CuToComplexFloat(const cuDoubleComplex &v) {
        return make_cuFloatComplex(static_cast<cytnx_float>(cuCreal(v)),
                                   static_cast<cytnx_float>(cuCimag(v)));
      }

      template <typename TO, typename TL, typename TR>
      __device__ inline TO CuMulDispatchOp(const TL &lhs, const TR &rhs) {
        if constexpr (std::is_same_v<TO, cuDoubleComplex>) {
          return cuCmul(CuToComplexDouble(lhs), CuToComplexDouble(rhs));
        } else if constexpr (std::is_same_v<TO, cuComplex>) {
          return cuCmulf(CuToComplexFloat(lhs), CuToComplexFloat(rhs));
        } else {
          return static_cast<TO>(lhs) * static_cast<TO>(rhs);
        }
      }

      template <typename TO, typename TL, typename TR>
      __global__ void cuMul_dispatch_constconst_kernel(TO *out, const TL lhs, const cytnx_uint64 n,
                                                       const TR rhs) {
        const cytnx_uint64 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = CuMulDispatchOp<TO>(lhs, rhs);
      }

      template <typename TO, typename TL, typename TR>
      __global__ void cuMul_dispatch_lconst_kernel(TO *out, const TL lhs, const cytnx_uint64 n,
                                                   const TR *rhs) {
        const cytnx_uint64 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = CuMulDispatchOp<TO>(lhs, rhs[idx]);
      }

      template <typename TO, typename TL, typename TR>
      __global__ void cuMul_dispatch_rconst_kernel(TO *out, const TL *lhs, const cytnx_uint64 n,
                                                   const TR rhs) {
        const cytnx_uint64 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = CuMulDispatchOp<TO>(lhs[idx], rhs);
      }

      template <typename TO, typename TL, typename TR>
      __global__ void cuMul_dispatch_tn_kernel(TO *out, const TL *lhs, const cytnx_uint64 n,
                                               const TR *rhs) {
        const cytnx_uint64 idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = CuMulDispatchOp<TO>(lhs[idx], rhs[idx]);
      }

      template <typename TO, typename TL, typename TR>
      __global__ void cuMul_dispatch_tn_kernel_nonconti(
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
          out[idx] = CuMulDispatchOp<TO>(lhs[Lidx], rhs[Ridx]);
        }
      }

      template <typename TL, typename TR>
      void cuMul_dispatch_typed(boost::intrusive_ptr<Storage_base> &out,
                                boost::intrusive_ptr<Storage_base> &Lin,
                                boost::intrusive_ptr<Storage_base> &Rin,
                                const unsigned long long &len,
                                const std::vector<cytnx_uint64> &shape,
                                const std::vector<cytnx_uint64> &invmapper_L,
                                const std::vector<cytnx_uint64> &invmapper_R) {
        using TO = Type_class::type_promote_gpu_t<TL, TR>;
        cytnx_error_msg(out->dtype() != Type_class::cy_typeid_gpu_v<TO>,
                        "[cuMul_dispatch] output dtype mismatch. got=%d expected=%d%s",
                        out->dtype(), Type_class::cy_typeid_gpu_v<TO>, "\n");

        TO *_out = reinterpret_cast<TO *>(out->data());
        const TL *_Lin = reinterpret_cast<const TL *>(Lin->data());
        const TR *_Rin = reinterpret_cast<const TR *>(Rin->data());

        cytnx_uint32 NBlocks = len / 512;
        if (len % 512) NBlocks += 1;

        if (Lin->size() == 1 and Rin->size() == 1) {
          cuMul_dispatch_constconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin[0]);
        } else if (Lin->size() == 1) {
          cuMul_dispatch_lconst_kernel<<<NBlocks, 512>>>(_out, _Lin[0], len, _Rin);
        } else if (Rin->size() == 1) {
          cuMul_dispatch_rconst_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin[0]);
        } else {
          if (shape.size() == 0) {
            cuMul_dispatch_tn_kernel<<<NBlocks, 512>>>(_out, _Lin, len, _Rin);
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

            cuMul_dispatch_tn_kernel_nonconti<<<NBlocks, 512,
                                                512 * shape.size() * sizeof(cytnx_uint64)>>>(
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
      void cuMul_dispatch_rhs(boost::intrusive_ptr<Storage_base> &out,
                              boost::intrusive_ptr<Storage_base> &Lin,
                              boost::intrusive_ptr<Storage_base> &Rin,
                              const unsigned long long &len, const std::vector<cytnx_uint64> &shape,
                              const std::vector<cytnx_uint64> &invmapper_L,
                              const std::vector<cytnx_uint64> &invmapper_R) {
        switch (Rin->dtype()) {
          case Type.ComplexDouble:
            cuMul_dispatch_typed<TL, cuDoubleComplex>(out, Lin, Rin, len, shape, invmapper_L,
                                                      invmapper_R);
            break;
          case Type.ComplexFloat:
            cuMul_dispatch_typed<TL, cuComplex>(out, Lin, Rin, len, shape, invmapper_L,
                                                invmapper_R);
            break;
          case Type.Double:
            cuMul_dispatch_typed<TL, cytnx_double>(out, Lin, Rin, len, shape, invmapper_L,
                                                   invmapper_R);
            break;
          case Type.Float:
            cuMul_dispatch_typed<TL, cytnx_float>(out, Lin, Rin, len, shape, invmapper_L,
                                                  invmapper_R);
            break;
          case Type.Int64:
            cuMul_dispatch_typed<TL, cytnx_int64>(out, Lin, Rin, len, shape, invmapper_L,
                                                  invmapper_R);
            break;
          case Type.Uint64:
            cuMul_dispatch_typed<TL, cytnx_uint64>(out, Lin, Rin, len, shape, invmapper_L,
                                                   invmapper_R);
            break;
          case Type.Int32:
            cuMul_dispatch_typed<TL, cytnx_int32>(out, Lin, Rin, len, shape, invmapper_L,
                                                  invmapper_R);
            break;
          case Type.Uint32:
            cuMul_dispatch_typed<TL, cytnx_uint32>(out, Lin, Rin, len, shape, invmapper_L,
                                                   invmapper_R);
            break;
          case Type.Int16:
            cuMul_dispatch_typed<TL, cytnx_int16>(out, Lin, Rin, len, shape, invmapper_L,
                                                  invmapper_R);
            break;
          case Type.Uint16:
            cuMul_dispatch_typed<TL, cytnx_uint16>(out, Lin, Rin, len, shape, invmapper_L,
                                                   invmapper_R);
            break;
          case Type.Bool:
            cuMul_dispatch_typed<TL, cytnx_bool>(out, Lin, Rin, len, shape, invmapper_L,
                                                 invmapper_R);
            break;
          default:
            cytnx_error_msg(true, "[cuMul_dispatch] unsupported rhs dtype: %d%s", Rin->dtype(),
                            "\n");
        }
      }

    }  // namespace

    void cuMul_dispatch(boost::intrusive_ptr<Storage_base> &out,
                        boost::intrusive_ptr<Storage_base> &Lin,
                        boost::intrusive_ptr<Storage_base> &Rin, const unsigned long long &len,
                        const std::vector<cytnx_uint64> &shape,
                        const std::vector<cytnx_uint64> &invmapper_L,
                        const std::vector<cytnx_uint64> &invmapper_R) {
      const unsigned int expected_dtype = Type.type_promote(Lin->dtype(), Rin->dtype());
      cytnx_error_msg(out->dtype() != expected_dtype,
                      "[cuMul_dispatch] output dtype mismatch. got=%d expected=%d%s", out->dtype(),
                      expected_dtype, "\n");

      switch (Lin->dtype()) {
        case Type.ComplexDouble:
          cuMul_dispatch_rhs<cuDoubleComplex>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        case Type.ComplexFloat:
          cuMul_dispatch_rhs<cuComplex>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        case Type.Double:
          cuMul_dispatch_rhs<cytnx_double>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        case Type.Float:
          cuMul_dispatch_rhs<cytnx_float>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        case Type.Int64:
          cuMul_dispatch_rhs<cytnx_int64>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        case Type.Uint64:
          cuMul_dispatch_rhs<cytnx_uint64>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        case Type.Int32:
          cuMul_dispatch_rhs<cytnx_int32>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        case Type.Uint32:
          cuMul_dispatch_rhs<cytnx_uint32>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        case Type.Int16:
          cuMul_dispatch_rhs<cytnx_int16>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        case Type.Uint16:
          cuMul_dispatch_rhs<cytnx_uint16>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        case Type.Bool:
          cuMul_dispatch_rhs<cytnx_bool>(out, Lin, Rin, len, shape, invmapper_L, invmapper_R);
          break;
        default:
          cytnx_error_msg(true, "[cuMul_dispatch] unsupported lhs dtype: %d%s", Lin->dtype(), "\n");
      }
    }

  }  // namespace linalg_internal
}  // namespace cytnx
