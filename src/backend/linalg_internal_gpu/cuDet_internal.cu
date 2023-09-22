#include "cuDet_internal.hpp"
#include "cytnx_error.hpp"
#include "Type.hpp"
#include "backend/lapack_wrapper.hpp"

#include "../utils_internal_gpu/cuAlloc_gpu.hpp"

#ifdef UNI_MAGMA
  #include "magma_v2.h"
#endif

namespace cytnx {

  namespace linalg_internal {

    void cuDet_internal_cd(void* out, const boost::intrusive_ptr<Storage_base>& in,
                           const cytnx_uint64& L) {
#ifdef UNI_MAGMA
      cytnx_complex128* od = (cytnx_complex128*)out;  // result on cpu!
      cuDoubleComplex* _in = (cuDoubleComplex*)utils_internal::cuMalloc_gpu(
        in->len * sizeof(cuDoubleComplex));  // unify mem.
      checkCudaErrors(
        cudaMemcpy(_in, in->Mem, sizeof(cytnx_complex128) * in->len, cudaMemcpyDeviceToDevice));

      magma_int_t* ipiv;
      magma_imalloc_cpu(&ipiv, L + 1);
      magma_int_t N = L;
      magma_int_t info;
      magma_zgetrf_gpu(N, N, _in, N, ipiv, &info);
      cytnx_error_msg(info != 0, "[ERROR] magma_zgetrf_gpu fail with info= %d\n", info);

      // since we do unify mem, direct access element is possible:
      od[0] = 1;
      bool neg = 0;
      for (magma_int_t i = 0; i < N; i++) {
        od[0] *= ((cytnx_complex128*)_in)[i * N + i];
        if (ipiv[i] != (i + 1)) neg = !neg;
      }
      magma_free_cpu(ipiv);
      cudaFree(_in);
      if (neg) od[0] *= -1;

#else
      cytnx_error_msg(true,
                      "[ERROR][internal Det] Det for Tensor on GPU require magma. please "
                      "re-compiling cytnx with magma.%s",
                      "\n");
#endif
    }

    void cuDet_internal_cf(void* out, const boost::intrusive_ptr<Storage_base>& in,
                           const cytnx_uint64& L) {
#ifdef UNI_MAGMA
      cytnx_complex64* od = (cytnx_complex64*)out;  // result on cpu!
      cuFloatComplex* _in = (cuFloatComplex*)utils_internal::cuMalloc_gpu(
        in->len * sizeof(cuFloatComplex));  // unify mem.
      checkCudaErrors(
        cudaMemcpy(_in, in->Mem, sizeof(cytnx_complex64) * in->len, cudaMemcpyDeviceToDevice));

      magma_int_t* ipiv;
      magma_imalloc_cpu(&ipiv, L + 1);
      magma_int_t N = L;
      magma_int_t info;
      magma_cgetrf_gpu(N, N, _in, N, ipiv, &info);
      cytnx_error_msg(info != 0, "[ERROR] magma_cgetrf_gpu fail with info= %d\n", info);

      // since we do unify mem, direct access element is possible:
      od[0] = 1;
      bool neg = 0;
      for (magma_int_t i = 0; i < N; i++) {
        od[0] *= ((cytnx_complex64*)_in)[i * N + i];
        if (ipiv[i] != (i + 1)) neg = !neg;
      }
      magma_free_cpu(ipiv);
      cudaFree(_in);
      if (neg) od[0] *= -1;

#else
      cytnx_error_msg(true,
                      "[ERROR][internal Det] Det for Tensor on GPU require magma. please "
                      "re-compiling cytnx with magma.%s",
                      "\n");
#endif
    }

    void cuDet_internal_d(void* out, const boost::intrusive_ptr<Storage_base>& in,
                          const cytnx_uint64& L) {
#ifdef UNI_MAGMA
      cytnx_double* od = (cytnx_double*)out;  // result on cpu!
      cytnx_double* _in =
        (cytnx_double*)utils_internal::cuMalloc_gpu(in->len * sizeof(cytnx_double));  // unify mem.
      checkCudaErrors(
        cudaMemcpy(_in, in->Mem, sizeof(cytnx_double) * in->len, cudaMemcpyDeviceToDevice));

      magma_int_t* ipiv;
      magma_imalloc_cpu(&ipiv, L + 1);
      magma_int_t N = L;
      magma_int_t info;
      magma_dgetrf_gpu(N, N, _in, N, ipiv, &info);
      cytnx_error_msg(info != 0, "[ERROR] magma_dgetrf_gpu fail with info= %d\n", info);

      // since we do unify mem, direct access element is possible:
      od[0] = 1;
      bool neg = 0;
      for (magma_int_t i = 0; i < N; i++) {
        od[0] *= _in[i * N + i];
        if (ipiv[i] != (i + 1)) neg = !neg;
      }
      magma_free_cpu(ipiv);
      cudaFree(_in);
      if (neg) od[0] *= -1;

#else
      cytnx_error_msg(true,
                      "[ERROR][internal Det] Det for Tensor on GPU require magma. please "
                      "re-compiling cytnx with magma.%s",
                      "\n");
#endif
    }

    void cuDet_internal_f(void* out, const boost::intrusive_ptr<Storage_base>& in,
                          const cytnx_uint64& L) {
#ifdef UNI_MAGMA
      cytnx_float* od = (cytnx_float*)out;  // result on cpu!
      cytnx_float* _in =
        (cytnx_float*)utils_internal::cuMalloc_gpu(in->len * sizeof(cytnx_float));  // unify mem.
      checkCudaErrors(
        cudaMemcpy(_in, in->Mem, sizeof(cytnx_float) * in->len, cudaMemcpyDeviceToDevice));

      magma_int_t* ipiv;
      magma_imalloc_cpu(&ipiv, L + 1);
      magma_int_t N = L;
      magma_int_t info;
      magma_sgetrf_gpu(N, N, _in, N, ipiv, &info);
      cytnx_error_msg(info != 0, "[ERROR] magma_sgetrf_gpu fail with info= %d\n", info);

      // since we do unify mem, direct access element is possible:
      od[0] = 1;
      bool neg = 0;
      for (magma_int_t i = 0; i < N; i++) {
        od[0] *= _in[i * N + i];
        if (ipiv[i] != (i + 1)) neg = !neg;
      }
      magma_free_cpu(ipiv);
      cudaFree(_in);
      if (neg) od[0] *= -1;

#else
      cytnx_error_msg(true,
                      "[ERROR][internal Det] Det for Tensor on GPU require magma. please "
                      "re-compiling cytnx with magma.%s",
                      "\n");
#endif
    }

  }  // namespace linalg_internal
}  // namespace cytnx
