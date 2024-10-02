#include "cuDet_internal.hpp"
#include "cytnx_error.hpp"
#include "Type.hpp"
#include "backend/lapack_wrapper.hpp"

#include "../utils_internal_gpu/cuAlloc_gpu.hpp"

namespace cytnx {

  namespace linalg_internal {

    void cuDet_internal_cd(void* out, const boost::intrusive_ptr<Storage_base>& in,
                           const cytnx_uint64& L) {
      cytnx_complex128* od = (cytnx_complex128*)out;  // result on cpu!
      cuDoubleComplex* _in = (cuDoubleComplex*)utils_internal::cuMalloc_gpu(
        in->len * sizeof(cuDoubleComplex));  // unify mem.
      checkCudaErrors(
        cudaMemcpy(_in, in->Mem, sizeof(cytnx_complex128) * in->len, cudaMemcpyDeviceToDevice));

      cusolverDnHandle_t cusolverH;
      cusolverDnCreate(&cusolverH);

      int* devIpiv;
      int* devInfo;
      checkCudaErrors(cudaMalloc((void**)&devIpiv, L * sizeof(int)));
      checkCudaErrors(cudaMalloc((void**)&devInfo, sizeof(int)));

      int workspace_size = 0;
      cuDoubleComplex* workspace = NULL;
      cusolverDnZgetrf_bufferSize(cusolverH, L, L, _in, L, &workspace_size);
      checkCudaErrors(cudaMalloc((void**)&workspace, workspace_size * sizeof(cuDoubleComplex)));

      cusolverDnZgetrf(cusolverH, L, L, _in, L, workspace, devIpiv, devInfo);

      int info;
      checkCudaErrors(cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
      // If the info > 0, that means the U factor is exactly singular, and the det is 0.
      cytnx_error_msg(info < 0, "[ERROR] cusolverDnZgetrf fail with info= %d\n", info);

      // since we do unify mem, direct access element is possible:
      od[0] = 1;
      bool neg = 0;
      int* ipiv = new int[L];
      checkCudaErrors(cudaMemcpy(ipiv, devIpiv, L * sizeof(int), cudaMemcpyDeviceToHost));
      for (int i = 0; i < L; i++) {
        od[0] *= ((cytnx_complex128*)_in)[i * L + i];
        if (ipiv[i] != (i + 1)) neg = !neg;
      }
      delete[] ipiv;
      cudaFree(devIpiv);
      cudaFree(devInfo);
      cudaFree(workspace);
      cudaFree(_in);
      cusolverDnDestroy(cusolverH);
      if (neg) od[0] *= -1;

      if (info > 0) od[0] = 0;
    }

    void cuDet_internal_cf(void* out, const boost::intrusive_ptr<Storage_base>& in,
                           const cytnx_uint64& L) {
      cytnx_complex64* od = (cytnx_complex64*)out;  // result on cpu!
      cuFloatComplex* _in = (cuFloatComplex*)utils_internal::cuMalloc_gpu(
        in->len * sizeof(cuFloatComplex));  // unify mem.
      checkCudaErrors(
        cudaMemcpy(_in, in->Mem, sizeof(cytnx_complex64) * in->len, cudaMemcpyDeviceToDevice));

      cusolverDnHandle_t cusolverH;
      cusolverDnCreate(&cusolverH);

      int* devIpiv;
      int* devInfo;
      checkCudaErrors(cudaMalloc((void**)&devIpiv, L * sizeof(int)));
      checkCudaErrors(cudaMalloc((void**)&devInfo, sizeof(int)));

      int workspace_size = 0;
      cuFloatComplex* workspace = NULL;
      cusolverDnCgetrf_bufferSize(cusolverH, L, L, _in, L, &workspace_size);
      checkCudaErrors(cudaMalloc((void**)&workspace, workspace_size * sizeof(cuFloatComplex)));

      cusolverDnCgetrf(cusolverH, L, L, _in, L, workspace, devIpiv, devInfo);

      int info;
      checkCudaErrors(cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
      // If the info > 0, that means the U factor is exactly singular, and the det is 0.
      cytnx_error_msg(info < 0, "[ERROR] cusolverDnCgetrf fail with info= %d\n", info);

      // since we do unify mem, direct access element is possible:
      od[0] = 1;
      bool neg = 0;
      int* ipiv = new int[L];
      checkCudaErrors(cudaMemcpy(ipiv, devIpiv, L * sizeof(int), cudaMemcpyDeviceToHost));
      for (int i = 0; i < L; i++) {
        od[0] *= ((cytnx_complex64*)_in)[i * L + i];
        if (ipiv[i] != (i + 1)) neg = !neg;
      }
      delete[] ipiv;
      cudaFree(devIpiv);
      cudaFree(devInfo);
      cudaFree(workspace);
      cudaFree(_in);
      cusolverDnDestroy(cusolverH);
      if (neg) od[0] *= -1;

      if (info > 0) od[0] = 0;
    }

    void cuDet_internal_d(void* out, const boost::intrusive_ptr<Storage_base>& in,
                          const cytnx_uint64& L) {
      cytnx_double* od = (cytnx_double*)out;  // result on cpu!
      cytnx_double* _in =
        (cytnx_double*)utils_internal::cuMalloc_gpu(in->len * sizeof(cytnx_double));  // unify mem.
      checkCudaErrors(
        cudaMemcpy(_in, in->Mem, sizeof(cytnx_double) * in->len, cudaMemcpyDeviceToDevice));

      cusolverDnHandle_t cusolverH;
      cusolverDnCreate(&cusolverH);

      int* devIpiv;
      int* devInfo;
      checkCudaErrors(cudaMalloc((void**)&devIpiv, L * sizeof(int)));
      checkCudaErrors(cudaMalloc((void**)&devInfo, sizeof(int)));

      int workspace_size = 0;
      cytnx_double* workspace = NULL;
      cusolverDnDgetrf_bufferSize(cusolverH, L, L, _in, L, &workspace_size);
      checkCudaErrors(cudaMalloc((void**)&workspace, workspace_size * sizeof(cytnx_double)));

      cusolverDnDgetrf(cusolverH, L, L, _in, L, workspace, devIpiv, devInfo);

      int info;
      checkCudaErrors(cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
      // If the info > 0, that means the U factor is exactly singular, and the det is 0.
      cytnx_error_msg(info < 0, "[ERROR] cusolverDnDgetrf fail with info= %d\n", info);

      // since we do unify mem, direct access element is possible:
      od[0] = 1;
      bool neg = 0;
      int* ipiv = new int[L];
      checkCudaErrors(cudaMemcpy(ipiv, devIpiv, L * sizeof(int), cudaMemcpyDeviceToHost));
      for (int i = 0; i < L; i++) {
        od[0] *= _in[i * L + i];
        if (ipiv[i] != (i + 1)) neg = !neg;
      }
      delete[] ipiv;
      cudaFree(devIpiv);
      cudaFree(devInfo);
      cudaFree(workspace);
      cudaFree(_in);
      cusolverDnDestroy(cusolverH);
      if (neg) od[0] *= -1;

      if (info > 0) od[0] = 0;
    }

    void cuDet_internal_f(void* out, const boost::intrusive_ptr<Storage_base>& in,
                          const cytnx_uint64& L) {
      cytnx_float* od = (cytnx_float*)out;  // result on cpu!
      cytnx_float* _in =
        (cytnx_float*)utils_internal::cuMalloc_gpu(in->len * sizeof(cytnx_float));  // unify mem.
      checkCudaErrors(
        cudaMemcpy(_in, in->Mem, sizeof(cytnx_float) * in->len, cudaMemcpyDeviceToDevice));

      cusolverDnHandle_t cusolverH;
      cusolverDnCreate(&cusolverH);

      int* devIpiv;
      int* devInfo;
      checkCudaErrors(cudaMalloc((void**)&devIpiv, L * sizeof(int)));
      checkCudaErrors(cudaMalloc((void**)&devInfo, sizeof(int)));

      int workspace_size = 0;
      cytnx_float* workspace = NULL;
      cusolverDnSgetrf_bufferSize(cusolverH, L, L, _in, L, &workspace_size);
      checkCudaErrors(cudaMalloc((void**)&workspace, workspace_size * sizeof(cytnx_float)));

      cusolverDnSgetrf(cusolverH, L, L, _in, L, workspace, devIpiv, devInfo);

      int info;
      checkCudaErrors(cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
      // If the info > 0, that means the U factor is exactly singular, and the det is 0.
      cytnx_error_msg(info < 0, "[ERROR] cusolverDnSgetrf fail with info= %d\n", info);

      // since we do unify mem, direct access element is possible:
      od[0] = 1;
      bool neg = 0;
      int* ipiv = new int[L];
      checkCudaErrors(cudaMemcpy(ipiv, devIpiv, L * sizeof(int), cudaMemcpyDeviceToHost));
      for (int i = 0; i < L; i++) {
        od[0] *= _in[i * L + i];
        if (ipiv[i] != (i + 1)) neg = !neg;
      }
      delete[] ipiv;
      cudaFree(devIpiv);
      cudaFree(devInfo);
      cudaFree(workspace);
      cudaFree(_in);
      cusolverDnDestroy(cusolverH);
      if (neg) od[0] *= -1;

      if (info > 0) od[0] = 0;
    }

  }  // namespace linalg_internal
}  // namespace cytnx
