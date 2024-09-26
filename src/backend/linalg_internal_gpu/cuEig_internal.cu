#include "linalg/linalg_internal_gpu/cuEig_internal.hpp"
#include "cytnx_error.hpp"
#include "Type.hpp"
#include "backend/lapack_wrapper.hpp"

#include "cuAlloc_gpu.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// cuEig
    void cuEig_internal_cd(const boost::intrusive_ptr<Storage_base> &in,
                           boost::intrusive_ptr<Storage_base> &e,
                           boost::intrusive_ptr<Storage_base> &v, const cytnx_int64 &L) {
      cusolverDnHandle_t cusolverH = NULL;
      cudaStream_t stream = NULL;
      cusolverDnCreate(&cusolverH);
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
      cusolverDnSetStream(cusolverH, stream);

      cuDoubleComplex *d_A, *d_W, *d_V;
      cudaMalloc((void **)&d_A, sizeof(cuDoubleComplex) * L * L);
      cudaMalloc((void **)&d_W, sizeof(cuDoubleComplex) * L);
      cudaMemcpy(d_A, in->Mem, sizeof(cuDoubleComplex) * L * L, cudaMemcpyHostToDevice);

      int lwork = 0;
      cusolverDnZgeev_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, L, d_A, L, d_W, d_V, L, NULL,
                                 1, &lwork);
      cuDoubleComplex *d_work;
      cudaMalloc((void **)&d_work, sizeof(cuDoubleComplex) * lwork);

      int *devInfo;
      cudaMalloc((void **)&devInfo, sizeof(int));

      cusolverDnZgeev(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_OP_N, L, d_A, L, d_W, d_V, L,
                      NULL, 1, d_work, lwork, devInfo);

      cudaMemcpy(e->Mem, d_W, sizeof(cuDoubleComplex) * L, cudaMemcpyDeviceToHost);
      if (v->dtype != Type.Void) {
        cudaMemcpy(v->Mem, d_V, sizeof(cuDoubleComplex) * L * L, cudaMemcpyDeviceToHost);
      }

      cudaFree(d_A);
      cudaFree(d_W);
      cudaFree(d_V);
      cudaFree(d_work);
      cudaFree(devInfo);
      cusolverDnDestroy(cusolverH);
      cudaStreamDestroy(stream);
    }

    void cuEig_internal_cf(const boost::intrusive_ptr<Storage_base> &in,
                           boost::intrusive_ptr<Storage_base> &e,
                           boost::intrusive_ptr<Storage_base> &v, const cytnx_int64 &L) {
      cusolverDnHandle_t cusolverH = NULL;
      cudaStream_t stream = NULL;
      cusolverDnCreate(&cusolverH);
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
      cusolverDnSetStream(cusolverH, stream);

      cuFloatComplex *d_A, *d_W, *d_V;
      cudaMalloc((void **)&d_A, sizeof(cuFloatComplex) * L * L);
      cudaMalloc((void **)&d_W, sizeof(cuFloatComplex) * L);
      cudaMemcpy(d_A, in->Mem, sizeof(cuFloatComplex) * L * L, cudaMemcpyHostToDevice);

      int lwork = 0;
      cusolverDnCgeev_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, L, d_A, L, d_W, d_V, L, NULL,
                                 1, &lwork);
      cuFloatComplex *d_work;
      cudaMalloc((void **)&d_work, sizeof(cuFloatComplex) * lwork);

      int *devInfo;
      cudaMalloc((void **)&devInfo, sizeof(int));

      cusolverDnCgeev(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_OP_N, L, d_A, L, d_W, d_V, L,
                      NULL, 1, d_work, lwork, devInfo);

      cudaMemcpy(e->Mem, d_W, sizeof(cuFloatComplex) * L, cudaMemcpyDeviceToHost);
      if (v->dtype != Type.Void) {
        cudaMemcpy(v->Mem, d_V, sizeof(cuFloatComplex) * L * L, cudaMemcpyDeviceToHost);
      }

      cudaFree(d_A);
      cudaFree(d_W);
      cudaFree(d_V);
      cudaFree(d_work);
      cudaFree(devInfo);
      cusolverDnDestroy(cusolverH);
      cudaStreamDestroy(stream);
    }

    void cuEig_internal_d(const boost::intrusive_ptr<Storage_base> &in,
                          boost::intrusive_ptr<Storage_base> &e,
                          boost::intrusive_ptr<Storage_base> &v, const cytnx_int64 &L) {
      cusolverDnHandle_t cusolverH = NULL;
      cudaStream_t stream = NULL;
      cusolverDnCreate(&cusolverH);
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
      cusolverDnSetStream(cusolverH, stream);

      double *d_A, *d_W, *d_V;
      cudaMalloc((void **)&d_A, sizeof(double) * L * L);
      cudaMalloc((void **)&d_W, sizeof(double) * L);
      cudaMemcpy(d_A, in->Mem, sizeof(double) * L * L, cudaMemcpyHostToDevice);

      int lwork = 0;
      cusolverDnDgeev_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, L, d_A, L, d_W, d_V, L, NULL,
                                 1, &lwork);
      double *d_work;
      cudaMalloc((void **)&d_work, sizeof(double) * lwork);

      int *devInfo;
      cudaMalloc((void **)&devInfo, sizeof(int));

      cusolverDnDgeev(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_OP_N, L, d_A, L, d_W, d_V, L,
                      NULL, 1, d_work, lwork, devInfo);

      cudaMemcpy(e->Mem, d_W, sizeof(double) * L, cudaMemcpyDeviceToHost);
      if (v->dtype != Type.Void) {
        cudaMemcpy(v->Mem, d_V, sizeof(double) * L * L, cudaMemcpyDeviceToHost);
      }

      cudaFree(d_A);
      cudaFree(d_W);
      cudaFree(d_V);
      cudaFree(d_work);
      cudaFree(devInfo);
      cusolverDnDestroy(cusolverH);
      cudaStreamDestroy(stream);
    }

    void cuEig_internal_f(const boost::intrusive_ptr<Storage_base> &in,
                          boost::intrusive_ptr<Storage_base> &e,
                          boost::intrusive_ptr<Storage_base> &v, const cytnx_int64 &L) {
      cusolverDnHandle_t cusolverH = NULL;
      cudaStream_t stream = NULL;
      cusolverDnCreate(&cusolverH);
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
      cusolverDnSetStream(cusolverH, stream);

      float *d_A, *d_W, *d_V;
      cudaMalloc((void **)&d_A, sizeof(float) * L * L);
      cudaMalloc((void **)&d_W, sizeof(float) * L);
      cudaMemcpy(d_A, in->Mem, sizeof(float) * L * L, cudaMemcpyHostToDevice);

      int lwork = 0;
      cusolverDnSgeev_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, L, d_A, L, d_W, d_V, L, NULL,
                                 1, &lwork);
      float *d_work;
      cudaMalloc((void **)&d_work, sizeof(float) * lwork);

      int *devInfo;
      cudaMalloc((void **)&devInfo, sizeof(int));

      cusolverDnSgeev(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_OP_N, L, d_A, L, d_W, d_V, L,
                      NULL, 1, d_work, lwork, devInfo);

      cudaMemcpy(e->Mem, d_W, sizeof(float) * L, cudaMemcpyDeviceToHost);
      if (v->dtype != Type.Void) {
        cudaMemcpy(v->Mem, d_V, sizeof(float) * L * L, cudaMemcpyDeviceToHost);
      }

      cudaFree(d_A);
      cudaFree(d_W);
      cudaFree(d_V);
      cudaFree(d_work);
      cudaFree(devInfo);
      cusolverDnDestroy(cusolverH);
      cudaStreamDestroy(stream);
    }

  }  // namespace linalg_internal
}  // namespace cytnx
