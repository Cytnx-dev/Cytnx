#include "cuEigh_internal.hpp"
#include "cytnx_error.hpp"
#include "Type.hpp"
#include "backend/lapack_wrapper.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// cuEigh
    void cuEigh_internal_cd(const boost::intrusive_ptr<Storage_base> &in,
                            boost::intrusive_ptr<Storage_base> &e,
                            boost::intrusive_ptr<Storage_base> &v, const cytnx_int64 &L) {
      cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
      if (v->dtype == Type.Void) jobz = CUSOLVER_EIG_MODE_NOVECTOR;

      // create handles:
      cusolverDnHandle_t cusolverH = NULL;
      checkCudaErrors(cusolverDnCreate(&cusolverH));

      cytnx_complex128 *tA;
      if (v != NULL) {
        tA = (cytnx_complex128 *)v->Mem;
        checkCudaErrors(cudaMemcpy(v->Mem, in->Mem, sizeof(cytnx_complex128) * cytnx_uint64(L) * L,
                                   cudaMemcpyDeviceToDevice));
      } else {
        checkCudaErrors(cudaMalloc((void **)&tA, cytnx_uint64(L) * L * sizeof(cytnx_complex128)));
        checkCudaErrors(cudaMemcpy(tA, in->Mem, sizeof(cytnx_complex128) * cytnx_uint64(L) * L,
                                   cudaMemcpyDeviceToDevice));
      }

      // query buffer:
      cytnx_int32 lwork = 0;
      cytnx_int32 b32L = L;
      checkCudaErrors(cusolverDnZheevd_bufferSize(cusolverH, jobz, CUBLAS_FILL_MODE_UPPER, b32L,
                                                  (cuDoubleComplex *)tA, b32L,
                                                  (cytnx_double *)e->Mem, &lwork));

      // allocate working space:
      cytnx_complex128 *work;
      checkCudaErrors(cudaMalloc((void **)&work, sizeof(cytnx_complex128) * lwork));

      // call :
      cytnx_int32 info;
      cytnx_int32 *devinfo;
      checkCudaErrors(cudaMalloc((void **)&devinfo, sizeof(cytnx_int32)));
      checkCudaErrors(cusolverDnZheevd(cusolverH, jobz, CUBLAS_FILL_MODE_UPPER, b32L,
                                       (cuDoubleComplex *)tA, b32L, (cytnx_double *)e->Mem,
                                       (cuDoubleComplex *)work, lwork, devinfo));

      // get info
      checkCudaErrors(cudaMemcpy(&info, devinfo, sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in cuBlas function 'cusolverDnZheevd': cuBlas INFO = ", info);

      cudaFree(work);
      if (v->dtype == Type.Void) cudaFree(tA);

      cudaFree(devinfo);
      cusolverDnDestroy(cusolverH);
    }
    void cuEigh_internal_cf(const boost::intrusive_ptr<Storage_base> &in,
                            boost::intrusive_ptr<Storage_base> &e,
                            boost::intrusive_ptr<Storage_base> &v, const cytnx_int64 &L) {
      cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
      if (v->dtype == Type.Void) jobz = CUSOLVER_EIG_MODE_NOVECTOR;

      // create handles:
      cusolverDnHandle_t cusolverH = NULL;
      checkCudaErrors(cusolverDnCreate(&cusolverH));

      cytnx_complex64 *tA;
      if (v != NULL) {
        tA = (cytnx_complex64 *)v->Mem;
        checkCudaErrors(cudaMemcpy(v->Mem, in->Mem, sizeof(cytnx_complex64) * cytnx_uint64(L) * L,
                                   cudaMemcpyDeviceToDevice));
      } else {
        checkCudaErrors(cudaMalloc((void **)&tA, cytnx_uint64(L) * L * sizeof(cytnx_complex64)));
        checkCudaErrors(cudaMemcpy(tA, in->Mem, sizeof(cytnx_complex64) * cytnx_uint64(L) * L,
                                   cudaMemcpyDeviceToDevice));
      }

      // query buffer:
      cytnx_int32 lwork = 0;
      cytnx_int32 b32L = L;
      checkCudaErrors(cusolverDnCheevd_bufferSize(cusolverH, jobz, CUBLAS_FILL_MODE_UPPER, b32L,
                                                  (cuFloatComplex *)tA, b32L, (cytnx_float *)e->Mem,
                                                  &lwork));

      // allocate working space:
      cytnx_complex64 *work;
      checkCudaErrors(cudaMalloc((void **)&work, sizeof(cytnx_complex64) * lwork));

      // call :
      cytnx_int32 info;
      cytnx_int32 *devinfo;
      checkCudaErrors(cudaMalloc((void **)&devinfo, sizeof(cytnx_int32)));
      checkCudaErrors(cusolverDnCheevd(cusolverH, jobz, CUBLAS_FILL_MODE_UPPER, b32L,
                                       (cuFloatComplex *)tA, b32L, (cytnx_float *)e->Mem,
                                       (cuFloatComplex *)work, lwork, devinfo));

      // get info
      checkCudaErrors(cudaMemcpy(&info, devinfo, sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in cuBlas function 'cusolverDnZheevd': cuBlas INFO = ", info);

      cudaFree(work);
      if (v->dtype == Type.Void) cudaFree(tA);

      cudaFree(devinfo);
      cusolverDnDestroy(cusolverH);
    }
    void cuEigh_internal_d(const boost::intrusive_ptr<Storage_base> &in,
                           boost::intrusive_ptr<Storage_base> &e,
                           boost::intrusive_ptr<Storage_base> &v, const cytnx_int64 &L) {
      cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
      if (v->dtype == Type.Void) jobz = CUSOLVER_EIG_MODE_NOVECTOR;

      // create handles:
      cusolverDnHandle_t cusolverH = NULL;
      checkCudaErrors(cusolverDnCreate(&cusolverH));

      cytnx_double *tA;
      if (v->dtype != Type.Void) {
        tA = (cytnx_double *)v->Mem;
        checkCudaErrors(cudaMemcpy(v->Mem, in->Mem, sizeof(cytnx_double) * cytnx_uint64(L) * L,
                                   cudaMemcpyDeviceToDevice));
      } else {
        checkCudaErrors(cudaMalloc((void **)&tA, cytnx_uint64(L) * L * sizeof(cytnx_double)));
        checkCudaErrors(cudaMemcpy(tA, in->Mem, sizeof(cytnx_double) * cytnx_uint64(L) * L,
                                   cudaMemcpyDeviceToDevice));
      }

      // query buffer:
      cytnx_int32 lwork = 0;
      cytnx_int32 b32L = L;
      checkCudaErrors(cusolverDnDsyevd_bufferSize(cusolverH, jobz, CUBLAS_FILL_MODE_UPPER, b32L, tA,
                                                  b32L, (cytnx_double *)e->Mem, &lwork));

      // allocate working space:
      cytnx_double *work;
      checkCudaErrors(cudaMalloc((void **)&work, sizeof(cytnx_double) * lwork));

      // call :
      cytnx_int32 info;
      cytnx_int32 *devinfo;
      checkCudaErrors(cudaMalloc((void **)&devinfo, sizeof(cytnx_int32)));
      checkCudaErrors(cusolverDnDsyevd(cusolverH, jobz, CUBLAS_FILL_MODE_UPPER, b32L, tA, b32L,
                                       (cytnx_double *)e->Mem, work, lwork, devinfo));

      // get info
      checkCudaErrors(cudaMemcpy(&info, devinfo, sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in cuBlas function 'cusolverDnDsysevd': cuBlas INFO = ", info);

      cudaFree(work);
      if (v->dtype == Type.Void) cudaFree(tA);

      cudaFree(devinfo);
      cusolverDnDestroy(cusolverH);
    }
    void cuEigh_internal_f(const boost::intrusive_ptr<Storage_base> &in,
                           boost::intrusive_ptr<Storage_base> &e,
                           boost::intrusive_ptr<Storage_base> &v, const cytnx_int64 &L) {
      cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
      if (v->dtype == Type.Void) jobz = CUSOLVER_EIG_MODE_NOVECTOR;

      // create handles:
      cusolverDnHandle_t cusolverH = NULL;
      checkCudaErrors(cusolverDnCreate(&cusolverH));

      cytnx_float *tA;
      if (v->dtype != Type.Void) {
        tA = (cytnx_float *)v->Mem;
        checkCudaErrors(cudaMemcpy(v->Mem, in->Mem, sizeof(cytnx_float) * cytnx_uint64(L) * L,
                                   cudaMemcpyDeviceToDevice));
      } else {
        checkCudaErrors(cudaMalloc((void **)&tA, cytnx_uint64(L) * L * sizeof(cytnx_float)));
        checkCudaErrors(cudaMemcpy(tA, in->Mem, sizeof(cytnx_float) * cytnx_uint64(L) * L,
                                   cudaMemcpyDeviceToDevice));
      }

      // query buffer:
      cytnx_int32 lwork = 0;
      cytnx_int32 b32L = L;
      checkCudaErrors(cusolverDnSsyevd_bufferSize(cusolverH, jobz, CUBLAS_FILL_MODE_UPPER, b32L, tA,
                                                  b32L, (cytnx_float *)e->Mem, &lwork));

      // allocate working space:
      cytnx_float *work;
      checkCudaErrors(cudaMalloc((void **)&work, sizeof(cytnx_float) * lwork));

      // call :
      cytnx_int32 info;
      cytnx_int32 *devinfo;
      checkCudaErrors(cudaMalloc((void **)&devinfo, sizeof(cytnx_int32)));
      checkCudaErrors(cusolverDnSsyevd(cusolverH, jobz, CUBLAS_FILL_MODE_UPPER, b32L, tA, b32L,
                                       (cytnx_float *)e->Mem, work, lwork, devinfo));

      // get info
      checkCudaErrors(cudaMemcpy(&info, devinfo, sizeof(cytnx_int32), cudaMemcpyDeviceToHost));
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in cuBlas function 'cusolverDnDsysevd': cuBlas INFO = ", info);

      cudaFree(work);
      if (v->dtype == Type.Void) cudaFree(tA);

      cudaFree(devinfo);
      cusolverDnDestroy(cusolverH);
    }

  }  // namespace linalg_internal
}  // namespace cytnx
