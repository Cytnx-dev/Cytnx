#include "cuGeSvd_internal.hpp"
#include "cuConj_inplace_internal.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// cuGeSvd
    void cuGeSvd_internal_cd(const boost::intrusive_ptr<Storage_base> &in,
                             boost::intrusive_ptr<Storage_base> &U,
                             boost::intrusive_ptr<Storage_base> &vT,
                             boost::intrusive_ptr<Storage_base> &S, const cytnx_int64 &M,
                             const cytnx_int64 &N) {
      using data_type = cytnx_complex128;
      using d_data_type = cuDoubleComplex;
      cusolverEigMode_t jobz;
      // if U and vT are NULL ptr, then it will not be computed.
      jobz = (U->dtype == Type.Void and vT->dtype == Type.Void) ? CUSOLVER_EIG_MODE_NOVECTOR
                                                                : CUSOLVER_EIG_MODE_VECTOR;

      // const int econ = 0; /* i.e. 'A' in gesvd  */
      cytnx_int32 econ = 1; /* i.e. 'S' in gesvd  */

      // create handles:
      cusolverDnHandle_t cusolverH = NULL;
      checkCudaErrors(cusolverDnCreate(&cusolverH));

      cuDoubleComplex *Mij;
      checkCudaErrors(cudaMalloc((void **)&Mij, M * N * sizeof(data_type)));
      checkCudaErrors(
        cudaMemcpy(Mij, in->Mem, sizeof(data_type) * M * N, cudaMemcpyDeviceToDevice));

      cytnx_int64 min = std::min(M, N);
      cytnx_int64 max = std::max(M, N);
      cytnx_int64 ldA = N, ldu = N, ldvT = M;

      void *UMem = nullptr, *vTMem = nullptr;
      if (U->Mem) {
        UMem = U->Mem;
      } else {
        if (jobz == CUSOLVER_EIG_MODE_VECTOR)
          checkCudaErrors(cudaMalloc(&UMem, max * max * sizeof(data_type)));
      }
      if (vT->Mem) {
        vTMem = vT->Mem;
      } else {
        if (jobz == CUSOLVER_EIG_MODE_VECTOR)
          checkCudaErrors(cudaMalloc(&vTMem, max * max * sizeof(data_type)));
      }
      gesvdjInfo_t gesvdj_params = NULL;
      // const double tol = 1.e-14;
      // const int max_sweeps = 100;
      checkCudaErrors(cusolverDnCreateGesvdjInfo(&gesvdj_params));
      // checkCudaErrors(cusolverDnXgesvdjSetTolerance(gesvdj_params, tol));
      // checkCudaErrors(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps));

      cytnx_int32 lwork = 0;
      void *d_work = nullptr;
      checkCudaErrors(cusolverDnZgesvdj_bufferSize(
        cusolverH, jobz, econ, N, M, (d_data_type *)Mij, ldA, (cytnx_double *)S->Mem,
        (d_data_type *)vTMem, ldu, (d_data_type *)UMem, ldvT, &lwork, gesvdj_params));

      checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(data_type) * lwork));

      cytnx_int32 *devinfo;
      checkCudaErrors(cudaMalloc((void **)&devinfo, sizeof(cytnx_int32)));
      checkCudaErrors(cudaMemset(devinfo, 0, sizeof(cytnx_int32)));

      checkCudaErrors(cusolverDnZgesvdj(cusolverH, jobz, econ, N, M, (d_data_type *)Mij, ldA,
                                        (cytnx_double *)S->Mem, (d_data_type *)vTMem, ldu,
                                        (d_data_type *)UMem, ldvT, (d_data_type *)d_work, lwork,
                                        devinfo, gesvdj_params));
      if (jobz == CUSOLVER_EIG_MODE_VECTOR) {
        U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});
        linalg_internal::cuConj_inplace_internal_cd(U, M * min);
      }

      cytnx_int32 info;
      // get info
      checkCudaErrors(cudaMemcpy(&info, devinfo, sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(
        info != 0, "%s %d %s", "Error in cuBlas function 'cusolverDnZgesvdj': cuBlas INFO = ", info,
        "If info>0, possibly svd not converge, if info<0, see cusolver manual for more info.");

      checkCudaErrors(cudaFree(d_work));
      checkCudaErrors(cudaFree(Mij));
      if (UMem != nullptr and U->dtype == Type.Void) {
        checkCudaErrors(cudaFree(UMem));
      }
      if (vTMem != nullptr and vT->dtype == Type.Void) {
        checkCudaErrors(cudaFree(vTMem));
      }
      checkCudaErrors(cudaFree(devinfo));
      checkCudaErrors(cusolverDnDestroy(cusolverH));
    }
    void cuGeSvd_internal_cf(const boost::intrusive_ptr<Storage_base> &in,
                             boost::intrusive_ptr<Storage_base> &U,
                             boost::intrusive_ptr<Storage_base> &vT,
                             boost::intrusive_ptr<Storage_base> &S, const cytnx_int64 &M,
                             const cytnx_int64 &N) {
      using data_type = cytnx_complex64;
      using d_data_type = cuFloatComplex;
      cusolverEigMode_t jobz;
      // if U and vT are NULL ptr, then it will not be computed.
      jobz = (U->dtype == Type.Void and vT->dtype == Type.Void) ? CUSOLVER_EIG_MODE_NOVECTOR
                                                                : CUSOLVER_EIG_MODE_VECTOR;

      // const int econ = 0; /* i.e. 'A' in gesvd  */
      cytnx_int32 econ = 1; /* i.e. 'S' in gesvd  */

      // create handles:
      cusolverDnHandle_t cusolverH = NULL;
      checkCudaErrors(cusolverDnCreate(&cusolverH));

      cuDoubleComplex *Mij;
      checkCudaErrors(cudaMalloc((void **)&Mij, M * N * sizeof(data_type)));
      checkCudaErrors(
        cudaMemcpy(Mij, in->Mem, sizeof(data_type) * M * N, cudaMemcpyDeviceToDevice));

      cytnx_int64 min = std::min(M, N);
      cytnx_int64 max = std::max(M, N);
      cytnx_int64 ldA = N, ldu = N, ldvT = M;

      void *UMem = nullptr, *vTMem = nullptr;
      if (U->Mem) {
        UMem = U->Mem;
      } else {
        if (jobz == CUSOLVER_EIG_MODE_VECTOR)
          checkCudaErrors(cudaMalloc(&UMem, max * max * sizeof(data_type)));
      }
      if (vT->Mem) {
        vTMem = vT->Mem;
      } else {
        if (jobz == CUSOLVER_EIG_MODE_VECTOR)
          checkCudaErrors(cudaMalloc(&vTMem, max * max * sizeof(data_type)));
      }
      gesvdjInfo_t gesvdj_params = NULL;
      // const double tol = 1.e-14;
      // const int max_sweeps = 100;
      checkCudaErrors(cusolverDnCreateGesvdjInfo(&gesvdj_params));
      // checkCudaErrors(cusolverDnXgesvdjSetTolerance(gesvdj_params, tol));
      // checkCudaErrors(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps));

      cytnx_int32 lwork = 0;
      void *d_work = nullptr;
      checkCudaErrors(cusolverDnCgesvdj_bufferSize(
        cusolverH, jobz, econ, N, M, (d_data_type *)Mij, ldA, (cytnx_float *)S->Mem,
        (d_data_type *)vTMem, ldu, (d_data_type *)UMem, ldvT, &lwork, gesvdj_params));

      checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(data_type) * lwork));

      cytnx_int32 *devinfo;
      checkCudaErrors(cudaMalloc((void **)&devinfo, sizeof(cytnx_int32)));
      checkCudaErrors(cudaMemset(devinfo, 0, sizeof(cytnx_int32)));

      checkCudaErrors(cusolverDnCgesvdj(cusolverH, jobz, econ, N, M, (d_data_type *)Mij, ldA,
                                        (cytnx_float *)S->Mem, (d_data_type *)vTMem, ldu,
                                        (d_data_type *)UMem, ldvT, (d_data_type *)d_work, lwork,
                                        devinfo, gesvdj_params));
      if (jobz == CUSOLVER_EIG_MODE_VECTOR) {
        U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});
        linalg_internal::cuConj_inplace_internal_cf(U, M * min);
      }

      cytnx_int32 info;
      // get info
      checkCudaErrors(cudaMemcpy(&info, devinfo, sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(
        info != 0, "%s %d %s", "Error in cuBlas function 'cusolverDnCgesvdj': cuBlas INFO = ", info,
        "If info>0, possibly svd not converge, if info<0, see cusolver manual for more info.");

      checkCudaErrors(cudaFree(d_work));
      checkCudaErrors(cudaFree(Mij));
      if (UMem != nullptr and U->dtype == Type.Void) {
        checkCudaErrors(cudaFree(UMem));
      }
      if (vTMem != nullptr and vT->dtype == Type.Void) {
        checkCudaErrors(cudaFree(vTMem));
      }
      checkCudaErrors(cudaFree(devinfo));
      checkCudaErrors(cusolverDnDestroy(cusolverH));
    }
    void cuGeSvd_internal_d(const boost::intrusive_ptr<Storage_base> &in,
                            boost::intrusive_ptr<Storage_base> &U,
                            boost::intrusive_ptr<Storage_base> &vT,
                            boost::intrusive_ptr<Storage_base> &S, const cytnx_int64 &M,
                            const cytnx_int64 &N) {
      using data_type = cytnx_double;
      cusolverEigMode_t jobz;
      // if U and vT are NULL ptr, then it will not be computed.
      jobz = (U->dtype == Type.Void and vT->dtype == Type.Void) ? CUSOLVER_EIG_MODE_NOVECTOR
                                                                : CUSOLVER_EIG_MODE_VECTOR;

      // const int econ = 0; /* i.e. 'A' in gesvd  */
      cytnx_int32 econ = 1; /* i.e. 'S' in gesvd  */

      // create handles:
      cusolverDnHandle_t cusolverH = NULL;
      checkCudaErrors(cusolverDnCreate(&cusolverH));

      cuDoubleComplex *Mij;
      checkCudaErrors(cudaMalloc((void **)&Mij, M * N * sizeof(data_type)));
      checkCudaErrors(
        cudaMemcpy(Mij, in->Mem, sizeof(data_type) * M * N, cudaMemcpyDeviceToDevice));

      cytnx_int64 min = std::min(M, N);
      cytnx_int64 max = std::max(M, N);
      cytnx_int64 ldA = N, ldu = N, ldvT = M;

      void *UMem = nullptr, *vTMem = nullptr;
      if (U->Mem) {
        UMem = U->Mem;
      } else {
        if (jobz == CUSOLVER_EIG_MODE_VECTOR)
          checkCudaErrors(cudaMalloc(&UMem, max * max * sizeof(data_type)));
      }
      if (vT->Mem) {
        vTMem = vT->Mem;
      } else {
        if (jobz == CUSOLVER_EIG_MODE_VECTOR)
          checkCudaErrors(cudaMalloc(&vTMem, max * max * sizeof(data_type)));
      }
      gesvdjInfo_t gesvdj_params = NULL;
      // const double tol = 1.e-14;
      // const int max_sweeps = 100;
      checkCudaErrors(cusolverDnCreateGesvdjInfo(&gesvdj_params));
      // checkCudaErrors(cusolverDnXgesvdjSetTolerance(gesvdj_params, tol));
      // checkCudaErrors(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps));

      cytnx_int32 lwork = 0;
      void *d_work = nullptr;
      checkCudaErrors(cusolverDnDgesvdj_bufferSize(
        cusolverH, jobz, econ, N, M, (data_type *)Mij, ldA, (data_type *)S->Mem, (data_type *)vTMem,
        ldu, (data_type *)UMem, ldvT, &lwork, gesvdj_params));

      checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(data_type) * lwork));

      cytnx_int32 *devinfo;
      checkCudaErrors(cudaMalloc((void **)&devinfo, sizeof(cytnx_int32)));
      checkCudaErrors(cudaMemset(devinfo, 0, sizeof(cytnx_int32)));

      checkCudaErrors(cusolverDnDgesvdj(
        cusolverH, jobz, econ, N, M, (data_type *)Mij, ldA, (data_type *)S->Mem, (data_type *)vTMem,
        ldu, (data_type *)UMem, ldvT, (data_type *)d_work, lwork, devinfo, gesvdj_params));
      if (jobz == CUSOLVER_EIG_MODE_VECTOR) {
        U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});
      }

      cytnx_int32 info;
      // get info
      checkCudaErrors(cudaMemcpy(&info, devinfo, sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(
        info != 0, "%s %d %s", "Error in cuBlas function 'cusolverDnDgesvdj': cuBlas INFO = ", info,
        "If info>0, possibly svd not converge, if info<0, see cusolver manual for more info.");

      checkCudaErrors(cudaFree(d_work));
      checkCudaErrors(cudaFree(Mij));
      if (UMem != nullptr and U->dtype == Type.Void) {
        checkCudaErrors(cudaFree(UMem));
      }
      if (vTMem != nullptr and vT->dtype == Type.Void) {
        checkCudaErrors(cudaFree(vTMem));
      }
      checkCudaErrors(cudaFree(devinfo));
      checkCudaErrors(cusolverDnDestroy(cusolverH));
    }
    void cuGeSvd_internal_f(const boost::intrusive_ptr<Storage_base> &in,
                            boost::intrusive_ptr<Storage_base> &U,
                            boost::intrusive_ptr<Storage_base> &vT,
                            boost::intrusive_ptr<Storage_base> &S, const cytnx_int64 &M,
                            const cytnx_int64 &N) {
      using data_type = cytnx_float;
      cusolverEigMode_t jobz;
      // if U and vT are NULL ptr, then it will not be computed.
      jobz = (U->dtype == Type.Void and vT->dtype == Type.Void) ? CUSOLVER_EIG_MODE_NOVECTOR
                                                                : CUSOLVER_EIG_MODE_VECTOR;

      // const int econ = 0; /* i.e. 'A' in gesvd  */
      cytnx_int32 econ = 1; /* i.e. 'S' in gesvd  */

      // create handles:
      cusolverDnHandle_t cusolverH = NULL;
      checkCudaErrors(cusolverDnCreate(&cusolverH));

      cuDoubleComplex *Mij;
      checkCudaErrors(cudaMalloc((void **)&Mij, M * N * sizeof(data_type)));
      checkCudaErrors(
        cudaMemcpy(Mij, in->Mem, sizeof(data_type) * M * N, cudaMemcpyDeviceToDevice));

      cytnx_int64 min = std::min(M, N);
      cytnx_int64 max = std::max(M, N);
      cytnx_int64 ldA = N, ldu = N, ldvT = M;

      void *UMem = nullptr, *vTMem = nullptr;
      if (U->Mem) {
        UMem = U->Mem;
      } else {
        if (jobz == CUSOLVER_EIG_MODE_VECTOR)
          checkCudaErrors(cudaMalloc(&UMem, max * max * sizeof(data_type)));
      }
      if (vT->Mem) {
        vTMem = vT->Mem;
      } else {
        if (jobz == CUSOLVER_EIG_MODE_VECTOR)
          checkCudaErrors(cudaMalloc(&vTMem, max * max * sizeof(data_type)));
      }
      gesvdjInfo_t gesvdj_params = NULL;
      // const double tol = 1.e-14;
      // const int max_sweeps = 100;
      checkCudaErrors(cusolverDnCreateGesvdjInfo(&gesvdj_params));
      // checkCudaErrors(cusolverDnXgesvdjSetTolerance(gesvdj_params, tol));
      // checkCudaErrors(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps));

      cytnx_int32 lwork = 0;
      void *d_work = nullptr;
      checkCudaErrors(cusolverDnSgesvdj_bufferSize(
        cusolverH, jobz, econ, N, M, (data_type *)Mij, ldA, (data_type *)S->Mem, (data_type *)vTMem,
        ldu, (data_type *)UMem, ldvT, &lwork, gesvdj_params));

      checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(data_type) * lwork));

      cytnx_int32 *devinfo;
      checkCudaErrors(cudaMalloc((void **)&devinfo, sizeof(cytnx_int32)));
      checkCudaErrors(cudaMemset(devinfo, 0, sizeof(cytnx_int32)));

      checkCudaErrors(cusolverDnSgesvdj(
        cusolverH, jobz, econ, N, M, (data_type *)Mij, ldA, (data_type *)S->Mem, (data_type *)vTMem,
        ldu, (data_type *)UMem, ldvT, (data_type *)d_work, lwork, devinfo, gesvdj_params));
      if (jobz == CUSOLVER_EIG_MODE_VECTOR) {
        U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});
      }

      cytnx_int32 info;
      // get info
      checkCudaErrors(cudaMemcpy(&info, devinfo, sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(
        info != 0, "%s %d %s", "Error in cuBlas function 'cusolverDnSgesvdj': cuBlas INFO = ", info,
        "If info>0, possibly svd not converge, if info<0, see cusolver manual for more info.");

      checkCudaErrors(cudaFree(d_work));
      checkCudaErrors(cudaFree(Mij));
      if (UMem != nullptr and U->dtype == Type.Void) {
        checkCudaErrors(cudaFree(UMem));
      }
      if (vTMem != nullptr and vT->dtype == Type.Void) {
        checkCudaErrors(cudaFree(vTMem));
      }
      checkCudaErrors(cudaFree(devinfo));
      checkCudaErrors(cusolverDnDestroy(cusolverH));
    }

  }  // namespace linalg_internal
}  // namespace cytnx
