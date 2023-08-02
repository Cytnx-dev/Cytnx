#include "linalg/linalg_internal_gpu/cuGeSvd_internal.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// cuGeSvd
    void cuGeSvd_internal_cd(const boost::intrusive_ptr<Storage_base> &in,
                             boost::intrusive_ptr<Storage_base> &U,
                             boost::intrusive_ptr<Storage_base> &vT,
                             boost::intrusive_ptr<Storage_base> &S, const cytnx_int64 &M,
                             const cytnx_int64 &N) {
      assert(sizeof(cuDoubleComplex) == sizeof(cytnx_complex128));
      signed char jobu, jobv;

      // if U and vT are NULL ptr, then it will not be computed.
      jobu = (U->dtype == Type.Void) ? 'N' : 'S';
      jobv = (vT->dtype == Type.Void) ? 'N' : 'S';

      // create handles:
      cusolverDnHandle_t cusolverH = NULL;
      checkCudaErrors(cusolverDnCreate(&cusolverH));

      cuDoubleComplex *Mij;
      checkCudaErrors(cudaMalloc((void **)&Mij, M * N * sizeof(cuDoubleComplex)));
      checkCudaErrors(
        cudaMemcpy(Mij, in->Mem, sizeof(cuDoubleComplex) * M * N, cudaMemcpyDeviceToDevice));

      cytnx_int32 min = std::min(M, N);
      cytnx_int32 max = std::max(M, N);
      cytnx_int32 ldA = N, ldu = N, ldvT = min;
      if (N < M) {
        ldA = M, ldu = M, ldvT = min;
      }
      cytnx_int32 lwork = 0;

      void *UMem, *vTMem;
      if (U->Mem) {
        UMem = U->Mem;
      } else {
        if (jobu == 'S') checkCudaErrors(cudaMalloc(&UMem, max * max * sizeof(cuDoubleComplex)));
      }
      if (vT->Mem) {
        vTMem = vT->Mem;
      } else {
        if (jobv == 'S') checkCudaErrors(cudaMalloc(&vTMem, max * max * sizeof(cuDoubleComplex)));
      }

      // query working space :
      checkCudaErrors(cusolverDnZgesvd_bufferSize(cusolverH, N, M, &lwork));

      // allocate working space:
      cuDoubleComplex *work;
      cytnx_double *rwork = NULL;
      checkCudaErrors(cudaMalloc((void **)&work, lwork * sizeof(cuDoubleComplex)));
      // checkCudaErrors(cudaMalloc((void**)&rwork,(min-1)*sizeof(cytnx_double64)));

      cytnx_int32 *devinfo;
      checkCudaErrors(cudaMalloc((void **)&devinfo, sizeof(cytnx_int32)));
      checkCudaErrors(cudaMemset(devinfo, 0, sizeof(cytnx_int32)));

      cytnx_int32 info;
      /// compute:
      if (N >= M) {
        checkCudaErrors(cusolverDnZgesvd(cusolverH, jobv, jobu, N, M, (cuDoubleComplex *)Mij, ldA,
                                         (cytnx_double *)S->Mem, (cuDoubleComplex *)vTMem, ldu,
                                         (cuDoubleComplex *)UMem, ldvT, work, lwork, rwork,
                                         devinfo));
      } else {
        checkCudaErrors(cusolverDnZgesvd(cusolverH, jobu, jobv, M, N, (cuDoubleComplex *)Mij, ldA,
                                         (cytnx_double *)S->Mem, (cuDoubleComplex *)UMem, ldu,
                                         (cuDoubleComplex *)vTMem, ldvT, work, lwork, rwork,
                                         devinfo));
        if (U->dtype != Type.Void)
          U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});
        // linalg_internal::cuConj_inplace_internal_cd(U,M*min);
        if (vT->dtype != Type.Void)
          vT->Move_memory_({(cytnx_uint64)N, (cytnx_uint64)min}, {1, 0}, {1, 0});
        linalg_internal::cuConj_inplace_internal_cd(vT, N * min);
      }
      // get info
      checkCudaErrors(cudaMemcpy(&info, devinfo, sizeof(cytnx_int32), cudaMemcpyDeviceToHost));
      cytnx_error_msg(
        info != 0, "%s %d %s", "Error in cuBlas function 'cusolverDnZgesvd': cuBlas INFO = ", info,
        "If info>0, possibly svd not converge, if info<0, see cusolver manual for more info.");

      checkCudaErrors(cudaFree(work));
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
      assert(sizeof(cuFloatComplex) == sizeof(cytnx_complex64));
      signed char jobu, jobv;

      // if U and vT are NULL ptr, then it will not be computed.
      jobu = (U->dtype == Type.Void) ? 'N' : 'S';
      jobv = (vT->dtype == Type.Void) ? 'N' : 'S';

      // create handles:
      cusolverDnHandle_t cusolverH = NULL;
      checkCudaErrors(cusolverDnCreate(&cusolverH));

      cuFloatComplex *Mij;
      checkCudaErrors(cudaMalloc((void **)&Mij, M * N * sizeof(cuFloatComplex)));
      checkCudaErrors(
        cudaMemcpy(Mij, in->Mem, sizeof(cytnx_complex64) * M * N, cudaMemcpyDeviceToDevice));

      cytnx_int32 min = std::min(M, N);
      cytnx_int32 max = std::max(M, N);
      cytnx_int32 ldA = N, ldu = N, ldvT = min;
      if (N < M) {
        ldA = M, ldu = M, ldvT = min;
      }
      cytnx_int32 lwork = 0;

      void *UMem, *vTMem;
      if (U->Mem) {
        UMem = U->Mem;
      } else {
        if (jobu == 'S') checkCudaErrors(cudaMalloc(&UMem, max * max * sizeof(cuFloatComplex)));
      }
      if (vT->Mem) {
        vTMem = vT->Mem;
      } else {
        if (jobv == 'S') checkCudaErrors(cudaMalloc(&vTMem, max * max * sizeof(cuFloatComplex)));
      }

      // query working space :
      checkCudaErrors(cusolverDnCgesvd_bufferSize(cusolverH, N, M, &lwork));

      // allocate working space:
      cuFloatComplex *work;
      cytnx_float *rwork = NULL;
      checkCudaErrors(cudaMalloc((void **)&work, lwork * sizeof(cuFloatComplex)));
      // checkCudaErrors(cudaMalloc((void**)&rwork,(min-1)*sizeof(cytnx_float64)));

      cytnx_int32 *devinfo;
      checkCudaErrors(cudaMalloc((void **)&devinfo, sizeof(cytnx_int32)));
      checkCudaErrors(cudaMemset(devinfo, 0, sizeof(cytnx_int32)));

      cytnx_int32 info;
      /// compute:
      if (N >= M) {
        checkCudaErrors(cusolverDnCgesvd(
          cusolverH, jobv, jobu, N, M, (cuFloatComplex *)Mij, ldA, (cytnx_float *)S->Mem,
          (cuFloatComplex *)vTMem, ldu, (cuFloatComplex *)UMem, ldvT, work, lwork, rwork, devinfo));
      } else {
        checkCudaErrors(cusolverDnCgesvd(
          cusolverH, jobu, jobv, M, N, (cuFloatComplex *)Mij, ldA, (cytnx_float *)S->Mem,
          (cuFloatComplex *)UMem, ldu, (cuFloatComplex *)vTMem, ldvT, work, lwork, rwork, devinfo));
        if (U->dtype != Type.Void)
          U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});
        // linalg_internal::cuConj_inplace_internal_cf(U,M*min);
        if (vT->dtype != Type.Void)
          vT->Move_memory_({(cytnx_uint64)N, (cytnx_uint64)min}, {1, 0}, {1, 0});
        linalg_internal::cuConj_inplace_internal_cf(vT, N * min);
      }

      // get info
      checkCudaErrors(cudaMemcpy(&info, devinfo, sizeof(cytnx_int32), cudaMemcpyDeviceToHost));
      cytnx_error_msg(
        info != 0, "%s %d %s", "Error in cuBlas function 'cusolverDnCgesvd': cuBlas INFO = ", info,
        "If info>0, possibly svd not converge, if info<0, see cusolver manual for more info.");

      checkCudaErrors(cudaFree(work));
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
      signed char jobu, jobv;

      // if U and vT are NULL ptr, then it will not be computed.
      jobu = (U->dtype == Type.Void) ? 'N' : 'S';
      jobv = (vT->dtype == Type.Void) ? 'N' : 'S';

      // create handles:
      cusolverDnHandle_t cusolverH = NULL;
      checkCudaErrors(cusolverDnCreate(&cusolverH));

      cytnx_double *Mij;
      checkCudaErrors(cudaMalloc((void **)&Mij, M * N * sizeof(cytnx_double)));
      checkCudaErrors(
        cudaMemcpy(Mij, in->Mem, sizeof(cytnx_double) * M * N, cudaMemcpyDeviceToDevice));

      cytnx_int32 min = std::min(M, N);
      cytnx_int32 max = std::max(M, N);
      cytnx_int32 ldA = N, ldu = N, ldvT = min;
      if (N < M) {
        ldA = M, ldu = M, ldvT = min;
      }
      cytnx_int32 lwork = 0;

      void *UMem, *vTMem;
      if (U->Mem) {
        UMem = U->Mem;
      } else {
        if (jobu == 'S') checkCudaErrors(cudaMalloc(&UMem, max * max * sizeof(cytnx_double)));
      }
      if (vT->Mem) {
        vTMem = vT->Mem;
      } else {
        if (jobv == 'S') checkCudaErrors(cudaMalloc(&vTMem, max * max * sizeof(cytnx_double)));
      }

      // query working space :
      checkCudaErrors(cusolverDnDgesvd_bufferSize(cusolverH, N, M, &lwork));

      // allocate working space:
      cytnx_double *work;
      cytnx_double *rwork = NULL;
      checkCudaErrors(cudaMalloc((void **)&work, lwork * sizeof(cytnx_double)));
      checkCudaErrors(cudaMalloc((void **)&rwork, (min - 1) * sizeof(cytnx_double)));

      cytnx_int32 *devinfo;
      checkCudaErrors(cudaMalloc((void **)&devinfo, sizeof(cytnx_int32)));
      checkCudaErrors(cudaMemset(devinfo, 0, sizeof(cytnx_int32)));

      cytnx_int32 info;
      /// compute:
      if (N >= M) {
        cusolverDnDgesvd(cusolverH, jobv, jobu, N, M, (cytnx_double *)Mij, ldA,
                         (cytnx_double *)S->Mem, (cytnx_double *)vTMem, ldu, (cytnx_double *)UMem,
                         ldvT, work, lwork, rwork, devinfo);
      } else {
        cusolverDnDgesvd(cusolverH, jobu, jobv, M, N, (cytnx_double *)Mij, ldA,
                         (cytnx_double *)S->Mem, (cytnx_double *)UMem, ldu, (cytnx_double *)vTMem,
                         ldvT, work, lwork, rwork, devinfo);
        if (U->dtype != Type.Void)
          U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});
        if (vT->dtype != Type.Void)
          vT->Move_memory_({(cytnx_uint64)N, (cytnx_uint64)min}, {1, 0}, {1, 0});
      }

      // get info
      checkCudaErrors(cudaMemcpy(&info, devinfo, sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(
        info != 0, "%s %d %s", "Error in cuBlas function 'cusolverDnDgesvd': cuBlas INFO = ", info,
        "If info>0, possibly svd not converge, if info<0, see cusolver manual for more info.");

      checkCudaErrors(cudaFree(work));
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
      signed char jobu, jobv;

      // if U and vT are NULL ptr, then it will not be computed.
      jobu = (U->dtype == Type.Void) ? 'N' : 'S';
      jobv = (vT->dtype == Type.Void) ? 'N' : 'S';

      // create handles:
      cusolverDnHandle_t cusolverH = NULL;
      checkCudaErrors(cusolverDnCreate(&cusolverH));

      cytnx_float *Mij;
      checkCudaErrors(cudaMalloc((void **)&Mij, M * N * sizeof(cytnx_float)));
      checkCudaErrors(
        cudaMemcpy(Mij, in->Mem, sizeof(cytnx_float) * M * N, cudaMemcpyDeviceToDevice));

      cytnx_int32 min = std::min(M, N);
      cytnx_int32 max = std::max(M, N);
      cytnx_int32 ldA = N, ldu = N, ldvT = min;
      if (N < M) {
        ldA = M, ldu = M, ldvT = min;
      }
      cytnx_int32 lwork = 0;

      void *UMem, *vTMem;
      if (U->Mem) {
        UMem = U->Mem;
      } else {
        if (jobu == 'S') checkCudaErrors(cudaMalloc(&UMem, max * max * sizeof(cytnx_float)));
      }
      if (vT->Mem) {
        vTMem = vT->Mem;
      } else {
        if (jobv == 'S') checkCudaErrors(cudaMalloc(&vTMem, max * max * sizeof(cytnx_float)));
      }

      // query working space :
      checkCudaErrors(cusolverDnSgesvd_bufferSize(cusolverH, N, M, &lwork));

      // allocate working space:
      cytnx_float *work;
      cytnx_float *rwork = NULL;
      checkCudaErrors(cudaMalloc((void **)&work, lwork * sizeof(cytnx_float)));
      // checkCudaErrors(cudaMalloc((void**)&rwork,(min-1)*sizeof(cytnx_float64)));

      cytnx_int32 *devinfo;
      checkCudaErrors(cudaMalloc((void **)&devinfo, sizeof(cytnx_int32)));
      checkCudaErrors(cudaMemset(devinfo, 0, sizeof(cytnx_int32)));

      cytnx_int32 info;
      /// compute:
      if (N >= M) {
        checkCudaErrors(cusolverDnSgesvd(cusolverH, jobv, jobu, N, M, (cytnx_float *)Mij, ldA,
                                         (cytnx_float *)S->Mem, (cytnx_float *)vTMem, ldu,
                                         (cytnx_float *)UMem, ldvT, work, lwork, rwork, devinfo));
      } else {
        checkCudaErrors(cusolverDnSgesvd(cusolverH, jobu, jobv, M, N, (cytnx_float *)Mij, ldA,
                                         (cytnx_float *)S->Mem, (cytnx_float *)UMem, ldu,
                                         (cytnx_float *)vTMem, ldvT, work, lwork, rwork, devinfo));
        if (U->dtype != Type.Void)
          U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});
        if (vT->dtype != Type.Void)
          vT->Move_memory_({(cytnx_uint64)N, (cytnx_uint64)min}, {1, 0}, {1, 0});
      }

      // get info
      checkCudaErrors(cudaMemcpy(&info, devinfo, sizeof(cytnx_int32), cudaMemcpyDeviceToHost));
      cytnx_error_msg(
        info != 0, "%s %d %s", "Error in cuBlas function 'cusolverDnSgesvd': cuBlas INFO = ", info,
        "If info>0, possibly svd not converge, if info<0, see cusolver manual for more info.");

      checkCudaErrors(cudaFree(work));
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
