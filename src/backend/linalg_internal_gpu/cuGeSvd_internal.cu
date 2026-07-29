#include "cuGeSvd_internal.hpp"
#include "cuConj_inplace_internal.hpp"
#include "backend/utils_internal_gpu/cuScopedResource_gpu.hpp"

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
      // if U and vT are void, then it will not be computed.
      jobz = (U->dtype() == Type.Void and vT->dtype() == Type.Void) ? CUSOLVER_EIG_MODE_NOVECTOR
                                                                    : CUSOLVER_EIG_MODE_VECTOR;

      // const int econ = 0; /* i.e. 'A' in gesvd  */
      cytnx_int32 econ = 1; /* i.e. 'S' in gesvd  */

      // create handles:
      // Scoped resources (#1145, #1146): the info check below throws, so ownership -- not the
      // cleanup block that used to sit at the tail of this function -- is what prevents a leak.
      utils_internal::CusolverDnHandle cusolverH;

      utils_internal::DeviceBuffer<data_type> Mij(M * N);
      checkCudaErrors(
        cudaMemcpy(Mij.get(), in->data(), sizeof(data_type) * M * N, cudaMemcpyDeviceToDevice));

      cytnx_int64 min = std::min(M, N);
      cytnx_int64 max = std::max(M, N);
      cytnx_int64 ldA = N, ldu = N, ldvT = M;

      // UMem/vTMem alias U's and vT's storage when it exists and are temporaries otherwise; the
      // owned_* buffers are non-empty only in the temporary case, matching the old conditional
      // cudaFree on `dtype() == Type.Void`.
      void *UMem = nullptr, *vTMem = nullptr;
      utils_internal::DeviceBuffer<data_type> owned_UMem, owned_vTMem;
      if (U->data()) {
        UMem = U->data();
      } else {
        if (jobz == CUSOLVER_EIG_MODE_VECTOR) {
          owned_UMem = utils_internal::DeviceBuffer<data_type>(max * max);
          UMem = owned_UMem.get();
        }
      }
      if (vT->data()) {
        vTMem = vT->data();
      } else {
        if (jobz == CUSOLVER_EIG_MODE_VECTOR) {
          owned_vTMem = utils_internal::DeviceBuffer<data_type>(max * max);
          vTMem = owned_vTMem.get();
        }
      }
      // const double tol = 1.e-14;
      // const int max_sweeps = 100;
      utils_internal::GesvdjInfo gesvdj_params;
      // checkCudaErrors(cusolverDnXgesvdjSetTolerance(gesvdj_params, tol));
      // checkCudaErrors(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps));

      cytnx_int32 lwork = 0;
      checkCudaErrors(cusolverDnZgesvdj_bufferSize(
        cusolverH.get(), jobz, econ, N, M, (d_data_type *)Mij.get(), ldA, (cytnx_double *)S->data(),
        (d_data_type *)vTMem, ldu, (d_data_type *)UMem, ldvT, &lwork, gesvdj_params.get()));

      utils_internal::DeviceBuffer<data_type> d_work(lwork);

      utils_internal::DeviceBuffer<cytnx_int32> devinfo(1);
      checkCudaErrors(cudaMemset(devinfo.get(), 0, sizeof(cytnx_int32)));

      checkCudaErrors(cusolverDnZgesvdj(cusolverH.get(), jobz, econ, N, M, (d_data_type *)Mij.get(),
                                        ldA, (cytnx_double *)S->data(), (d_data_type *)vTMem, ldu,
                                        (d_data_type *)UMem, ldvT, (d_data_type *)d_work.get(),
                                        lwork, devinfo.get(), gesvdj_params.get()));
      if (U->data() and jobz == CUSOLVER_EIG_MODE_VECTOR) {
        U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});
        linalg_internal::cuConj_inplace_internal_cd(U, M * min);
      }

      cytnx_int32 info;
      // get info
      checkCudaErrors(
        cudaMemcpy(&info, devinfo.get(), sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(
        info != 0, "%s %d %s", "Error in cuBlas function 'cusolverDnZgesvdj': cuBlas INFO = ", info,
        "If info>0, possibly svd not converge, if info<0, see cusolver manual for more info.");
    }
    void cuGeSvd_internal_cf(const boost::intrusive_ptr<Storage_base> &in,
                             boost::intrusive_ptr<Storage_base> &U,
                             boost::intrusive_ptr<Storage_base> &vT,
                             boost::intrusive_ptr<Storage_base> &S, const cytnx_int64 &M,
                             const cytnx_int64 &N) {
      using data_type = cytnx_complex64;
      using d_data_type = cuFloatComplex;
      cusolverEigMode_t jobz;
      // if U and vT are void, then it will not be computed.
      jobz = (U->dtype() == Type.Void and vT->dtype() == Type.Void) ? CUSOLVER_EIG_MODE_NOVECTOR
                                                                    : CUSOLVER_EIG_MODE_VECTOR;

      // const int econ = 0; /* i.e. 'A' in gesvd  */
      cytnx_int32 econ = 1; /* i.e. 'S' in gesvd  */

      // create handles:
      // Scoped resources (#1145, #1146): the info check below throws, so ownership -- not the
      // cleanup block that used to sit at the tail of this function -- is what prevents a leak.
      utils_internal::CusolverDnHandle cusolverH;

      utils_internal::DeviceBuffer<data_type> Mij(M * N);
      checkCudaErrors(
        cudaMemcpy(Mij.get(), in->data(), sizeof(data_type) * M * N, cudaMemcpyDeviceToDevice));

      cytnx_int64 min = std::min(M, N);
      cytnx_int64 max = std::max(M, N);
      cytnx_int64 ldA = N, ldu = N, ldvT = M;

      // UMem/vTMem alias U's and vT's storage when it exists and are temporaries otherwise; the
      // owned_* buffers are non-empty only in the temporary case, matching the old conditional
      // cudaFree on `dtype() == Type.Void`.
      void *UMem = nullptr, *vTMem = nullptr;
      utils_internal::DeviceBuffer<data_type> owned_UMem, owned_vTMem;
      if (U->data()) {
        UMem = U->data();
      } else {
        if (jobz == CUSOLVER_EIG_MODE_VECTOR) {
          owned_UMem = utils_internal::DeviceBuffer<data_type>(max * max);
          UMem = owned_UMem.get();
        }
      }
      if (vT->data()) {
        vTMem = vT->data();
      } else {
        if (jobz == CUSOLVER_EIG_MODE_VECTOR) {
          owned_vTMem = utils_internal::DeviceBuffer<data_type>(max * max);
          vTMem = owned_vTMem.get();
        }
      }
      // const double tol = 1.e-14;
      // const int max_sweeps = 100;
      utils_internal::GesvdjInfo gesvdj_params;
      // checkCudaErrors(cusolverDnXgesvdjSetTolerance(gesvdj_params, tol));
      // checkCudaErrors(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps));

      cytnx_int32 lwork = 0;
      checkCudaErrors(cusolverDnCgesvdj_bufferSize(
        cusolverH.get(), jobz, econ, N, M, (d_data_type *)Mij.get(), ldA, (cytnx_float *)S->data(),
        (d_data_type *)vTMem, ldu, (d_data_type *)UMem, ldvT, &lwork, gesvdj_params.get()));

      utils_internal::DeviceBuffer<data_type> d_work(lwork);

      utils_internal::DeviceBuffer<cytnx_int32> devinfo(1);
      checkCudaErrors(cudaMemset(devinfo.get(), 0, sizeof(cytnx_int32)));

      checkCudaErrors(cusolverDnCgesvdj(cusolverH.get(), jobz, econ, N, M, (d_data_type *)Mij.get(),
                                        ldA, (cytnx_float *)S->data(), (d_data_type *)vTMem, ldu,
                                        (d_data_type *)UMem, ldvT, (d_data_type *)d_work.get(),
                                        lwork, devinfo.get(), gesvdj_params.get()));
      if (U->data() and jobz == CUSOLVER_EIG_MODE_VECTOR) {
        U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});
        linalg_internal::cuConj_inplace_internal_cf(U, M * min);
      }

      cytnx_int32 info;
      // get info
      checkCudaErrors(
        cudaMemcpy(&info, devinfo.get(), sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(
        info != 0, "%s %d %s", "Error in cuBlas function 'cusolverDnCgesvdj': cuBlas INFO = ", info,
        "If info>0, possibly svd not converge, if info<0, see cusolver manual for more info.");
    }
    void cuGeSvd_internal_d(const boost::intrusive_ptr<Storage_base> &in,
                            boost::intrusive_ptr<Storage_base> &U,
                            boost::intrusive_ptr<Storage_base> &vT,
                            boost::intrusive_ptr<Storage_base> &S, const cytnx_int64 &M,
                            const cytnx_int64 &N) {
      using data_type = cytnx_double;
      cusolverEigMode_t jobz;
      // if U and vT are void, then it will not be computed.
      jobz = (U->dtype() == Type.Void and vT->dtype() == Type.Void) ? CUSOLVER_EIG_MODE_NOVECTOR
                                                                    : CUSOLVER_EIG_MODE_VECTOR;

      // const int econ = 0; /* i.e. 'A' in gesvd  */
      cytnx_int32 econ = 1; /* i.e. 'S' in gesvd  */

      // create handles:
      // Scoped resources (#1145, #1146): the info check below throws, so ownership -- not the
      // cleanup block that used to sit at the tail of this function -- is what prevents a leak.
      utils_internal::CusolverDnHandle cusolverH;

      utils_internal::DeviceBuffer<data_type> Mij(M * N);
      checkCudaErrors(
        cudaMemcpy(Mij.get(), in->data(), sizeof(data_type) * M * N, cudaMemcpyDeviceToDevice));

      cytnx_int64 min = std::min(M, N);
      cytnx_int64 max = std::max(M, N);
      cytnx_int64 ldA = N, ldu = N, ldvT = M;

      // UMem/vTMem alias U's and vT's storage when it exists and are temporaries otherwise; the
      // owned_* buffers are non-empty only in the temporary case, matching the old conditional
      // cudaFree on `dtype() == Type.Void`.
      void *UMem = nullptr, *vTMem = nullptr;
      utils_internal::DeviceBuffer<data_type> owned_UMem, owned_vTMem;
      if (U->data()) {
        UMem = U->data();
      } else {
        if (jobz == CUSOLVER_EIG_MODE_VECTOR) {
          owned_UMem = utils_internal::DeviceBuffer<data_type>(max * max);
          UMem = owned_UMem.get();
        }
      }
      if (vT->data()) {
        vTMem = vT->data();
      } else {
        if (jobz == CUSOLVER_EIG_MODE_VECTOR) {
          owned_vTMem = utils_internal::DeviceBuffer<data_type>(max * max);
          vTMem = owned_vTMem.get();
        }
      }
      // const double tol = 1.e-14;
      // const int max_sweeps = 100;
      utils_internal::GesvdjInfo gesvdj_params;
      // checkCudaErrors(cusolverDnXgesvdjSetTolerance(gesvdj_params, tol));
      // checkCudaErrors(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps));

      cytnx_int32 lwork = 0;
      checkCudaErrors(cusolverDnDgesvdj_bufferSize(
        cusolverH.get(), jobz, econ, N, M, (data_type *)Mij.get(), ldA, (data_type *)S->data(),
        (data_type *)vTMem, ldu, (data_type *)UMem, ldvT, &lwork, gesvdj_params.get()));

      utils_internal::DeviceBuffer<data_type> d_work(lwork);

      utils_internal::DeviceBuffer<cytnx_int32> devinfo(1);
      checkCudaErrors(cudaMemset(devinfo.get(), 0, sizeof(cytnx_int32)));

      checkCudaErrors(cusolverDnDgesvdj(cusolverH.get(), jobz, econ, N, M, (data_type *)Mij.get(),
                                        ldA, (data_type *)S->data(), (data_type *)vTMem, ldu,
                                        (data_type *)UMem, ldvT, (data_type *)d_work.get(), lwork,
                                        devinfo.get(), gesvdj_params.get()));
      if (U->data() and jobz == CUSOLVER_EIG_MODE_VECTOR) {
        U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});
      }

      cytnx_int32 info;
      // get info
      checkCudaErrors(
        cudaMemcpy(&info, devinfo.get(), sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(
        info != 0, "%s %d %s", "Error in cuBlas function 'cusolverDnDgesvdj': cuBlas INFO = ", info,
        "If info>0, possibly svd not converge, if info<0, see cusolver manual for more info.");
    }
    void cuGeSvd_internal_f(const boost::intrusive_ptr<Storage_base> &in,
                            boost::intrusive_ptr<Storage_base> &U,
                            boost::intrusive_ptr<Storage_base> &vT,
                            boost::intrusive_ptr<Storage_base> &S, const cytnx_int64 &M,
                            const cytnx_int64 &N) {
      using data_type = cytnx_float;
      cusolverEigMode_t jobz;
      // if U and vT are void, then it will not be computed.
      jobz = (U->dtype() == Type.Void and vT->dtype() == Type.Void) ? CUSOLVER_EIG_MODE_NOVECTOR
                                                                    : CUSOLVER_EIG_MODE_VECTOR;

      // const int econ = 0; /* i.e. 'A' in gesvd  */
      cytnx_int32 econ = 1; /* i.e. 'S' in gesvd  */

      // create handles:
      // Scoped resources (#1145, #1146): the info check below throws, so ownership -- not the
      // cleanup block that used to sit at the tail of this function -- is what prevents a leak.
      utils_internal::CusolverDnHandle cusolverH;

      utils_internal::DeviceBuffer<data_type> Mij(M * N);
      checkCudaErrors(
        cudaMemcpy(Mij.get(), in->data(), sizeof(data_type) * M * N, cudaMemcpyDeviceToDevice));

      cytnx_int64 min = std::min(M, N);
      cytnx_int64 max = std::max(M, N);
      cytnx_int64 ldA = N, ldu = N, ldvT = M;

      // UMem/vTMem alias U's and vT's storage when it exists and are temporaries otherwise; the
      // owned_* buffers are non-empty only in the temporary case, matching the old conditional
      // cudaFree on `dtype() == Type.Void`.
      void *UMem = nullptr, *vTMem = nullptr;
      utils_internal::DeviceBuffer<data_type> owned_UMem, owned_vTMem;
      if (U->data()) {
        UMem = U->data();
      } else {
        if (jobz == CUSOLVER_EIG_MODE_VECTOR) {
          owned_UMem = utils_internal::DeviceBuffer<data_type>(max * max);
          UMem = owned_UMem.get();
        }
      }
      if (vT->data()) {
        vTMem = vT->data();
      } else {
        if (jobz == CUSOLVER_EIG_MODE_VECTOR) {
          owned_vTMem = utils_internal::DeviceBuffer<data_type>(max * max);
          vTMem = owned_vTMem.get();
        }
      }
      // const double tol = 1.e-14;
      // const int max_sweeps = 100;
      utils_internal::GesvdjInfo gesvdj_params;
      // checkCudaErrors(cusolverDnXgesvdjSetTolerance(gesvdj_params, tol));
      // checkCudaErrors(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps));

      cytnx_int32 lwork = 0;
      checkCudaErrors(cusolverDnSgesvdj_bufferSize(
        cusolverH.get(), jobz, econ, N, M, (data_type *)Mij.get(), ldA, (data_type *)S->data(),
        (data_type *)vTMem, ldu, (data_type *)UMem, ldvT, &lwork, gesvdj_params.get()));

      utils_internal::DeviceBuffer<data_type> d_work(lwork);

      utils_internal::DeviceBuffer<cytnx_int32> devinfo(1);
      checkCudaErrors(cudaMemset(devinfo.get(), 0, sizeof(cytnx_int32)));

      checkCudaErrors(cusolverDnSgesvdj(cusolverH.get(), jobz, econ, N, M, (data_type *)Mij.get(),
                                        ldA, (data_type *)S->data(), (data_type *)vTMem, ldu,
                                        (data_type *)UMem, ldvT, (data_type *)d_work.get(), lwork,
                                        devinfo.get(), gesvdj_params.get()));
      if (U->data() and jobz == CUSOLVER_EIG_MODE_VECTOR) {
        U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});
      }

      cytnx_int32 info;
      // get info
      checkCudaErrors(
        cudaMemcpy(&info, devinfo.get(), sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(
        info != 0, "%s %d %s", "Error in cuBlas function 'cusolverDnSgesvdj': cuBlas INFO = ", info,
        "If info>0, possibly svd not converge, if info<0, see cusolver manual for more info.");
    }

  }  // namespace linalg_internal
}  // namespace cytnx
