#include "cuEigh_internal.hpp"
#include "cytnx_error.hpp"
#include "Type.hpp"
#include "backend/lapack_wrapper.hpp"
#include "backend/utils_internal_gpu/cuScopedResource_gpu.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// cuEigh
    void cuEigh_internal_cd(const boost::intrusive_ptr<Storage_base> &in,
                            boost::intrusive_ptr<Storage_base> &e,
                            boost::intrusive_ptr<Storage_base> &v, const cytnx_int64 &L) {
      cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
      if (v->dtype() == Type.Void) jobz = CUSOLVER_EIG_MODE_NOVECTOR;

      // create handles:
      // Scoped resources (#1146): the info check below throws, so ownership -- not the cleanup
      // block that used to sit at the tail of this function -- is what prevents a leak.
      utils_internal::CusolverDnHandle cusolverH;

      // `tA` aliases v's storage when eigenvectors are wanted and is a temporary otherwise;
      // `owned_tA` is non-empty only in the second case, matching the old conditional cudaFree.
      cytnx_complex128 *tA;
      utils_internal::DeviceBuffer<cytnx_complex128> owned_tA;
      if (v->dtype() != Type.Void) {
        tA = (cytnx_complex128 *)v->data();
        checkCudaErrors(cudaMemcpy(v->data(), in->data(),
                                   sizeof(cytnx_complex128) * cytnx_uint64(L) * L,
                                   cudaMemcpyDeviceToDevice));
      } else {
        owned_tA = utils_internal::DeviceBuffer<cytnx_complex128>(cytnx_uint64(L) * L);
        tA = owned_tA.get();
        checkCudaErrors(cudaMemcpy(tA, in->data(), sizeof(cytnx_complex128) * cytnx_uint64(L) * L,
                                   cudaMemcpyDeviceToDevice));
      }

      // query buffer:
      cytnx_int32 lwork = 0;
      cytnx_int32 b32L = L;
      checkCudaErrors(cusolverDnZheevd_bufferSize(cusolverH.get(), jobz, CUBLAS_FILL_MODE_UPPER,
                                                  b32L, (cuDoubleComplex *)tA, b32L,
                                                  (cytnx_double *)e->data(), &lwork));

      // allocate working space:
      utils_internal::DeviceBuffer<cytnx_complex128> work(lwork);

      // call :
      cytnx_int32 info;
      utils_internal::DeviceBuffer<cytnx_int32> devinfo(1);
      checkCudaErrors(cusolverDnZheevd(cusolverH.get(), jobz, CUBLAS_FILL_MODE_UPPER, b32L,
                                       (cuDoubleComplex *)tA, b32L, (cytnx_double *)e->data(),
                                       (cuDoubleComplex *)work.get(), lwork, devinfo.get()));

      // get info
      checkCudaErrors(
        cudaMemcpy(&info, devinfo.get(), sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in cuBlas function 'cusolverDnZheevd': cuBlas INFO = ", info);
    }
    void cuEigh_internal_cf(const boost::intrusive_ptr<Storage_base> &in,
                            boost::intrusive_ptr<Storage_base> &e,
                            boost::intrusive_ptr<Storage_base> &v, const cytnx_int64 &L) {
      cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
      if (v->dtype() == Type.Void) jobz = CUSOLVER_EIG_MODE_NOVECTOR;

      // create handles:
      utils_internal::CusolverDnHandle cusolverH;

      cytnx_complex64 *tA;
      utils_internal::DeviceBuffer<cytnx_complex64> owned_tA;
      if (v->dtype() != Type.Void) {
        tA = (cytnx_complex64 *)v->data();
        checkCudaErrors(cudaMemcpy(v->data(), in->data(),
                                   sizeof(cytnx_complex64) * cytnx_uint64(L) * L,
                                   cudaMemcpyDeviceToDevice));
      } else {
        owned_tA = utils_internal::DeviceBuffer<cytnx_complex64>(cytnx_uint64(L) * L);
        tA = owned_tA.get();
        checkCudaErrors(cudaMemcpy(tA, in->data(), sizeof(cytnx_complex64) * cytnx_uint64(L) * L,
                                   cudaMemcpyDeviceToDevice));
      }

      // query buffer:
      cytnx_int32 lwork = 0;
      cytnx_int32 b32L = L;
      checkCudaErrors(cusolverDnCheevd_bufferSize(cusolverH.get(), jobz, CUBLAS_FILL_MODE_UPPER,
                                                  b32L, (cuFloatComplex *)tA, b32L,
                                                  (cytnx_float *)e->data(), &lwork));

      // allocate working space:
      utils_internal::DeviceBuffer<cytnx_complex64> work(lwork);

      // call :
      cytnx_int32 info;
      utils_internal::DeviceBuffer<cytnx_int32> devinfo(1);
      checkCudaErrors(cusolverDnCheevd(cusolverH.get(), jobz, CUBLAS_FILL_MODE_UPPER, b32L,
                                       (cuFloatComplex *)tA, b32L, (cytnx_float *)e->data(),
                                       (cuFloatComplex *)work.get(), lwork, devinfo.get()));

      // get info
      checkCudaErrors(
        cudaMemcpy(&info, devinfo.get(), sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in cuBlas function 'cusolverDnZheevd': cuBlas INFO = ", info);
    }
    void cuEigh_internal_d(const boost::intrusive_ptr<Storage_base> &in,
                           boost::intrusive_ptr<Storage_base> &e,
                           boost::intrusive_ptr<Storage_base> &v, const cytnx_int64 &L) {
      cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
      if (v->dtype() == Type.Void) jobz = CUSOLVER_EIG_MODE_NOVECTOR;

      // create handles:
      utils_internal::CusolverDnHandle cusolverH;

      cytnx_double *tA;
      utils_internal::DeviceBuffer<cytnx_double> owned_tA;
      if (v->dtype() != Type.Void) {
        tA = (cytnx_double *)v->data();
        checkCudaErrors(cudaMemcpy(v->data(), in->data(),
                                   sizeof(cytnx_double) * cytnx_uint64(L) * L,
                                   cudaMemcpyDeviceToDevice));
      } else {
        owned_tA = utils_internal::DeviceBuffer<cytnx_double>(cytnx_uint64(L) * L);
        tA = owned_tA.get();
        checkCudaErrors(cudaMemcpy(tA, in->data(), sizeof(cytnx_double) * cytnx_uint64(L) * L,
                                   cudaMemcpyDeviceToDevice));
      }

      // query buffer:
      cytnx_int32 lwork = 0;
      cytnx_int32 b32L = L;
      checkCudaErrors(cusolverDnDsyevd_bufferSize(cusolverH.get(), jobz, CUBLAS_FILL_MODE_UPPER,
                                                  b32L, tA, b32L, (cytnx_double *)e->data(),
                                                  &lwork));

      // allocate working space:
      utils_internal::DeviceBuffer<cytnx_double> work(lwork);

      // call :
      cytnx_int32 info;
      utils_internal::DeviceBuffer<cytnx_int32> devinfo(1);
      checkCudaErrors(cusolverDnDsyevd(cusolverH.get(), jobz, CUBLAS_FILL_MODE_UPPER, b32L, tA,
                                       b32L, (cytnx_double *)e->data(), work.get(), lwork,
                                       devinfo.get()));

      // get info
      checkCudaErrors(
        cudaMemcpy(&info, devinfo.get(), sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in cuBlas function 'cusolverDnDsysevd': cuBlas INFO = ", info);
    }
    void cuEigh_internal_f(const boost::intrusive_ptr<Storage_base> &in,
                           boost::intrusive_ptr<Storage_base> &e,
                           boost::intrusive_ptr<Storage_base> &v, const cytnx_int64 &L) {
      cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
      if (v->dtype() == Type.Void) jobz = CUSOLVER_EIG_MODE_NOVECTOR;

      // create handles:
      utils_internal::CusolverDnHandle cusolverH;

      cytnx_float *tA;
      utils_internal::DeviceBuffer<cytnx_float> owned_tA;
      if (v->dtype() != Type.Void) {
        tA = (cytnx_float *)v->data();
        checkCudaErrors(cudaMemcpy(v->data(), in->data(), sizeof(cytnx_float) * cytnx_uint64(L) * L,
                                   cudaMemcpyDeviceToDevice));
      } else {
        owned_tA = utils_internal::DeviceBuffer<cytnx_float>(cytnx_uint64(L) * L);
        tA = owned_tA.get();
        checkCudaErrors(cudaMemcpy(tA, in->data(), sizeof(cytnx_float) * cytnx_uint64(L) * L,
                                   cudaMemcpyDeviceToDevice));
      }

      // query buffer:
      cytnx_int32 lwork = 0;
      cytnx_int32 b32L = L;
      checkCudaErrors(cusolverDnSsyevd_bufferSize(cusolverH.get(), jobz, CUBLAS_FILL_MODE_UPPER,
                                                  b32L, tA, b32L, (cytnx_float *)e->data(),
                                                  &lwork));

      // allocate working space:
      utils_internal::DeviceBuffer<cytnx_float> work(lwork);

      // call :
      cytnx_int32 info;
      utils_internal::DeviceBuffer<cytnx_int32> devinfo(1);
      checkCudaErrors(cusolverDnSsyevd(cusolverH.get(), jobz, CUBLAS_FILL_MODE_UPPER, b32L, tA,
                                       b32L, (cytnx_float *)e->data(), work.get(), lwork,
                                       devinfo.get()));

      // get info
      checkCudaErrors(
        cudaMemcpy(&info, devinfo.get(), sizeof(cytnx_int32), cudaMemcpyDeviceToHost));
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in cuBlas function 'cusolverDnDsysevd': cuBlas INFO = ", info);
    }

  }  // namespace linalg_internal
}  // namespace cytnx
