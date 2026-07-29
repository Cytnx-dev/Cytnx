#include "cuInvM_inplace_internal.hpp"
#include "cytnx_error.hpp"
#include "Type.hpp"
#include "backend/lapack_wrapper.hpp"
#include "backend/utils_internal_gpu/cuScopedResource_gpu.hpp"

#include <vector>

namespace cytnx {
  namespace linalg_internal {

    void cuInvM_inplace_internal_d(boost::intrusive_ptr<Storage_base> &ten, const cytnx_int64 &L) {
      // Scoped resources (#1146): the info checks below throw, so ownership -- not a cleanup block
      // at the tail of the function -- is what keeps these from leaking.
      utils_internal::CusolverDnHandle cusolverH;

      cytnx_int32 info;
      cytnx_int32 lwork = 0;
      utils_internal::DeviceBuffer<cytnx_int32> ipiv(L + 1);
      utils_internal::DeviceBuffer<cytnx_int32> devinfo(1);
      // trf:
      checkCudaErrors(
        cusolverDnDgetrf_bufferSize(cusolverH.get(), L, L, (cytnx_double *)ten->data(), L, &lwork));
      utils_internal::DeviceBuffer<cytnx_double> d_work(lwork);

      checkCudaErrors(cusolverDnDgetrf(cusolverH.get(), L, L, (cytnx_double *)ten->data(), L,
                                       d_work.get(), ipiv.get(), devinfo.get()));
      checkCudaErrors(
        cudaMemcpy(&info, devinfo.get(), sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(info != 0, "%s %d",
                      "ERROR in cuSolver function 'cusolverDnDgetrf': cuBlas INFO = ", info);

      // trs AX = B with B = I
      utils_internal::DeviceBuffer<cytnx_double> d_I(L * L);
      std::vector<cytnx_double> h_I(L * L, 0);
      for (cytnx_int64 i = 0; i < L; i++) h_I[i * L + i] = 1;

      checkCudaErrors(
        cudaMemcpy(d_I.get(), h_I.data(), sizeof(cytnx_double) * L * L, cudaMemcpyHostToDevice));

      checkCudaErrors(cusolverDnDgetrs(cusolverH.get(), CUBLAS_OP_N, L, L,
                                       (cytnx_double *)ten->data(), L, ipiv.get(), d_I.get(), L,
                                       devinfo.get()));
      checkCudaErrors(
        cudaMemcpy(&info, devinfo.get(), sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(info != 0, "%s %d",
                      "ERROR in cuSolver function 'cusolverDnDgetrs': cuBlas INFO = ", info);

      checkCudaErrors(
        cudaMemcpy(ten->data(), d_I.get(), sizeof(cytnx_double) * L * L, cudaMemcpyDeviceToDevice));
    }

    void cuInvM_inplace_internal_f(boost::intrusive_ptr<Storage_base> &ten, const cytnx_int64 &L) {
      utils_internal::CusolverDnHandle cusolverH;

      cytnx_int32 info;
      cytnx_int32 lwork = 0;
      utils_internal::DeviceBuffer<cytnx_int32> ipiv(L + 1);
      utils_internal::DeviceBuffer<cytnx_int32> devinfo(1);
      // trf:
      checkCudaErrors(
        cusolverDnSgetrf_bufferSize(cusolverH.get(), L, L, (cytnx_float *)ten->data(), L, &lwork));
      utils_internal::DeviceBuffer<cytnx_float> d_work(lwork);

      checkCudaErrors(cusolverDnSgetrf(cusolverH.get(), L, L, (cytnx_float *)ten->data(), L,
                                       d_work.get(), ipiv.get(), devinfo.get()));
      checkCudaErrors(
        cudaMemcpy(&info, devinfo.get(), sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(info != 0, "%s %d",
                      "ERROR in cuSolver function 'cusolverDnSgetrf': cuBlas INFO = ", info);

      // trs AX = B with B = I
      utils_internal::DeviceBuffer<cytnx_float> d_I(L * L);
      std::vector<cytnx_float> h_I(L * L, 0);
      for (cytnx_int64 i = 0; i < L; i++) h_I[i * L + i] = 1;

      checkCudaErrors(
        cudaMemcpy(d_I.get(), h_I.data(), sizeof(cytnx_float) * L * L, cudaMemcpyHostToDevice));

      checkCudaErrors(cusolverDnSgetrs(cusolverH.get(), CUBLAS_OP_N, L, L,
                                       (cytnx_float *)ten->data(), L, ipiv.get(), d_I.get(), L,
                                       devinfo.get()));
      checkCudaErrors(
        cudaMemcpy(&info, devinfo.get(), sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(info != 0, "%s %d",
                      "ERROR in cuSolver function 'cusolverDnSgetrs': cuBlas INFO = ", info);

      checkCudaErrors(
        cudaMemcpy(ten->data(), d_I.get(), sizeof(cytnx_float) * L * L, cudaMemcpyDeviceToDevice));
    }

    void cuInvM_inplace_internal_cd(boost::intrusive_ptr<Storage_base> &ten, const cytnx_int64 &L) {
      utils_internal::CusolverDnHandle cusolverH;

      cytnx_int32 info;
      cytnx_int32 lwork = 0;
      utils_internal::DeviceBuffer<cytnx_int32> ipiv(L + 1);
      utils_internal::DeviceBuffer<cytnx_int32> devinfo(1);
      // trf:
      checkCudaErrors(cusolverDnZgetrf_bufferSize(cusolverH.get(), L, L,
                                                  (cuDoubleComplex *)ten->data(), L, &lwork));
      utils_internal::DeviceBuffer<cytnx_complex128> d_work(lwork);

      checkCudaErrors(cusolverDnZgetrf(cusolverH.get(), L, L, (cuDoubleComplex *)ten->data(), L,
                                       (cuDoubleComplex *)d_work.get(), ipiv.get(), devinfo.get()));
      checkCudaErrors(
        cudaMemcpy(&info, devinfo.get(), sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(info != 0, "%s %d",
                      "ERROR in cuSolver function 'cusolverDnZgetrf': cuBlas INFO = ", info);

      // trs AX = B with B = I
      utils_internal::DeviceBuffer<cytnx_complex128> d_I(L * L);
      std::vector<cytnx_complex128> h_I(L * L, 0);
      for (cytnx_int64 i = 0; i < L; i++) h_I[i * L + i] = cytnx_complex128(1, 0);

      checkCudaErrors(cudaMemcpy(d_I.get(), h_I.data(), sizeof(cytnx_complex128) * L * L,
                                 cudaMemcpyHostToDevice));

      checkCudaErrors(cusolverDnZgetrs(cusolverH.get(), CUBLAS_OP_N, L, L,
                                       (cuDoubleComplex *)ten->data(), L, ipiv.get(),
                                       (cuDoubleComplex *)d_I.get(), L, devinfo.get()));
      checkCudaErrors(
        cudaMemcpy(&info, devinfo.get(), sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(info != 0, "%s %d",
                      "ERROR in cuSolver function 'cusolverDnZgetrs': cuBlas INFO = ", info);

      checkCudaErrors(cudaMemcpy(ten->data(), d_I.get(), sizeof(cytnx_complex128) * L * L,
                                 cudaMemcpyDeviceToDevice));
    }

    void cuInvM_inplace_internal_cf(boost::intrusive_ptr<Storage_base> &ten, const cytnx_int64 &L) {
      utils_internal::CusolverDnHandle cusolverH;

      cytnx_int32 info;
      cytnx_int32 lwork = 0;
      utils_internal::DeviceBuffer<cytnx_int32> ipiv(L + 1);
      utils_internal::DeviceBuffer<cytnx_int32> devinfo(1);
      // trf:
      checkCudaErrors(cusolverDnCgetrf_bufferSize(cusolverH.get(), L, L,
                                                  (cuFloatComplex *)ten->data(), L, &lwork));
      utils_internal::DeviceBuffer<cytnx_complex64> d_work(lwork);

      checkCudaErrors(cusolverDnCgetrf(cusolverH.get(), L, L, (cuFloatComplex *)ten->data(), L,
                                       (cuFloatComplex *)d_work.get(), ipiv.get(), devinfo.get()));
      checkCudaErrors(
        cudaMemcpy(&info, devinfo.get(), sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(info != 0, "%s %d",
                      "ERROR in cuSolver function 'cusolverDnCgetrf': cuBlas INFO = ", info);

      // trs AX = B with B = I
      utils_internal::DeviceBuffer<cytnx_complex64> d_I(L * L);
      std::vector<cytnx_complex64> h_I(L * L, 0);
      for (cytnx_int64 i = 0; i < L; i++) h_I[i * L + i] = cytnx_complex64(1, 0);

      checkCudaErrors(
        cudaMemcpy(d_I.get(), h_I.data(), sizeof(cytnx_complex64) * L * L, cudaMemcpyHostToDevice));

      checkCudaErrors(cusolverDnCgetrs(cusolverH.get(), CUBLAS_OP_N, L, L,
                                       (cuFloatComplex *)ten->data(), L, ipiv.get(),
                                       (cuFloatComplex *)d_I.get(), L, devinfo.get()));
      checkCudaErrors(
        cudaMemcpy(&info, devinfo.get(), sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(info != 0, "%s %d",
                      "ERROR in cuSolver function 'cusolverDnCgetrs': cuBlas INFO = ", info);

      checkCudaErrors(cudaMemcpy(ten->data(), d_I.get(), sizeof(cytnx_complex64) * L * L,
                                 cudaMemcpyDeviceToDevice));
    }

  }  // namespace linalg_internal

}  // namespace cytnx
