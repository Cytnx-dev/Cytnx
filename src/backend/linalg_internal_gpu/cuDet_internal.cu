#include "cuDet_internal.hpp"
#include "cytnx_error.hpp"
#include "Type.hpp"
#include "backend/lapack_wrapper.hpp"

#include "backend/utils_internal_gpu/cuAlloc_gpu.hpp"
#include "backend/utils_internal_gpu/cuScopedResource_gpu.hpp"

#include <vector>

namespace cytnx {

  namespace linalg_internal {

    void cuDet_internal_cd(void* out, const boost::intrusive_ptr<Storage_base>& in,
                           const cytnx_uint64& L) {
      cytnx_complex128* od = (cytnx_complex128*)out;  // result on cpu!
      // Scoped resources (#1146): the info check below throws, so ownership -- not a cleanup block
      // at the tail of the function -- is what keeps these from leaking.
      // Managed (unified) memory: the diagonal is read from the host below.
      auto _in = utils_internal::DeviceBuffer<cuDoubleComplex>::managed(in->size());
      checkCudaErrors(cudaMemcpy(_in.get(), in->data(), sizeof(cytnx_complex128) * in->size(),
                                 cudaMemcpyDeviceToDevice));

      utils_internal::CusolverDnHandle cusolverH;

      utils_internal::DeviceBuffer<int> devIpiv(L);
      utils_internal::DeviceBuffer<int> devInfo(1);

      int workspace_size = 0;
      cusolverDnZgetrf_bufferSize(cusolverH.get(), L, L, _in.get(), L, &workspace_size);
      utils_internal::DeviceBuffer<cuDoubleComplex> workspace(workspace_size);

      cusolverDnZgetrf(cusolverH.get(), L, L, _in.get(), L, workspace.get(), devIpiv.get(),
                       devInfo.get());

      int info;
      checkCudaErrors(cudaMemcpy(&info, devInfo.get(), sizeof(int), cudaMemcpyDeviceToHost));
      cytnx_error_msg(info < 0, "[ERROR] cusolverDnZgetrf fail with info= %d\n", info);
      // TODO: info > 0 means U[info - 1, info - 1] is zero, which implies the determinant is
      // zero. The steps below can be skipped.

      // since we do unify mem, direct access element is possible:
      od[0] = 1;
      bool neg = 0;
      std::vector<int> ipiv(L);
      checkCudaErrors(
        cudaMemcpy(ipiv.data(), devIpiv.get(), L * sizeof(int), cudaMemcpyDeviceToHost));
      for (cytnx_uint64 i = 0; i < L; i++) {
        od[0] *= ((cytnx_complex128*)_in.get())[i * L + i];
        if (ipiv[i] != static_cast<int>(i + 1)) neg = !neg;
      }
      if (neg) od[0] *= -1;
    }

    void cuDet_internal_cf(void* out, const boost::intrusive_ptr<Storage_base>& in,
                           const cytnx_uint64& L) {
      cytnx_complex64* od = (cytnx_complex64*)out;  // result on cpu!
      // Managed (unified) memory: the diagonal is read from the host below.
      auto _in = utils_internal::DeviceBuffer<cuFloatComplex>::managed(in->size());
      checkCudaErrors(cudaMemcpy(_in.get(), in->data(), sizeof(cytnx_complex64) * in->size(),
                                 cudaMemcpyDeviceToDevice));

      utils_internal::CusolverDnHandle cusolverH;

      utils_internal::DeviceBuffer<int> devIpiv(L);
      utils_internal::DeviceBuffer<int> devInfo(1);

      int workspace_size = 0;
      cusolverDnCgetrf_bufferSize(cusolverH.get(), L, L, _in.get(), L, &workspace_size);
      utils_internal::DeviceBuffer<cuFloatComplex> workspace(workspace_size);

      cusolverDnCgetrf(cusolverH.get(), L, L, _in.get(), L, workspace.get(), devIpiv.get(),
                       devInfo.get());

      int info;
      checkCudaErrors(cudaMemcpy(&info, devInfo.get(), sizeof(int), cudaMemcpyDeviceToHost));
      cytnx_error_msg(info < 0, "[ERROR] cusolverDnCgetrf fail with info= %d\n", info);
      // TODO: info > 0 means U[info - 1, info - 1] is zero, which implies the determinant is
      // zero. The steps below can be skipped.

      // since we do unify mem, direct access element is possible:
      od[0] = 1;
      bool neg = 0;
      std::vector<int> ipiv(L);
      checkCudaErrors(
        cudaMemcpy(ipiv.data(), devIpiv.get(), L * sizeof(int), cudaMemcpyDeviceToHost));
      for (cytnx_uint64 i = 0; i < L; i++) {
        od[0] *= ((cytnx_complex64*)_in.get())[i * L + i];
        if (ipiv[i] != static_cast<int>(i + 1)) neg = !neg;
      }
      if (neg) od[0] *= -1;
    }

    void cuDet_internal_d(void* out, const boost::intrusive_ptr<Storage_base>& in,
                          const cytnx_uint64& L) {
      cytnx_double* od = (cytnx_double*)out;  // result on cpu!
      // Managed (unified) memory: the diagonal is read from the host below.
      auto _in = utils_internal::DeviceBuffer<cytnx_double>::managed(in->size());
      checkCudaErrors(cudaMemcpy(_in.get(), in->data(), sizeof(cytnx_double) * in->size(),
                                 cudaMemcpyDeviceToDevice));

      utils_internal::CusolverDnHandle cusolverH;

      utils_internal::DeviceBuffer<int> devIpiv(L);
      utils_internal::DeviceBuffer<int> devInfo(1);

      int workspace_size = 0;
      cusolverDnDgetrf_bufferSize(cusolverH.get(), L, L, _in.get(), L, &workspace_size);
      utils_internal::DeviceBuffer<cytnx_double> workspace(workspace_size);

      cusolverDnDgetrf(cusolverH.get(), L, L, _in.get(), L, workspace.get(), devIpiv.get(),
                       devInfo.get());

      int info;
      checkCudaErrors(cudaMemcpy(&info, devInfo.get(), sizeof(int), cudaMemcpyDeviceToHost));
      cytnx_error_msg(info < 0, "[ERROR] cusolverDnDgetrf fail with info= %d\n", info);
      // TODO: info > 0 means U[info - 1, info - 1] is zero, which implies the determinant is
      // zero. The steps below can be skipped.

      // since we do unify mem, direct access element is possible:
      od[0] = 1;
      bool neg = 0;
      std::vector<int> ipiv(L);
      checkCudaErrors(
        cudaMemcpy(ipiv.data(), devIpiv.get(), L * sizeof(int), cudaMemcpyDeviceToHost));
      for (cytnx_uint64 i = 0; i < L; i++) {
        od[0] *= _in.get()[i * L + i];
        if (ipiv[i] != static_cast<int>(i + 1)) neg = !neg;
      }
      if (neg) od[0] *= -1;
    }

    void cuDet_internal_f(void* out, const boost::intrusive_ptr<Storage_base>& in,
                          const cytnx_uint64& L) {
      cytnx_float* od = (cytnx_float*)out;  // result on cpu!
      // Managed (unified) memory: the diagonal is read from the host below.
      auto _in = utils_internal::DeviceBuffer<cytnx_float>::managed(in->size());
      checkCudaErrors(cudaMemcpy(_in.get(), in->data(), sizeof(cytnx_float) * in->size(),
                                 cudaMemcpyDeviceToDevice));

      utils_internal::CusolverDnHandle cusolverH;

      utils_internal::DeviceBuffer<int> devIpiv(L);
      utils_internal::DeviceBuffer<int> devInfo(1);

      int workspace_size = 0;
      cusolverDnSgetrf_bufferSize(cusolverH.get(), L, L, _in.get(), L, &workspace_size);
      utils_internal::DeviceBuffer<cytnx_float> workspace(workspace_size);

      cusolverDnSgetrf(cusolverH.get(), L, L, _in.get(), L, workspace.get(), devIpiv.get(),
                       devInfo.get());

      int info;
      checkCudaErrors(cudaMemcpy(&info, devInfo.get(), sizeof(int), cudaMemcpyDeviceToHost));
      cytnx_error_msg(info < 0, "[ERROR] cusolverDnSgetrf fail with info= %d\n", info);
      // TODO: info > 0 means U[info - 1, info - 1] is zero, which implies the determinant is
      // zero. The steps below can be skipped.

      // since we do unify mem, direct access element is possible:
      od[0] = 1;
      bool neg = 0;
      std::vector<int> ipiv(L);
      checkCudaErrors(
        cudaMemcpy(ipiv.data(), devIpiv.get(), L * sizeof(int), cudaMemcpyDeviceToHost));
      for (cytnx_uint64 i = 0; i < L; i++) {
        od[0] *= _in.get()[i * L + i];
        if (ipiv[i] != static_cast<int>(i + 1)) neg = !neg;
      }
      if (neg) od[0] *= -1;
    }

  }  // namespace linalg_internal
}  // namespace cytnx
