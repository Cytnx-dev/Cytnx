#include "cuSvd_internal.hpp"
#include "backend/linalg_internal_interface.hpp"
#include "backend/utils_internal_gpu/cuScopedResource_gpu.hpp"

#include <vector>

namespace cytnx {

  namespace linalg_internal {

    /// cuSvd
    void cuSvd_internal_cd(const boost::intrusive_ptr<Storage_base> &in,
                           boost::intrusive_ptr<Storage_base> &U,
                           boost::intrusive_ptr<Storage_base> &vT,
                           boost::intrusive_ptr<Storage_base> &S, const cytnx_int64 &M,
                           const cytnx_int64 &N) {
      using data_type = cytnx_complex128;
      cudaDataType cuda_data_type = CUDA_C_64F;
      cudaDataType cuda_data_typeR = CUDA_R_64F;
      assert(sizeof(cuDoubleComplex) == sizeof(cytnx_complex128));

      cusolverEigMode_t jobz;
      // if U and vT are void, then it will not be computed.
      jobz = (U->dtype() == Type.Void and vT->dtype() == Type.Void) ? CUSOLVER_EIG_MODE_NOVECTOR
                                                                    : CUSOLVER_EIG_MODE_VECTOR;

      // const int econ = 0; /* i.e. 'A' in gesvd  */
      cytnx_int32 econ = 1; /* i.e. 'S' in gesvd  */

      // create handles:
      // Scoped resources (#1146): the info check below throws, so ownership -- not the cleanup
      // block that used to sit at the tail of this function -- is what prevents a leak.
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

      std::size_t d_lwork = 0; /* size of workspace */
      void *d_work = nullptr; /* device workspace for getrf */
      std::size_t h_lwork = 0; /* size of workspace */
      void *h_work = nullptr; /* host workspace for getrf */
      cytnx_double h_err_sigma;
      // query working space :
      checkCudaErrors(cusolverDnXgesvdp_bufferSize(cusolverH.get(), nullptr, /* params */
                                                   jobz, econ, N, M, cuda_data_type, /* dataTypeA */
                                                   Mij.get(), ldA, cuda_data_typeR, /* dataTypeS */
                                                   S->data(), cuda_data_type, /* dataTypeU */
                                                   vTMem, ldu, /* ldu */
                                                   cuda_data_type, /* dataTypeV */
                                                   UMem, ldvT, /* ldv */
                                                   cuda_data_type, /* computeType */
                                                   &d_lwork, &h_lwork));

      // allocate working space:
      utils_internal::DeviceBuffer<data_type> owned_d_work(d_lwork);
      d_work = owned_d_work.get();
      std::vector<char> owned_h_work;
      if (0 < h_lwork) {
        owned_h_work.resize(h_lwork);
        h_work = owned_h_work.data();
        if (d_work == nullptr) {
          throw std::runtime_error("Error: d_work not allocated.");
        }
      }

      utils_internal::DeviceBuffer<cytnx_int32> devinfo(1);
      checkCudaErrors(cudaMemset(devinfo.get(), 0, sizeof(cytnx_int32)));

      cytnx_int32 info;
      /// compute:
      cusolverDnXgesvdp(cusolverH.get(), nullptr, /* params */
                        jobz, econ, N, M, cuda_data_type, /* dataTypeA */
                        Mij.get(), ldA, cuda_data_typeR, /* dataTypeS */
                        S->data(), cuda_data_type, /* dataTypeU */
                        vTMem, ldu, /* ldu */
                        cuda_data_type, /* dataTypeV */
                        UMem, ldvT, /* ldv */
                        cuda_data_type, /* computeType */
                        d_work, d_lwork, h_work, h_lwork, devinfo.get(), &h_err_sigma);
      if (jobz == CUSOLVER_EIG_MODE_VECTOR) {
        U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});
        linalg_internal::cuConj_inplace_internal_cd(U, M * min);
      }
      // get info
      checkCudaErrors(
        cudaMemcpy(&info, devinfo.get(), sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_warning_msg(
        h_err_sigma > 1e-12,
        "Warning: Singular values approach zero, SVD may not be accurate, err_sigma = %E\n",
        h_err_sigma);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in cuBlas function 'cusolverDnXgesvdp': cuBlas INFO = ", info);
    }
    void cuSvd_internal_cf(const boost::intrusive_ptr<Storage_base> &in,
                           boost::intrusive_ptr<Storage_base> &U,
                           boost::intrusive_ptr<Storage_base> &vT,
                           boost::intrusive_ptr<Storage_base> &S, const cytnx_int64 &M,
                           const cytnx_int64 &N) {
      using data_type = cytnx_complex64;
      cudaDataType cuda_data_type = CUDA_C_32F;
      cudaDataType cuda_data_typeR = CUDA_R_32F;
      assert(sizeof(cuFloatComplex) == sizeof(cytnx_complex64));

      cusolverEigMode_t jobz;
      // if U and vT are void, then it will not be computed.
      jobz = (U->dtype() == Type.Void and vT->dtype() == Type.Void) ? CUSOLVER_EIG_MODE_NOVECTOR
                                                                    : CUSOLVER_EIG_MODE_VECTOR;

      // const int econ = 0; /* i.e. 'A' in gesvd  */
      cytnx_int32 econ = 1; /* i.e. 'S' in gesvd  */

      // create handles:
      // Scoped resources (#1146): the info check below throws, so ownership -- not the cleanup
      // block that used to sit at the tail of this function -- is what prevents a leak.
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

      std::size_t d_lwork = 0; /* size of workspace */
      void *d_work = nullptr; /* device workspace for getrf */
      std::size_t h_lwork = 0; /* size of workspace */
      void *h_work = nullptr; /* host workspace for getrf */
      cytnx_double h_err_sigma;
      // query working space :
      checkCudaErrors(cusolverDnXgesvdp_bufferSize(cusolverH.get(), nullptr, /* params */
                                                   jobz, econ, N, M, cuda_data_type, /* dataTypeA */
                                                   Mij.get(), ldA, cuda_data_typeR, /* dataTypeS */
                                                   S->data(), cuda_data_type, /* dataTypeU */
                                                   vTMem, ldu, /* ldu */
                                                   cuda_data_type, /* dataTypeV */
                                                   UMem, ldvT, /* ldv */
                                                   cuda_data_type, /* computeType */
                                                   &d_lwork, &h_lwork));

      // allocate working space:
      utils_internal::DeviceBuffer<data_type> owned_d_work(d_lwork);
      d_work = owned_d_work.get();
      std::vector<char> owned_h_work;
      if (0 < h_lwork) {
        owned_h_work.resize(h_lwork);
        h_work = owned_h_work.data();
        if (d_work == nullptr) {
          throw std::runtime_error("Error: d_work not allocated.");
        }
      }

      utils_internal::DeviceBuffer<cytnx_int32> devinfo(1);
      checkCudaErrors(cudaMemset(devinfo.get(), 0, sizeof(cytnx_int32)));

      cytnx_int32 info;
      /// compute:
      cusolverDnXgesvdp(cusolverH.get(), nullptr, /* params */
                        jobz, econ, N, M, cuda_data_type, /* dataTypeA */
                        Mij.get(), ldA, cuda_data_typeR, /* dataTypeS */
                        S->data(), cuda_data_type, /* dataTypeU */
                        vTMem, ldu, /* ldu */
                        cuda_data_type, /* dataTypeV */
                        UMem, ldvT, /* ldv */
                        cuda_data_type, /* computeType */
                        d_work, d_lwork, h_work, h_lwork, devinfo.get(), &h_err_sigma);
      if (jobz == CUSOLVER_EIG_MODE_VECTOR) {
        U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});
        linalg_internal::cuConj_inplace_internal_cf(U, M * min);
      }
      // get info
      checkCudaErrors(
        cudaMemcpy(&info, devinfo.get(), sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_warning_msg(
        h_err_sigma > 1e-12,
        "Warning: Singular values approach zero, SVD may not be accurate, err_sigma = %E\n",
        h_err_sigma);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in cuBlas function 'cusolverDnXgesvdp': cuBlas INFO = ", info);
    }
    void cuSvd_internal_d(const boost::intrusive_ptr<Storage_base> &in,
                          boost::intrusive_ptr<Storage_base> &U,
                          boost::intrusive_ptr<Storage_base> &vT,
                          boost::intrusive_ptr<Storage_base> &S, const cytnx_int64 &M,
                          const cytnx_int64 &N) {
      using data_type = cytnx_double;
      cudaDataType cuda_data_type = CUDA_R_64F;
      cudaDataType cuda_data_typeR = CUDA_R_64F;

      cusolverEigMode_t jobz;
      // if U and vT are void, then it will not be computed.
      jobz = (U->dtype() == Type.Void and vT->dtype() == Type.Void) ? CUSOLVER_EIG_MODE_NOVECTOR
                                                                    : CUSOLVER_EIG_MODE_VECTOR;

      // const int econ = 0; /* i.e. 'A' in gesvd  */
      cytnx_int32 econ = 1; /* i.e. 'S' in gesvd  */

      // create handles:
      // Scoped resources (#1146): the info check below throws, so ownership -- not the cleanup
      // block that used to sit at the tail of this function -- is what prevents a leak.
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

      std::size_t d_lwork = 0; /* size of workspace */
      void *d_work = nullptr; /* device workspace for getrf */
      std::size_t h_lwork = 0; /* size of workspace */
      void *h_work = nullptr; /* host workspace for getrf */
      cytnx_double h_err_sigma;
      // query working space :
      checkCudaErrors(cusolverDnXgesvdp_bufferSize(cusolverH.get(), nullptr, /* params */
                                                   jobz, econ, N, M, cuda_data_type, /* dataTypeA */
                                                   Mij.get(), ldA, cuda_data_typeR, /* dataTypeS */
                                                   S->data(), cuda_data_type, /* dataTypeU */
                                                   vTMem, ldu, /* ldu */
                                                   cuda_data_type, /* dataTypeV */
                                                   UMem, ldvT, /* ldv */
                                                   cuda_data_type, /* computeType */
                                                   &d_lwork, &h_lwork));

      // allocate working space:
      utils_internal::DeviceBuffer<data_type> owned_d_work(d_lwork);
      d_work = owned_d_work.get();
      std::vector<char> owned_h_work;
      if (0 < h_lwork) {
        owned_h_work.resize(h_lwork);
        h_work = owned_h_work.data();
        if (d_work == nullptr) {
          throw std::runtime_error("Error: d_work not allocated.");
        }
      }

      utils_internal::DeviceBuffer<cytnx_int32> devinfo(1);
      checkCudaErrors(cudaMemset(devinfo.get(), 0, sizeof(cytnx_int32)));

      cytnx_int32 info;
      /// compute:
      cusolverDnXgesvdp(cusolverH.get(), nullptr, /* params */
                        jobz, econ, N, M, cuda_data_type, /* dataTypeA */
                        Mij.get(), ldA, cuda_data_typeR, /* dataTypeS */
                        S->data(), cuda_data_type, /* dataTypeU */
                        vTMem, ldu, /* ldu */
                        cuda_data_type, /* dataTypeV */
                        UMem, ldvT, /* ldv */
                        cuda_data_type, /* computeType */
                        d_work, d_lwork, h_work, h_lwork, devinfo.get(), &h_err_sigma);
      if (jobz == CUSOLVER_EIG_MODE_VECTOR)
        U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});

      // get info
      checkCudaErrors(
        cudaMemcpy(&info, devinfo.get(), sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_warning_msg(
        h_err_sigma > 1e-12,
        "Warning: Singular values approach zero, SVD may not be accurate, err_sigma = %E\n",
        h_err_sigma);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in cuBlas function 'cusolverDnXgesvdp': cuBlas INFO = ", info);
    }
    void cuSvd_internal_f(const boost::intrusive_ptr<Storage_base> &in,
                          boost::intrusive_ptr<Storage_base> &U,
                          boost::intrusive_ptr<Storage_base> &vT,
                          boost::intrusive_ptr<Storage_base> &S, const cytnx_int64 &M,
                          const cytnx_int64 &N) {
      using data_type = cytnx_float;
      cudaDataType cuda_data_type = CUDA_R_32F;
      cudaDataType cuda_data_typeR = CUDA_R_32F;

      cusolverEigMode_t jobz;
      // if U and vT are void, then it will not be computed.
      jobz = (U->dtype() == Type.Void and vT->dtype() == Type.Void) ? CUSOLVER_EIG_MODE_NOVECTOR
                                                                    : CUSOLVER_EIG_MODE_VECTOR;

      // const int econ = 0; /* i.e. 'A' in gesvd  */
      cytnx_int32 econ = 1; /* i.e. 'S' in gesvd  */

      // create handles:
      // Scoped resources (#1146): the info check below throws, so ownership -- not the cleanup
      // block that used to sit at the tail of this function -- is what prevents a leak.
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

      std::size_t d_lwork = 0; /* size of workspace */
      void *d_work = nullptr; /* device workspace for getrf */
      std::size_t h_lwork = 0; /* size of workspace */
      void *h_work = nullptr; /* host workspace for getrf */
      cytnx_double h_err_sigma;
      // query working space :
      checkCudaErrors(cusolverDnXgesvdp_bufferSize(cusolverH.get(), nullptr, /* params */
                                                   jobz, econ, N, M, cuda_data_type, /* dataTypeA */
                                                   Mij.get(), ldA, cuda_data_typeR, /* dataTypeS */
                                                   S->data(), cuda_data_type, /* dataTypeU */
                                                   vTMem, ldu, /* ldu */
                                                   cuda_data_type, /* dataTypeV */
                                                   UMem, ldvT, /* ldv */
                                                   cuda_data_type, /* computeType */
                                                   &d_lwork, &h_lwork));

      // allocate working space:
      utils_internal::DeviceBuffer<data_type> owned_d_work(d_lwork);
      d_work = owned_d_work.get();
      std::vector<char> owned_h_work;
      if (0 < h_lwork) {
        owned_h_work.resize(h_lwork);
        h_work = owned_h_work.data();
        if (d_work == nullptr) {
          throw std::runtime_error("Error: d_work not allocated.");
        }
      }

      utils_internal::DeviceBuffer<cytnx_int32> devinfo(1);
      checkCudaErrors(cudaMemset(devinfo.get(), 0, sizeof(cytnx_int32)));

      cytnx_int32 info;
      /// compute:
      cusolverDnXgesvdp(cusolverH.get(), nullptr, /* params */
                        jobz, econ, N, M, cuda_data_type, /* dataTypeA */
                        Mij.get(), ldA, cuda_data_typeR, /* dataTypeS */
                        S->data(), cuda_data_type, /* dataTypeU */
                        vTMem, ldu, /* ldu */
                        cuda_data_type, /* dataTypeV */
                        UMem, ldvT, /* ldv */
                        cuda_data_type, /* computeType */
                        d_work, d_lwork, h_work, h_lwork, devinfo.get(), &h_err_sigma);
      if (jobz == CUSOLVER_EIG_MODE_VECTOR)
        U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});

      // get info
      checkCudaErrors(
        cudaMemcpy(&info, devinfo.get(), sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_warning_msg(
        h_err_sigma > 1e-12,
        "Warning: Singular values approach zero, SVD may not be accurate, err_sigma = %E\n",
        h_err_sigma);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in cuBlas function 'cusolverDnXgesvdp': cuBlas INFO = ", info);
    }

  }  // namespace linalg_internal
}  // namespace cytnx
