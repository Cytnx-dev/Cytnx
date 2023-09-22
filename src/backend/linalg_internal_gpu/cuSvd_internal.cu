#include "cuSvd_internal.hpp"
#include "../linalg_internal_interface.hpp"

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

      size_t d_lwork = 0; /* size of workspace */
      void *d_work = nullptr; /* device workspace for getrf */
      size_t h_lwork = 0; /* size of workspace */
      void *h_work = nullptr; /* host workspace for getrf */
      cytnx_double h_err_sigma;
      // query working space :
      checkCudaErrors(cusolverDnXgesvdp_bufferSize(cusolverH, NULL, /* params */
                                                   jobz, econ, N, M, cuda_data_type, /* dataTypeA */
                                                   Mij, ldA, cuda_data_typeR, /* dataTypeS */
                                                   S->Mem, cuda_data_type, /* dataTypeU */
                                                   vTMem, ldu, /* ldu */
                                                   cuda_data_type, /* dataTypeV */
                                                   UMem, ldvT, /* ldv */
                                                   cuda_data_type, /* computeType */
                                                   &d_lwork, &h_lwork));

      // allocate working space:
      checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(data_type) * d_lwork));
      if (0 < h_lwork) {
        h_work = reinterpret_cast<void *>(malloc(h_lwork));
        if (d_work == nullptr) {
          throw std::runtime_error("Error: d_work not allocated.");
        }
      }

      cytnx_int32 *devinfo;
      checkCudaErrors(cudaMalloc((void **)&devinfo, sizeof(cytnx_int32)));
      checkCudaErrors(cudaMemset(devinfo, 0, sizeof(cytnx_int32)));

      cytnx_int32 info;
      /// compute:
      cusolverDnXgesvdp(cusolverH, NULL, /* params */
                        jobz, econ, N, M, cuda_data_type, /* dataTypeA */
                        Mij, ldA, cuda_data_typeR, /* dataTypeS */
                        S->Mem, cuda_data_type, /* dataTypeU */
                        vTMem, ldu, /* ldu */
                        cuda_data_type, /* dataTypeV */
                        UMem, ldvT, /* ldv */
                        cuda_data_type, /* computeType */
                        d_work, d_lwork, h_work, h_lwork, devinfo, &h_err_sigma);
      if (jobz == CUSOLVER_EIG_MODE_VECTOR) {
        U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});
        linalg_internal::cuConj_inplace_internal_cd(U, M * min);
      }
      // get info
      checkCudaErrors(cudaMemcpy(&info, devinfo, sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_warning_msg(
        h_err_sigma > 1e-12,
        "Warning: Singular values approach zero, SVD may not be accurate, err_sigma = %E\n",
        h_err_sigma);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in cuBlas function 'cusolverDnXgesvdp': cuBlas INFO = ", info);

      checkCudaErrors(cudaFree(Mij));
      if (UMem != nullptr and U->dtype == Type.Void) {
        checkCudaErrors(cudaFree(UMem));
      }
      if (vTMem != nullptr and vT->dtype == Type.Void) {
        checkCudaErrors(cudaFree(vTMem));
      }
      checkCudaErrors(cudaFree(devinfo));
      checkCudaErrors(cudaFree(d_work));
      free(h_work);
      checkCudaErrors(cusolverDnDestroy(cusolverH));
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

      size_t d_lwork = 0; /* size of workspace */
      void *d_work = nullptr; /* device workspace for getrf */
      size_t h_lwork = 0; /* size of workspace */
      void *h_work = nullptr; /* host workspace for getrf */
      cytnx_double h_err_sigma;
      // query working space :
      checkCudaErrors(cusolverDnXgesvdp_bufferSize(cusolverH, NULL, /* params */
                                                   jobz, econ, N, M, cuda_data_type, /* dataTypeA */
                                                   Mij, ldA, cuda_data_typeR, /* dataTypeS */
                                                   S->Mem, cuda_data_type, /* dataTypeU */
                                                   vTMem, ldu, /* ldu */
                                                   cuda_data_type, /* dataTypeV */
                                                   UMem, ldvT, /* ldv */
                                                   cuda_data_type, /* computeType */
                                                   &d_lwork, &h_lwork));

      // allocate working space:
      checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(data_type) * d_lwork));
      if (0 < h_lwork) {
        h_work = reinterpret_cast<void *>(malloc(h_lwork));
        if (d_work == nullptr) {
          throw std::runtime_error("Error: d_work not allocated.");
        }
      }

      cytnx_int32 *devinfo;
      checkCudaErrors(cudaMalloc((void **)&devinfo, sizeof(cytnx_int32)));
      checkCudaErrors(cudaMemset(devinfo, 0, sizeof(cytnx_int32)));

      cytnx_int32 info;
      /// compute:
      cusolverDnXgesvdp(cusolverH, NULL, /* params */
                        jobz, econ, N, M, cuda_data_type, /* dataTypeA */
                        Mij, ldA, cuda_data_typeR, /* dataTypeS */
                        S->Mem, cuda_data_type, /* dataTypeU */
                        vTMem, ldu, /* ldu */
                        cuda_data_type, /* dataTypeV */
                        UMem, ldvT, /* ldv */
                        cuda_data_type, /* computeType */
                        d_work, d_lwork, h_work, h_lwork, devinfo, &h_err_sigma);
      if (jobz == CUSOLVER_EIG_MODE_VECTOR) {
        U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});
        linalg_internal::cuConj_inplace_internal_cf(U, M * min);
      }
      // get info
      checkCudaErrors(cudaMemcpy(&info, devinfo, sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_warning_msg(
        h_err_sigma > 1e-12,
        "Warning: Singular values approach zero, SVD may not be accurate, err_sigma = %E\n",
        h_err_sigma);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in cuBlas function 'cusolverDnXgesvdp': cuBlas INFO = ", info);

      checkCudaErrors(cudaFree(Mij));
      if (UMem != nullptr and U->dtype == Type.Void) {
        checkCudaErrors(cudaFree(UMem));
      }
      if (vTMem != nullptr and vT->dtype == Type.Void) {
        checkCudaErrors(cudaFree(vTMem));
      }
      checkCudaErrors(cudaFree(devinfo));
      checkCudaErrors(cudaFree(d_work));
      free(h_work);
      checkCudaErrors(cusolverDnDestroy(cusolverH));
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

      size_t d_lwork = 0; /* size of workspace */
      void *d_work = nullptr; /* device workspace for getrf */
      size_t h_lwork = 0; /* size of workspace */
      void *h_work = nullptr; /* host workspace for getrf */
      cytnx_double h_err_sigma;
      // query working space :
      checkCudaErrors(cusolverDnXgesvdp_bufferSize(cusolverH, NULL, /* params */
                                                   jobz, econ, N, M, cuda_data_type, /* dataTypeA */
                                                   Mij, ldA, cuda_data_typeR, /* dataTypeS */
                                                   S->Mem, cuda_data_type, /* dataTypeU */
                                                   vTMem, ldu, /* ldu */
                                                   cuda_data_type, /* dataTypeV */
                                                   UMem, ldvT, /* ldv */
                                                   cuda_data_type, /* computeType */
                                                   &d_lwork, &h_lwork));

      // allocate working space:
      checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(data_type) * d_lwork));
      if (0 < h_lwork) {
        h_work = reinterpret_cast<void *>(malloc(h_lwork));
        if (d_work == nullptr) {
          throw std::runtime_error("Error: d_work not allocated.");
        }
      }

      cytnx_int32 *devinfo;
      checkCudaErrors(cudaMalloc((void **)&devinfo, sizeof(cytnx_int32)));
      checkCudaErrors(cudaMemset(devinfo, 0, sizeof(cytnx_int32)));

      cytnx_int32 info;
      /// compute:
      cusolverDnXgesvdp(cusolverH, NULL, /* params */
                        jobz, econ, N, M, cuda_data_type, /* dataTypeA */
                        Mij, ldA, cuda_data_typeR, /* dataTypeS */
                        S->Mem, cuda_data_type, /* dataTypeU */
                        vTMem, ldu, /* ldu */
                        cuda_data_type, /* dataTypeV */
                        UMem, ldvT, /* ldv */
                        cuda_data_type, /* computeType */
                        d_work, d_lwork, h_work, h_lwork, devinfo, &h_err_sigma);
      if (jobz == CUSOLVER_EIG_MODE_VECTOR)
        U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});

      // get info
      checkCudaErrors(cudaMemcpy(&info, devinfo, sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_warning_msg(
        h_err_sigma > 1e-12,
        "Warning: Singular values approach zero, SVD may not be accurate, err_sigma = %E\n",
        h_err_sigma);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in cuBlas function 'cusolverDnXgesvdp': cuBlas INFO = ", info);

      checkCudaErrors(cudaFree(Mij));
      if (UMem != nullptr and U->dtype == Type.Void) {
        checkCudaErrors(cudaFree(UMem));
      }
      if (vTMem != nullptr and vT->dtype == Type.Void) {
        checkCudaErrors(cudaFree(vTMem));
      }
      checkCudaErrors(cudaFree(devinfo));
      checkCudaErrors(cudaFree(d_work));
      free(h_work);
      checkCudaErrors(cusolverDnDestroy(cusolverH));
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

      size_t d_lwork = 0; /* size of workspace */
      void *d_work = nullptr; /* device workspace for getrf */
      size_t h_lwork = 0; /* size of workspace */
      void *h_work = nullptr; /* host workspace for getrf */
      cytnx_double h_err_sigma;
      // query working space :
      checkCudaErrors(cusolverDnXgesvdp_bufferSize(cusolverH, NULL, /* params */
                                                   jobz, econ, N, M, cuda_data_type, /* dataTypeA */
                                                   Mij, ldA, cuda_data_typeR, /* dataTypeS */
                                                   S->Mem, cuda_data_type, /* dataTypeU */
                                                   vTMem, ldu, /* ldu */
                                                   cuda_data_type, /* dataTypeV */
                                                   UMem, ldvT, /* ldv */
                                                   cuda_data_type, /* computeType */
                                                   &d_lwork, &h_lwork));

      // allocate working space:
      checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(data_type) * d_lwork));
      if (0 < h_lwork) {
        h_work = reinterpret_cast<void *>(malloc(h_lwork));
        if (d_work == nullptr) {
          throw std::runtime_error("Error: d_work not allocated.");
        }
      }

      cytnx_int32 *devinfo;
      checkCudaErrors(cudaMalloc((void **)&devinfo, sizeof(cytnx_int32)));
      checkCudaErrors(cudaMemset(devinfo, 0, sizeof(cytnx_int32)));

      cytnx_int32 info;
      /// compute:
      cusolverDnXgesvdp(cusolverH, NULL, /* params */
                        jobz, econ, N, M, cuda_data_type, /* dataTypeA */
                        Mij, ldA, cuda_data_typeR, /* dataTypeS */
                        S->Mem, cuda_data_type, /* dataTypeU */
                        vTMem, ldu, /* ldu */
                        cuda_data_type, /* dataTypeV */
                        UMem, ldvT, /* ldv */
                        cuda_data_type, /* computeType */
                        d_work, d_lwork, h_work, h_lwork, devinfo, &h_err_sigma);
      if (jobz == CUSOLVER_EIG_MODE_VECTOR)
        U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});

      // get info
      checkCudaErrors(cudaMemcpy(&info, devinfo, sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_warning_msg(
        h_err_sigma > 1e-12,
        "Warning: Singular values approach zero, SVD may not be accurate, err_sigma = %E\n",
        h_err_sigma);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in cuBlas function 'cusolverDnXgesvdp': cuBlas INFO = ", info);

      checkCudaErrors(cudaFree(Mij));
      if (UMem != nullptr and U->dtype == Type.Void) {
        checkCudaErrors(cudaFree(UMem));
      }
      if (vTMem != nullptr and vT->dtype == Type.Void) {
        checkCudaErrors(cudaFree(vTMem));
      }
      checkCudaErrors(cudaFree(devinfo));
      checkCudaErrors(cudaFree(d_work));
      free(h_work);
      checkCudaErrors(cusolverDnDestroy(cusolverH));
    }

  }  // namespace linalg_internal
}  // namespace cytnx
