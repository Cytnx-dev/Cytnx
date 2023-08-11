#include "linalg/linalg_internal_gpu/cuGeSvd_internal.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// cuGeSvd
    void cuGeSvd_internal_cd(const boost::intrusive_ptr<Storage_base> &in,
                             boost::intrusive_ptr<Storage_base> &U,
                             boost::intrusive_ptr<Storage_base> &vT,
                             boost::intrusive_ptr<Storage_base> &S, const cytnx_int64 &M,
                             const cytnx_int64 &N) {
      using data_type = cytnx_complex128;
      cudaDataType cuda_data_type = CUDA_C_64F;
      cudaDataType cuda_data_typeR = CUDA_R_64F;

      signed char jobu, jobv;

      // if U and vT are NULL ptr, then it will not be computed.
      jobu = (U->dtype == Type.Void) ? 'N' : 'S';
      jobv = (vT->dtype == Type.Void) ? 'N' : 'S';

      // create handles:
      cusolverDnHandle_t cusolverH = NULL;
      checkCudaErrors(cusolverDnCreate(&cusolverH));

      data_type *Mij;
      checkCudaErrors(cudaMalloc((void **)&Mij, M * N * sizeof(data_type)));
      checkCudaErrors(
        cudaMemcpy(Mij, in->Mem, sizeof(data_type) * M * N, cudaMemcpyDeviceToDevice));

      cytnx_int64 min = std::min(M, N);
      cytnx_int64 max = std::max(M, N);
      cytnx_int64 ldA = N, ldu = N, ldvT = min;
      if (N < M) {
        ldA = M, ldu = M, ldvT = min;
      }

      void *UMem, *vTMem;
      if (U->Mem) {
        UMem = U->Mem;
      } else {
        if (jobu == 'S') checkCudaErrors(cudaMalloc(&UMem, max * max * sizeof(data_type)));
      }
      if (vT->Mem) {
        vTMem = vT->Mem;
      } else {
        if (jobv == 'S') checkCudaErrors(cudaMalloc(&vTMem, max * max * sizeof(data_type)));
      }
      size_t d_lwork = 0;
      size_t h_lwork = 0;
      void *d_work = nullptr;
      void *h_work = nullptr;
      if (N >= M) {
        checkCudaErrors(cusolverDnXgesvd_bufferSize(
          cusolverH, NULL, jobv, jobu, N, M, cuda_data_type, Mij, ldA, cuda_data_typeR, S->Mem,
          cuda_data_type, vTMem, ldu, cuda_data_type, UMem, ldvT, cuda_data_type, &d_lwork,
          &h_lwork));
      } else {
        checkCudaErrors(cusolverDnXgesvd_bufferSize(
          cusolverH, NULL, jobu, jobv, M, N, cuda_data_type, Mij, ldA, cuda_data_typeR, S->Mem,
          cuda_data_type, UMem, ldu, cuda_data_type, vTMem, ldvT, cuda_data_type, &d_lwork,
          &h_lwork));
      }

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

      if (N >= M) {
        cusolverDnXgesvd(cusolverH, NULL, jobv, jobu, N, M, cuda_data_type, Mij, ldA,
                         cuda_data_typeR, S->Mem, cuda_data_type, vTMem, ldu, cuda_data_type, UMem,
                         ldvT, cuda_data_type, d_work, d_lwork, h_work, h_lwork, devinfo);
      } else {
        cusolverDnXgesvd(cusolverH, NULL, jobu, jobv, M, N, cuda_data_type, Mij, ldA,
                         cuda_data_typeR, S->Mem, cuda_data_type, UMem, ldu, cuda_data_type, vTMem,
                         ldvT, cuda_data_type, d_work, d_lwork, h_work, h_lwork, devinfo);
        if (U->dtype != Type.Void)
          U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});
        if (vT->dtype != Type.Void) {
          vT->Move_memory_({(cytnx_uint64)N, (cytnx_uint64)min}, {1, 0}, {1, 0});
          linalg_internal::cuConj_inplace_internal_cd(vT, N * min);
        }
      }

      cytnx_int32 info;
      // get info
      checkCudaErrors(cudaMemcpy(&info, devinfo, sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(
        info != 0, "%s %d %s", "Error in cuBlas function 'cusolverDnDgesvd': cuBlas INFO = ", info,
        "If info>0, possibly svd not converge, if info<0, see cusolver manual for more info.");

      checkCudaErrors(cudaFree(d_work));
      free(h_work);
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
      cudaDataType cuda_data_type = CUDA_C_32F;
      cudaDataType cuda_data_typeR = CUDA_R_32F;

      signed char jobu, jobv;

      // if U and vT are NULL ptr, then it will not be computed.
      jobu = (U->dtype == Type.Void) ? 'N' : 'S';
      jobv = (vT->dtype == Type.Void) ? 'N' : 'S';

      // create handles:
      cusolverDnHandle_t cusolverH = NULL;
      checkCudaErrors(cusolverDnCreate(&cusolverH));

      data_type *Mij;
      checkCudaErrors(cudaMalloc((void **)&Mij, M * N * sizeof(data_type)));
      checkCudaErrors(
        cudaMemcpy(Mij, in->Mem, sizeof(data_type) * M * N, cudaMemcpyDeviceToDevice));

      cytnx_int32 min = std::min(M, N);
      cytnx_int32 max = std::max(M, N);
      cytnx_int32 ldA = N, ldu = N, ldvT = min;
      if (N < M) {
        ldA = M, ldu = M, ldvT = min;
      }

      void *UMem, *vTMem;
      if (U->Mem) {
        UMem = U->Mem;
      } else {
        if (jobu == 'S') checkCudaErrors(cudaMalloc(&UMem, max * max * sizeof(data_type)));
      }
      if (vT->Mem) {
        vTMem = vT->Mem;
      } else {
        if (jobv == 'S') checkCudaErrors(cudaMalloc(&vTMem, max * max * sizeof(data_type)));
      }
      size_t d_lwork = 0;
      size_t h_lwork;
      void *d_work = nullptr;
      void *h_work = nullptr;
      if (N >= M) {
        checkCudaErrors(cusolverDnXgesvd_bufferSize(
          cusolverH, NULL, jobv, jobu, N, M, cuda_data_type, Mij, ldA, cuda_data_typeR, S->Mem,
          cuda_data_type, vTMem, ldu, cuda_data_type, UMem, ldvT, cuda_data_type, &d_lwork,
          &h_lwork));
      } else {
        checkCudaErrors(cusolverDnXgesvd_bufferSize(
          cusolverH, NULL, jobu, jobv, M, N, cuda_data_type, Mij, ldA, cuda_data_typeR, S->Mem,
          cuda_data_type, UMem, ldu, cuda_data_type, vTMem, ldvT, cuda_data_type, &d_lwork,
          &h_lwork));
      }

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

      if (N >= M) {
        cusolverDnXgesvd(cusolverH, NULL, jobv, jobu, N, M, cuda_data_type, Mij, ldA,
                         cuda_data_typeR, S->Mem, cuda_data_type, vTMem, ldu, cuda_data_type, UMem,
                         ldvT, cuda_data_type, d_work, d_lwork, h_work, h_lwork, devinfo);
      } else {
        cusolverDnXgesvd(cusolverH, NULL, jobu, jobv, M, N, cuda_data_type, Mij, ldA,
                         cuda_data_typeR, S->Mem, cuda_data_type, UMem, ldu, cuda_data_type, vTMem,
                         ldvT, cuda_data_type, d_work, d_lwork, h_work, h_lwork, devinfo);
        if (U->dtype != Type.Void)
          U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});
        if (vT->dtype != Type.Void) {
          vT->Move_memory_({(cytnx_uint64)N, (cytnx_uint64)min}, {1, 0}, {1, 0});
          linalg_internal::cuConj_inplace_internal_cf(vT, N * min);
        }
      }

      cytnx_int32 info;
      // get info
      checkCudaErrors(cudaMemcpy(&info, devinfo, sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(
        info != 0, "%s %d %s", "Error in cuBlas function 'cusolverDnDgesvd': cuBlas INFO = ", info,
        "If info>0, possibly svd not converge, if info<0, see cusolver manual for more info.");

      checkCudaErrors(cudaFree(d_work));
      free(h_work);
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
      cudaDataType cuda_data_type = CUDA_R_64F;
      cudaDataType cuda_data_typeR = CUDA_R_64F;

      signed char jobu, jobv;

      // if U and vT are NULL ptr, then it will not be computed.
      jobu = (U->dtype == Type.Void) ? 'N' : 'S';
      jobv = (vT->dtype == Type.Void) ? 'N' : 'S';

      // create handles:
      cusolverDnHandle_t cusolverH = NULL;
      checkCudaErrors(cusolverDnCreate(&cusolverH));

      data_type *Mij;
      checkCudaErrors(cudaMalloc((void **)&Mij, M * N * sizeof(data_type)));
      checkCudaErrors(
        cudaMemcpy(Mij, in->Mem, sizeof(data_type) * M * N, cudaMemcpyDeviceToDevice));

      cytnx_int64 min = std::min(M, N);
      cytnx_int64 max = std::max(M, N);
      cytnx_int64 ldA = N, ldu = N, ldvT = min;
      if (N < M) {
        ldA = M, ldu = M, ldvT = min;
      }

      void *UMem = nullptr, *vTMem = nullptr;
      if (U->Mem) {
        UMem = U->Mem;
      } else {
        if (jobu == 'S') checkCudaErrors(cudaMalloc(&UMem, max * max * sizeof(data_type)));
      }
      if (vT->Mem) {
        vTMem = vT->Mem;
      } else {
        if (jobv == 'S') checkCudaErrors(cudaMalloc(&vTMem, max * max * sizeof(data_type)));
      }
      size_t d_lwork = 0;
      size_t h_lwork;
      void *d_work = nullptr;
      void *h_work = nullptr;
      if (N >= M) {
        checkCudaErrors(cusolverDnXgesvd_bufferSize(
          cusolverH, NULL, jobv, jobu, N, M, cuda_data_type, Mij, ldA, cuda_data_typeR, S->Mem,
          cuda_data_type, vTMem, ldu, cuda_data_type, UMem, ldvT, cuda_data_type, &d_lwork,
          &h_lwork));
      } else {
        checkCudaErrors(cusolverDnXgesvd_bufferSize(
          cusolverH, NULL, jobu, jobv, M, N, cuda_data_type, Mij, ldA, cuda_data_typeR, S->Mem,
          cuda_data_type, UMem, ldu, cuda_data_type, vTMem, ldvT, cuda_data_type, &d_lwork,
          &h_lwork));
      }

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

      if (N >= M) {
        cusolverDnXgesvd(cusolverH, NULL, jobv, jobu, N, M, cuda_data_type, Mij, ldA,
                         cuda_data_typeR, S->Mem, cuda_data_type, vTMem, ldu, cuda_data_type, UMem,
                         ldvT, cuda_data_type, d_work, d_lwork, h_work, h_lwork, devinfo);
      } else {
        cusolverDnXgesvd(cusolverH, NULL, jobu, jobv, M, N, cuda_data_type, Mij, ldA,
                         cuda_data_typeR, S->Mem, cuda_data_type, UMem, ldu, cuda_data_type, vTMem,
                         ldvT, cuda_data_type, d_work, d_lwork, h_work, h_lwork, devinfo);
        if (U->dtype != Type.Void)
          U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});
        if (vT->dtype != Type.Void) {
          vT->Move_memory_({(cytnx_uint64)N, (cytnx_uint64)min}, {1, 0}, {1, 0});
        }
      }

      cytnx_int32 info;
      // get info
      checkCudaErrors(cudaMemcpy(&info, devinfo, sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(
        info != 0, "%s %d %s", "Error in cuBlas function 'cusolverDnDgesvd': cuBlas INFO = ", info,
        "If info>0, possibly svd not converge, if info<0, see cusolver manual for more info.");

      checkCudaErrors(cudaFree(d_work));
      free(h_work);
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
      cudaDataType cuda_data_type = CUDA_R_32F;
      cudaDataType cuda_data_typeR = CUDA_R_32F;

      signed char jobu, jobv;

      // if U and vT are NULL ptr, then it will not be computed.
      jobu = (U->dtype == Type.Void) ? 'N' : 'S';
      jobv = (vT->dtype == Type.Void) ? 'N' : 'S';

      // create handles:
      cusolverDnHandle_t cusolverH = NULL;
      checkCudaErrors(cusolverDnCreate(&cusolverH));

      data_type *Mij;
      checkCudaErrors(cudaMalloc((void **)&Mij, M * N * sizeof(data_type)));
      checkCudaErrors(
        cudaMemcpy(Mij, in->Mem, sizeof(data_type) * M * N, cudaMemcpyDeviceToDevice));

      cytnx_int32 min = std::min(M, N);
      cytnx_int32 max = std::max(M, N);
      cytnx_int32 ldA = N, ldu = N, ldvT = min;
      if (N < M) {
        ldA = M, ldu = M, ldvT = min;
      }

      void *UMem, *vTMem;
      if (U->Mem) {
        UMem = U->Mem;
      } else {
        if (jobu == 'S') checkCudaErrors(cudaMalloc(&UMem, max * max * sizeof(data_type)));
      }
      if (vT->Mem) {
        vTMem = vT->Mem;
      } else {
        if (jobv == 'S') checkCudaErrors(cudaMalloc(&vTMem, max * max * sizeof(data_type)));
      }
      size_t d_lwork = 0;
      size_t h_lwork;
      void *d_work = nullptr;
      void *h_work = nullptr;
      if (N >= M) {
        checkCudaErrors(cusolverDnXgesvd_bufferSize(
          cusolverH, NULL, jobv, jobu, N, M, cuda_data_type, Mij, ldA, cuda_data_typeR, S->Mem,
          cuda_data_type, vTMem, ldu, cuda_data_type, UMem, ldvT, cuda_data_type, &d_lwork,
          &h_lwork));
      } else {
        checkCudaErrors(cusolverDnXgesvd_bufferSize(
          cusolverH, NULL, jobu, jobv, M, N, cuda_data_type, Mij, ldA, cuda_data_typeR, S->Mem,
          cuda_data_type, UMem, ldu, cuda_data_type, vTMem, ldvT, cuda_data_type, &d_lwork,
          &h_lwork));
      }

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

      if (N >= M) {
        cusolverDnXgesvd(cusolverH, NULL, jobv, jobu, N, M, cuda_data_type, Mij, ldA,
                         cuda_data_typeR, S->Mem, cuda_data_type, vTMem, ldu, cuda_data_type, UMem,
                         ldvT, cuda_data_type, d_work, d_lwork, h_work, h_lwork, devinfo);
      } else {
        cusolverDnXgesvd(cusolverH, NULL, jobu, jobv, M, N, cuda_data_type, Mij, ldA,
                         cuda_data_typeR, S->Mem, cuda_data_type, UMem, ldu, cuda_data_type, vTMem,
                         ldvT, cuda_data_type, d_work, d_lwork, h_work, h_lwork, devinfo);
        if (U->dtype != Type.Void)
          U->Move_memory_({(cytnx_uint64)min, (cytnx_uint64)M}, {1, 0}, {1, 0});
        if (vT->dtype != Type.Void) {
          vT->Move_memory_({(cytnx_uint64)N, (cytnx_uint64)min}, {1, 0}, {1, 0});
        }
      }

      cytnx_int32 info;
      // get info
      checkCudaErrors(cudaMemcpy(&info, devinfo, sizeof(cytnx_int32), cudaMemcpyDeviceToHost));

      cytnx_error_msg(
        info != 0, "%s %d %s", "Error in cuBlas function 'cusolverDnDgesvd': cuBlas INFO = ", info,
        "If info>0, possibly svd not converge, if info<0, see cusolver manual for more info.");

      checkCudaErrors(cudaFree(d_work));
      free(h_work);
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
