#include "cuQR_internal.hpp"

namespace cytnx {

  namespace linalg_internal {

// cusolver API error checking
#define checkCuSolverErrors(err)                                        \
  do {                                                                  \
    cusolverStatus_t err_ = (err);                                      \
    if (err_ != CUSOLVER_STATUS_SUCCESS) {                              \
      printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("cusolver error");                       \
    }                                                                   \
  } while (0)

// cublas API error checking
#define checkCuBlasErrors(err)                                        \
  do {                                                                \
    cublasStatus_t err_ = (err);                                      \
    if (err_ != CUBLAS_STATUS_SUCCESS) {                              \
      printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("cublas error");                       \
    }                                                                 \
  } while (0)

    template <class T>
    void GetLowerTri(T *out, const T *elem, const cytnx_uint64 &M, const cytnx_uint64 &N) {
      cytnx_uint64 min = M < N ? M : N;
      for (cytnx_uint64 i = 0; i < min; i++) {
        // cudaMemcpy(out + i * N + i, elem + i * N + i, (N - i) * sizeof(T),
        // cudaMemcpyDeviceToDevice);
        cudaMemcpy(out + i * N + i, elem + i * M + i, (i + 1) * sizeof(T),
                   cudaMemcpyDeviceToDevice);
      }
    }
    
    template <class T>
    void GetUpTri(T *out, const T *elem, const cytnx_uint64 &M, const cytnx_uint64 &N) {
      cytnx_uint64 min = M < N ? M : N;
      for (cytnx_uint64 i = 0; i < min; i++) {
        cudaMemcpy(out + i * N + i, elem + i * N + i, (N - i) * sizeof(T),
                   cudaMemcpyDeviceToDevice);
      }
    }

    // template <class T>
    // void GetDiag(T *out, const T *elem, const cytnx_uint64 &M, const cytnx_uint64 &N,
    //              const cytnx_uint64 &diag_N) {
    //   cytnx_uint64 min = M < N ? M : N;
    //   min = min < diag_N ? min : diag_N;

    //   for (cytnx_uint64 i = 0; i < min; i++) out[i] = elem[i * N + i];
    // }

    /// QR
    void cuQR_internal_cd(const boost::intrusive_ptr<Storage_base> &in,
                          boost::intrusive_ptr<Storage_base> &Q,
                          boost::intrusive_ptr<Storage_base> &R,
                          boost::intrusive_ptr<Storage_base> &D,
                          boost::intrusive_ptr<Storage_base> &tau, const cytnx_int64 &M,
                          const cytnx_int64 &N, const bool &is_d) {
      // Q should be the same shape as in
      // tau should be the min(M,N)

      using data_type = cytnx_complex128;
      using d_data_type = cuDoubleComplex;

      data_type *pQ = (data_type *)Q->Mem;
      data_type *pR = (data_type *)R->Mem;
      data_type *ptau = (data_type *)tau->Mem;
      data_type *pin = (data_type *)in->Mem;
      // create handles:

      int64_t ldA = N;
      // lapack_int info;
      int64_t K = M < N ? M : N;

      cusolverDnHandle_t cusolverH = NULL;
      checkCuSolverErrors(cusolverDnCreate(&cusolverH));

      cusolverDnParams_t params;
      checkCuSolverErrors(cusolverDnCreateParams(&params));

      // checkCudaErrors(cusolverDnDestroyParams(&params));

      //checkCudaErrors(cudaMemcpy(pQ, pin, M * N * sizeof(data_type), cudaMemcpyDeviceToDevice));

      // A = Q R
      // At = Rt Qt
      // size_t d_size;
      // size_t h_size;
      // size_t h_size_org;
      // void *bufferOnDevice;
      // void *bufferOnHost;

      int *d_info = nullptr;
      d_data_type *d_work = nullptr;

      int lwork_geqrf = 0;
      int lwork_orgqr = 0;
      int lwork = 0;
      int info = 0;

      // checkCuSolverErrors(cusolverDnZgeqrf_bufferSize(
      //   cusolverH, params, N, M, /*A*/ d_data_type, pQ, ldA, /*tau*/ d_data_type, ptau,
      //   /*cudaDataType*/ d_data_type, &d_size, &h_size));

      checkCuSolverErrors(
        cusolverDnZgeqrf_bufferSize(cusolverH, N, M, (d_data_type *)pQ, ldA, &lwork_geqrf));

      checkCuSolverErrors(cusolverDnZungqr_bufferSize(cusolverH, M >= N ? N : M, M, K, (d_data_type *)pQ, ldA,
                                                      (d_data_type *)ptau, &lwork_orgqr));
      lwork = std::max(lwork_geqrf, lwork_orgqr);

      // checkCudaErrors(cudaMalloc(bufferOnDevice, d_size));
      // checkCudaErrors(cudaMalloc(bufferOnHost, h_size));

      checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(data_type) * lwork));

      // checkCuSolverErrors(cusolverDnZgeqrf(cusolverH, params, N, M, /*A*/ d_data_type, pQ, ldA,
      //                                      /*tau*/ d_data_type, ptau, /*cudaDataType*/
      //                                      d_data_type, bufferOnDevice, d_size, bufferOnHost,
      //                                      h_size, &info));
      checkCuSolverErrors(cusolverDnZgeqrf(cusolverH, N, M, (d_data_type *)pQ, ldA,
                                           (d_data_type *)ptau, d_work, lwork, d_info));

      cytnx_error_msg(info != 0, "%s %d %s",
                      "Error in cuBlas function 'cusolverDnZgeqrf': cuBlas INFO = ", info,
                      "see cusolver manual for more info.");

      cublasHandle_t cublasH = NULL;
      checkCuBlasErrors(cublasCreate(&cublasH));
      
      d_data_type *pQt = nullptr;
      const cuDoubleComplex h_one = make_cuDoubleComplex(1,0);
      const cuDoubleComplex h_zero = make_cuDoubleComplex(0,0);

      checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&pQt), sizeof(d_data_type) * M * N));

      //C = aOP(A)+bOP(B)
      //C = aOP(A)+bC
      checkCuBlasErrors(cublasZgeam(cublasH,  CUBLAS_OP_T, CUBLAS_OP_N, /* A : N */M, /* A : M */N,
                                  &h_one,
                                  (d_data_type *)pQ, /*ldA*/N,
                                  &h_zero,
                                  (d_data_type *)pQt, /*ldB*/ M,
                                 (d_data_type *)pQt, /*ldC*/ M));
      //Alternatively:
      //  pQt = (d_data_type*) Q->Move_memory({(cytnx_uint64)M, (cytnx_uint64)N}, {1, 0}, {1, 0})->Mem;
      //  Q->Move_memory_({(cytnx_uint64)M, (cytnx_uint64)N}, {1, 0}, {1, 0})->Mem;

      // cudaStream_t stream = NULL;
      // checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      // checkCuBlasErrors(cublasSetStream(cublasH, stream));
      // checkCudaErrors(cudaStreamSynchronize(stream));

      // getR:
      GetUpTri(pR, (data_type*)pQt, M, N);

      //GetLowerTri(pR, (data_type*) pQt, M, N);

      checkCuSolverErrors(cusolverDnZungqr(cusolverH,  M >= N ? N : M,  M, K, (d_data_type *)pQ,  ldA,
                                           (d_data_type *)ptau, d_work, lwork, d_info));

      cytnx_error_msg(info != 0, "%s %d %s",
                      "Error in cuBlas function 'cusolverDnZorgqr': cuBlas INFO = ", info,
                      "see cusolver manual for more info.");
      
      //C = aOP(A)+bOP(B)
      //C = aOP(A)+bC
      checkCuBlasErrors(cublasZgeam(cublasH,  CUBLAS_OP_T, CUBLAS_OP_N, /* A : N */M, /* A : M */N,
                                  &h_one,
                                  (d_data_type *)pQ, /*ldA*/N,
                                  &h_zero,
                                  (d_data_type *)pQt,/*ldB*/ M,
                                 (d_data_type *)pQt,/*ldC*/ M));

      // check Q**T*Q:
      // cublasHandle_t cublasH = NULL;
      // checkCuBlasErrors(cublasCreate(&cublasH));
      // const cuDoubleComplex h_one = make_cuDoubleComplex(1,0);
      // const cuDoubleComplex h_minus_one = make_cuDoubleComplex(-1,0);
      // checkCuBlasErrors(cublasZgemm(cublasH,
      //                             CUBLAS_OP_T,  // Q**T
      //                             CUBLAS_OP_N,  // Q
      //                             M,            // number of rows of R
      //                             M,            // number of columns of R
      //                             N,            // number of columns of Q**T
      //                             &h_minus_one, /* host pointer */
      //                             (d_data_type*)pQ,          // Q**T
      //                             ldA,
      //                             (d_data_type*)pQ,         // Q
      //                             ldA, &h_one, /* host pointer */
      //                             (d_data_type*)pR, M));
    }

    void cuQR_internal_cf(const boost::intrusive_ptr<Storage_base> &in,
                          boost::intrusive_ptr<Storage_base> &Q,
                          boost::intrusive_ptr<Storage_base> &R,
                          boost::intrusive_ptr<Storage_base> &D,
                          boost::intrusive_ptr<Storage_base> &tau, const cytnx_int64 &M,
                          const cytnx_int64 &N, const bool &is_d) {}

    void cuQR_internal_d(const boost::intrusive_ptr<Storage_base> &in,
                          boost::intrusive_ptr<Storage_base> &Q,
                          boost::intrusive_ptr<Storage_base> &R,
                          boost::intrusive_ptr<Storage_base> &D,
                          boost::intrusive_ptr<Storage_base> &tau, const cytnx_int64 &M,
                          const cytnx_int64 &N, const bool &is_d) {
      // Q should be the same shape as in
      // tau should be the min(M,N)

      using data_type = cytnx_double;
      using d_data_type = double;

      data_type *pQ = (data_type *)Q->Mem;
      data_type *pR = (data_type *)R->Mem;
      data_type *ptau = (data_type *)tau->Mem;
      data_type *pin = (data_type *)in->Mem;
      // create handles:

      int64_t ldA = N;
      // lapack_int info;
      int64_t K = M < N ? M : N;

      cusolverDnHandle_t cusolverH = NULL;
      checkCuSolverErrors(cusolverDnCreate(&cusolverH));

      cusolverDnParams_t params;
      checkCuSolverErrors(cusolverDnCreateParams(&params));

      // checkCudaErrors(cusolverDnDestroyParams(&params));

      checkCudaErrors(cudaMemcpy(pQ, pin, M * N * sizeof(data_type), cudaMemcpyDeviceToDevice));

      // size_t d_size;
      // size_t h_size;
      // size_t h_size_org;
      // void *bufferOnDevice;
      // void *bufferOnHost;

      int *d_info = nullptr;
      d_data_type *d_work = nullptr;

      int lwork_geqrf = 0;
      int lwork_orgqr = 0;
      int lwork = 0;
      int info = 0;

      // checkCuSolverErrors(cusolverDnZgeqrf_bufferSize(
      //   cusolverH, params, N, M, /*A*/ d_data_type, pQ, ldA, /*tau*/ d_data_type, ptau,
      //   /*cudaDataType*/ d_data_type, &d_size, &h_size));

      checkCuSolverErrors(
        cusolverDnDgeqrf_bufferSize(cusolverH, N, M, (d_data_type *)pQ, ldA, &lwork_geqrf));

      checkCuSolverErrors(cusolverDnDorgqr_bufferSize(cusolverH, M >= N ? N : M, M, K, (d_data_type *)pQ, ldA,
                                                      (d_data_type *)ptau, &lwork_orgqr));
      lwork = std::max(lwork_geqrf, lwork_orgqr);

      // checkCudaErrors(cudaMalloc(bufferOnDevice, d_size));
      // checkCudaErrors(cudaMalloc(bufferOnHost, h_size));

      checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(data_type) * lwork));

      // checkCuSolverErrors(cusolverDnZgeqrf(cusolverH, params, N, M, /*A*/ d_data_type, pQ, ldA,
      //                                      /*tau*/ d_data_type, ptau, /*cudaDataType*/
      //                                      d_data_type, bufferOnDevice, d_size, bufferOnHost,
      //                                      h_size, &info));
      checkCuSolverErrors(cusolverDnDgeqrf(cusolverH, N, M, (d_data_type *)pQ, ldA,
                                           (d_data_type *)ptau, d_work, lwork, d_info));

      cytnx_error_msg(info != 0, "%s %d %s",
                      "Error in cuBlas function 'cusolverDnZgeqrf': cuBlas INFO = ", info,
                      "see cusolver manual for more info.");

      cublasHandle_t cublasH = NULL;
      checkCuBlasErrors(cublasCreate(&cublasH));
      
      d_data_type *pQt = nullptr;
      const d_data_type h_one = 1.;
      const d_data_type h_zero = 0.;

      checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&pQt), sizeof(d_data_type) * M * N));

      cudaStream_t stream = NULL;
      checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      checkCuBlasErrors(cublasSetStream(cublasH, stream));
      checkCudaErrors(cudaStreamSynchronize(stream));
      checkCuBlasErrors(cublasDgeam(cublasH,  CUBLAS_OP_T, CUBLAS_OP_N, N, M,
                                &h_one,
                                (d_data_type *)pQ, M,
                                &h_zero,
                                (d_data_type *)pQt, N,
                                (d_data_type *)pQt, N));
      checkCudaErrors(cudaStreamSynchronize(stream));
      // getR:
      GetUpTri(pR, (data_type*) pQt, M, N);
      //GetLowerTri(pR, (data_type*) pQt, M, N);

      checkCuSolverErrors(cusolverDnDorgqr(cusolverH, M >= N ? N : M, M, K, (d_data_type *)pQ, ldA,
                                           (d_data_type *)ptau, d_work, lwork, d_info));

      cytnx_error_msg(info != 0, "%s %d %s",
                      "Error in cuBlas function 'cusolverDnDorgqr': cuBlas INFO = ", info,
                      "see cusolver manual for more info.");
      
      // check Q**T*Q:
      // cublasHandle_t cublasH = NULL;
      // checkCuBlasErrors(cublasCreate(&cublasH));
      // const cuDoubleComplex h_one = make_cuDoubleComplex(1,0);
      // const cuDoubleComplex h_minus_one = make_cuDoubleComplex(-1,0);

      // checkCuBlasErrors(cublasZgemm(cublasH,
      //                             CUBLAS_OP_T,  // Q**T
      //                             CUBLAS_OP_N,  // Q
      //                             M,            // number of rows of R
      //                             M,            // number of columns of R
      //                             N,            // number of columns of Q**T
      //                             &h_minus_one, /* host pointer */
      //                             (d_data_type*)pQ,          // Q**T
      //                             ldA,
      //                             (d_data_type*)pQ,         // Q
      //                             ldA, &h_one, /* host pointer */
      //                             (d_data_type*)pR, M));
    }
    void cuQR_internal_f(const boost::intrusive_ptr<Storage_base> &in,
                         boost::intrusive_ptr<Storage_base> &Q,
                         boost::intrusive_ptr<Storage_base> &R,
                         boost::intrusive_ptr<Storage_base> &D,
                         boost::intrusive_ptr<Storage_base> &tau, const cytnx_int64 &M,
                         const cytnx_int64 &N, const bool &is_d) {}

    // void QR_internal_d(const boost::intrusive_ptr<Storage_base> &in,
    //                    boost::intrusive_ptr<Storage_base> &Q, boost::intrusive_ptr<Storage_base>
    //                    &R, boost::intrusive_ptr<Storage_base> &D,
    //                    boost::intrusive_ptr<Storage_base> &tau, const cytnx_int64 &M,
    //                    const cytnx_int64 &N, const bool &is_d) {
    //   // Q should be the same shape as in
    //   // tau should be the min(M,N)

    //   cytnx_double *pQ = (cytnx_double *)Q->Mem;
    //   cytnx_double *pR = (cytnx_double *)R->Mem;
    //   cytnx_double *ptau = (cytnx_double *)tau->Mem;

    //   memcpy(pQ, in->Mem, M * N * sizeof(cytnx_double));

    //   lapack_int ldA = N;
    //   lapack_int info;
    //   lapack_int K = M < N ? M : N;

    //   // call linalg:
    //   info = LAPACKE_dgelqf(LAPACK_COL_MAJOR, N, M, pQ, ldA, ptau);
    //   cytnx_error_msg(info != 0, "%s %d",
    //                   "Error in Lapack function 'dgelqf': Lapack INFO = ", info);

    //   // getR:
    //   GetUpTri(pR, pQ, M, N);

    //   // getD:
    //   if (is_d) {
    //     cytnx_double *pD = (cytnx_double *)D->Mem;
    //     GetDiag(pD, pR, M, N, N);
    //     cytnx_uint64 min = M < N ? M : N;
    //     // normalize:
    //     for (cytnx_uint64 i = 0; i < min; i++) {
    //       for (cytnx_uint64 j = 0; j < N - i; j++) {
    //         pR[i * N + i + j] /= pD[i];
    //       }
    //     }
    //   }

    //   // getQ:
    //   // query lwork & alloc
    //   lapack_int col = M;
    //   lapack_int row = M >= N ? N : M;

    //   info = LAPACKE_dorglq(LAPACK_COL_MAJOR, row, col, K, pQ, ldA, ptau);
    //   cytnx_error_msg(info != 0, "%s %d",
    //                   "Error in Lapack function 'dorglq': Lapack INFO = ", info);
    // }
    // void QR_internal_f(const boost::intrusive_ptr<Storage_base> &in,
    //                    boost::intrusive_ptr<Storage_base> &Q, boost::intrusive_ptr<Storage_base>
    //                    &R, boost::intrusive_ptr<Storage_base> &D,
    //                    boost::intrusive_ptr<Storage_base> &tau, const cytnx_int64 &M,
    //                    const cytnx_int64 &N, const bool &is_d) {
    //   // Q should be the same shape as in
    //   // tau should be the min(M,N)

    //   cytnx_float *pQ = (cytnx_float *)Q->Mem;
    //   cytnx_float *pR = (cytnx_float *)R->Mem;
    //   cytnx_float *ptau = (cytnx_float *)tau->Mem;

    //   // cytnx_complex128* Mij = (cytnx_complex128*)malloc(M * N * sizeof(cytnx_complex128));
    //   memcpy(pQ, in->Mem, M * N * sizeof(cytnx_float));

    //   lapack_int ldA = N;
    //   lapack_int info;
    //   lapack_int K = M < N ? M : N;

    //   // call linalg:
    //   info = LAPACKE_sgelqf(LAPACK_COL_MAJOR, N, M, pQ, ldA, ptau);
    //   cytnx_error_msg(info != 0, "%s %d",
    //                   "Error in Lapack function 'sgelqf': Lapack INFO = ", info);

    //   // getR:
    //   GetUpTri(pR, pQ, M, N);

    //   // getD:
    //   if (is_d) {
    //     cytnx_float *pD = (cytnx_float *)D->Mem;
    //     GetDiag(pD, pR, M, N, N);
    //     cytnx_uint64 min = M < N ? M : N;
    //     // normalize:
    //     for (cytnx_uint64 i = 0; i < min; i++) {
    //       for (cytnx_uint64 j = 0; j < N - i; j++) {
    //         pR[i * N + i + j] /= pD[i];
    //       }
    //     }
    //   }

    //   // getQ:
    //   // query lwork & alloc
    //   lapack_int col = M;
    //   lapack_int row = M >= N ? N : M;

    //   info = LAPACKE_sorglq(LAPACK_COL_MAJOR, row, col, K, pQ, ldA, ptau);
    //   cytnx_error_msg(info != 0, "%s %d",
    //                   "Error in Lapack function 'sorglq': Lapack INFO = ", info);
    // }

  }  // namespace linalg_internal
}  // namespace cytnx
