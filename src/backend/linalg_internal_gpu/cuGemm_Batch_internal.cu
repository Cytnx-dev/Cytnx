#include "cuGemm_internal.hpp"
#include "cytnx_error.hpp"
#include "Type.hpp"
#include "backend/lapack_wrapper.hpp"

namespace cytnx {

  namespace linalg_internal {
    void cuGemm_Batch_internal_cd(const char *transa_array, const char *transb_array,
                                  const blas_int *m_array, const blas_int *n_array,
                                  const blas_int *k_array, const std::vector<Scalar> &alpha_array,
                                  const void **a_array, const blas_int *lda_array,
                                  const void **b_array, const blas_int *ldb_array,
                                  const std::vector<Scalar> &beta_array, void **c_array,
                                  const blas_int *ldc_array, const blas_int group_count,
                                  const blas_int *group_size) {
      std::vector<cytnx_complex128> alphas(alpha_array.size());
      std::vector<cytnx_complex128> betas(beta_array.size());
      for (size_t i = 0; i < alpha_array.size(); i++) alphas[i] = complex128(alpha_array[i]);
      for (size_t i = 0; i < beta_array.size(); i++) betas[i] = complex128(beta_array[i]);

      cytnx_uint64 idx = 0;
      for (cytnx_uint64 i = 0; i < group_count; i++) {
        for (cytnx_uint64 j = 0; j < group_size[i]; j++) {
          cublasOperation_t transa, transb;
          switch (transa_array[i]) {
            case 'N':
              transa = CUBLAS_OP_N;
              break;
            case 'T':
              transa = CUBLAS_OP_T;
              break;
            case 'C':
              transa = CUBLAS_OP_C;
              break;
            default:
              break;
          }
          switch (transb_array[i]) {
            case 'N':
              transb = CUBLAS_OP_N;
              break;
            case 'T':
              transb = CUBLAS_OP_T;
              break;
            case 'C':
              transb = CUBLAS_OP_C;
              break;
            default:
              break;
          }
          // create handles:
          cublasHandle_t cublasH = NULL;
          checkCudaErrors(cublasCreate(&cublasH));
          checkCudaErrors(cublasZgemm(
            cublasH, transa, transb, m_array[i], n_array[i], k_array[i],
            (cuDoubleComplex *)&alphas[i], (cuDoubleComplex *)a_array[idx], lda_array[i],
            (cuDoubleComplex *)b_array[idx], ldb_array[i], (cuDoubleComplex *)&betas[i],
            (cuDoubleComplex *)c_array[idx], ldc_array[i]));
          idx++;
          checkCudaErrors(cublasDestroy(cublasH));
        }
      }
    }

    void cuGemm_Batch_internal_cf(const char *transa_array, const char *transb_array,
                                  const blas_int *m_array, const blas_int *n_array,
                                  const blas_int *k_array, const std::vector<Scalar> &alpha_array,
                                  const void **a_array, const blas_int *lda_array,
                                  const void **b_array, const blas_int *ldb_array,
                                  const std::vector<Scalar> &beta_array, void **c_array,
                                  const blas_int *ldc_array, const blas_int group_count,
                                  const blas_int *group_size) {
      std::vector<cytnx_complex64> alphas(alpha_array.size());
      std::vector<cytnx_complex64> betas(beta_array.size());
      for (size_t i = 0; i < alpha_array.size(); i++) alphas[i] = complex64(alpha_array[i]);
      for (size_t i = 0; i < beta_array.size(); i++) betas[i] = complex64(beta_array[i]);

      cytnx_uint64 idx = 0;
      for (cytnx_uint64 i = 0; i < group_count; i++) {
        for (cytnx_uint64 j = 0; j < group_size[i]; j++) {
          cublasOperation_t transa, transb;
          switch (transa_array[i]) {
            case 'N':
              transa = CUBLAS_OP_N;
              break;
            case 'T':
              transa = CUBLAS_OP_T;
              break;
            case 'C':
              transa = CUBLAS_OP_C;
              break;
            default:
              break;
          }
          switch (transb_array[i]) {
            case 'N':
              transb = CUBLAS_OP_N;
              break;
            case 'T':
              transb = CUBLAS_OP_T;
              break;
            case 'C':
              transb = CUBLAS_OP_C;
              break;
            default:
              break;
          }
          // create handles:
          cublasHandle_t cublasH = NULL;
          checkCudaErrors(cublasCreate(&cublasH));
          checkCudaErrors(cublasCgemm(cublasH, transa, transb, m_array[i], n_array[i], k_array[i],
                                      (cuFloatComplex *)&alphas[i], (cuFloatComplex *)a_array[idx],
                                      lda_array[i], (cuFloatComplex *)b_array[idx], ldb_array[i],
                                      (cuFloatComplex *)&betas[i], (cuFloatComplex *)c_array[idx],
                                      ldc_array[i]));
          idx++;
          checkCudaErrors(cublasDestroy(cublasH));
        }
      }
    }

    void cuGemm_Batch_internal_d(const char *transa_array, const char *transb_array,
                                 const blas_int *m_array, const blas_int *n_array,
                                 const blas_int *k_array, const std::vector<Scalar> &alpha_array,
                                 const void **a_array, const blas_int *lda_array,
                                 const void **b_array, const blas_int *ldb_array,
                                 const std::vector<Scalar> &beta_array, void **c_array,
                                 const blas_int *ldc_array, const blas_int group_count,
                                 const blas_int *group_size) {
      std::vector<cytnx_double> alphas(alpha_array.size());
      std::vector<cytnx_double> betas(beta_array.size());
      for (size_t i = 0; i < alpha_array.size(); i++) alphas[i] = double(alpha_array[i]);
      for (size_t i = 0; i < beta_array.size(); i++) betas[i] = double(beta_array[i]);

      cytnx_uint64 idx = 0;
      for (cytnx_uint64 i = 0; i < group_count; i++) {
        for (cytnx_uint64 j = 0; j < group_size[i]; j++) {
          cublasOperation_t transa, transb;
          switch (transa_array[i]) {
            case 'N':
              transa = CUBLAS_OP_N;
              break;
            case 'T':
              transa = CUBLAS_OP_T;
              break;
            case 'C':
              transa = CUBLAS_OP_C;
              break;
            default:
              break;
          }
          switch (transb_array[i]) {
            case 'N':
              transb = CUBLAS_OP_N;
              break;
            case 'T':
              transb = CUBLAS_OP_T;
              break;
            case 'C':
              transb = CUBLAS_OP_C;
              break;
            default:
              break;
          }
          // create handles:
          cublasHandle_t cublasH = NULL;
          checkCudaErrors(cublasCreate(&cublasH));
          checkCudaErrors(cublasDgemm(
            cublasH, transa, transb, m_array[i], n_array[i], k_array[i], (cytnx_double *)&alphas[i],
            (cytnx_double *)a_array[idx], lda_array[i], (cytnx_double *)b_array[idx], ldb_array[i],
            (cytnx_double *)&betas[i], (cytnx_double *)c_array[idx], ldc_array[i]));
          idx++;
          checkCudaErrors(cublasDestroy(cublasH));
        }
      }
      // checkCudaErrors(cudaDeviceSynchronize());
    }
    void cuGemm_Batch_internal_f(const char *transa_array, const char *transb_array,
                                 const blas_int *m_array, const blas_int *n_array,
                                 const blas_int *k_array, const std::vector<Scalar> &alpha_array,
                                 const void **a_array, const blas_int *lda_array,
                                 const void **b_array, const blas_int *ldb_array,
                                 const std::vector<Scalar> &beta_array, void **c_array,
                                 const blas_int *ldc_array, const blas_int group_count,
                                 const blas_int *group_size) {
      std::vector<cytnx_float> alphas(alpha_array.size());
      std::vector<cytnx_float> betas(beta_array.size());
      for (size_t i = 0; i < alpha_array.size(); i++) alphas[i] = float(alpha_array[i]);
      for (size_t i = 0; i < beta_array.size(); i++) betas[i] = float(beta_array[i]);

      cytnx_uint64 idx = 0;
      for (cytnx_uint64 i = 0; i < group_count; i++) {
        for (cytnx_uint64 j = 0; j < group_size[i]; j++) {
          cublasOperation_t transa, transb;
          switch (transa_array[i]) {
            case 'N':
              transa = CUBLAS_OP_N;
              break;
            case 'T':
              transa = CUBLAS_OP_T;
              break;
            case 'C':
              transa = CUBLAS_OP_C;
              break;
            default:
              break;
          }
          switch (transb_array[i]) {
            case 'N':
              transb = CUBLAS_OP_N;
              break;
            case 'T':
              transb = CUBLAS_OP_T;
              break;
            case 'C':
              transb = CUBLAS_OP_C;
              break;
            default:
              break;
          }
          // create handles:
          cublasHandle_t cublasH = NULL;
          checkCudaErrors(cublasCreate(&cublasH));
          checkCudaErrors(cublasSgemm(
            cublasH, transa, transb, m_array[i], n_array[i], k_array[i], (cytnx_float *)&alphas[i],
            (cytnx_float *)a_array[idx], lda_array[i], (cytnx_float *)b_array[idx], ldb_array[i],
            (cytnx_float *)&betas[i], (cytnx_float *)c_array[idx], ldc_array[i]));
          idx++;
          checkCudaErrors(cublasDestroy(cublasH));
        }
      }
    }

  }  // namespace linalg_internal
}  // namespace cytnx
