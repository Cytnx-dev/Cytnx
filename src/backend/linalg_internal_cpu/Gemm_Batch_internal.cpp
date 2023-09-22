#include "Gemm_Batch_internal.hpp"
#include "cytnx_error.hpp"

#ifdef UNI_OMP
  #include <omp.h>
#endif

namespace cytnx {
  namespace linalg_internal {

    void Gemm_Batch_internal_cd(const char *transa_array, const char *transb_array,
                                const blas_int *m_array, const blas_int *n_array,
                                const blas_int *k_array, const std::vector<Scalar> &alpha_array,
                                const void **a_array, const blas_int *lda_array,
                                const void **b_array, const blas_int *ldb_array,
                                const std::vector<Scalar> &beta_array, void **c_array,
                                const blas_int *ldc_array, const blas_int group_count,
                                const blas_int *group_size) {
      cytnx_complex128 alphas[alpha_array.size()];
      cytnx_complex128 betas[beta_array.size()];
      for (size_t i = 0; i < alpha_array.size(); i++) alphas[i] = complex128(alpha_array[i]);
      for (size_t i = 0; i < beta_array.size(); i++) betas[i] = complex128(beta_array[i]);
#ifdef UNI_MKL
      zgemm_batch(transa_array, transb_array, m_array, n_array, k_array, alphas,
                  (const cytnx_complex128 **)a_array, lda_array, (const cytnx_complex128 **)b_array,
                  ldb_array, betas, (cytnx_complex128 **)c_array, ldc_array, &group_count,
                  group_size);
#else
      cytnx_error_msg(true, "[Gemm_Batch_internal] fatal error, Gemm_Batch required MKL.%s", "\n");
#endif
    }
    void Gemm_Batch_internal_cf(const char *transa_array, const char *transb_array,
                                const blas_int *m_array, const blas_int *n_array,
                                const blas_int *k_array, const std::vector<Scalar> &alpha_array,
                                const void **a_array, const blas_int *lda_array,
                                const void **b_array, const blas_int *ldb_array,
                                const std::vector<Scalar> &beta_array, void **c_array,
                                const blas_int *ldc_array, const blas_int group_count,
                                const blas_int *group_size) {
      cytnx_complex64 alphas[alpha_array.size()];
      cytnx_complex64 betas[beta_array.size()];
      for (size_t i = 0; i < alpha_array.size(); i++) alphas[i] = complex64(alpha_array[i]);
      for (size_t i = 0; i < beta_array.size(); i++) betas[i] = complex64(beta_array[i]);
#ifdef UNI_MKL
      cgemm_batch(transa_array, transb_array, m_array, n_array, k_array, alphas,
                  (const cytnx_complex64 **)a_array, lda_array, (const cytnx_complex64 **)b_array,
                  ldb_array, betas, (cytnx_complex64 **)c_array, ldc_array, &group_count,
                  group_size);
#else
      cytnx_error_msg(true, "[Gemm_Batch_internal] fatal error, Gemm_Batch required MKL.%s", "\n");
#endif
    }
    void Gemm_Batch_internal_d(const char *transa_array, const char *transb_array,
                               const blas_int *m_array, const blas_int *n_array,
                               const blas_int *k_array, const std::vector<Scalar> &alpha_array,
                               const void **a_array, const blas_int *lda_array,
                               const void **b_array, const blas_int *ldb_array,
                               const std::vector<Scalar> &beta_array, void **c_array,
                               const blas_int *ldc_array, const blas_int group_count,
                               const blas_int *group_size) {
      cytnx_double alphas[alpha_array.size()];
      cytnx_double betas[beta_array.size()];
      for (size_t i = 0; i < alpha_array.size(); i++) alphas[i] = double(alpha_array[i]);
      for (size_t i = 0; i < beta_array.size(); i++) betas[i] = double(beta_array[i]);
#ifdef UNI_MKL
      dgemm_batch(transa_array, transb_array, m_array, n_array, k_array, alphas,
                  (const cytnx_double **)a_array, lda_array, (const cytnx_double **)b_array,
                  ldb_array, betas, (cytnx_double **)c_array, ldc_array, &group_count, group_size);
#else
      cytnx_error_msg(true, "[Gemm_Batch_internal] fatal error, Gemm_Batch required MKL.%s", "\n");
#endif
    }
    void Gemm_Batch_internal_f(const char *transa_array, const char *transb_array,
                               const blas_int *m_array, const blas_int *n_array,
                               const blas_int *k_array, const std::vector<Scalar> &alpha_array,
                               const void **a_array, const blas_int *lda_array,
                               const void **b_array, const blas_int *ldb_array,
                               const std::vector<Scalar> &beta_array, void **c_array,
                               const blas_int *ldc_array, const blas_int group_count,
                               const blas_int *group_size) {
      cytnx_float alphas[alpha_array.size()];
      cytnx_float betas[beta_array.size()];
      for (size_t i = 0; i < alpha_array.size(); i++) alphas[i] = float(alpha_array[i]);
      for (size_t i = 0; i < beta_array.size(); i++) betas[i] = float(beta_array[i]);
#ifdef UNI_MKL
      sgemm_batch(transa_array, transb_array, m_array, n_array, k_array, alphas,
                  (const cytnx_float **)a_array, lda_array, (const cytnx_float **)b_array,
                  ldb_array, betas, (cytnx_float **)c_array, ldc_array, &group_count, group_size);
#else
      cytnx_error_msg(true, "[Gemm_Batch_internal] fatal error, Gemm_Batch required MKL.%s", "\n");
#endif
    }

  }  // namespace linalg_internal

};  // namespace cytnx
