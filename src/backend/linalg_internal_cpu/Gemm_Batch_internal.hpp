#ifndef __Gemm_Batch_internal_H__
#define __Gemm_Batch_internal_H__

#include <assert.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include "backend/Storage.hpp"
#include "Type.hpp"
#include "backend/lapack_wrapper.hpp"

namespace cytnx {

  namespace linalg_internal {

    void Gemm_Batch_internal_cd(const char *transa_array, const char *transb_array,
                                const blas_int *m_array, const blas_int *n_array,
                                const blas_int *k_array, const std::vector<Scalar> &alpha_array,
                                const void **a_array, const blas_int *lda_array,
                                const void **b_array, const blas_int *ldb_array,
                                const std::vector<Scalar> &beta_array, void **c_array,
                                const blas_int *ldc_array, const blas_int group_count,
                                const blas_int *group_size);
    void Gemm_Batch_internal_cf(const char *transa_array, const char *transb_array,
                                const blas_int *m_array, const blas_int *n_array,
                                const blas_int *k_array, const std::vector<Scalar> &alpha_array,
                                const void **a_array, const blas_int *lda_array,
                                const void **b_array, const blas_int *ldb_array,
                                const std::vector<Scalar> &beta_array, void **c_array,
                                const blas_int *ldc_array, const blas_int group_count,
                                const blas_int *group_size);
    void Gemm_Batch_internal_d(const char *transa_array, const char *transb_array,
                               const blas_int *m_array, const blas_int *n_array,
                               const blas_int *k_array, const std::vector<Scalar> &alpha_array,
                               const void **a_array, const blas_int *lda_array,
                               const void **b_array, const blas_int *ldb_array,
                               const std::vector<Scalar> &beta_array, void **c_array,
                               const blas_int *ldc_array, const blas_int group_count,
                               const blas_int *group_size);
    void Gemm_Batch_internal_f(const char *transa_array, const char *transb_array,
                               const blas_int *m_array, const blas_int *n_array,
                               const blas_int *k_array, const std::vector<Scalar> &alpha_array,
                               const void **a_array, const blas_int *lda_array,
                               const void **b_array, const blas_int *ldb_array,
                               const std::vector<Scalar> &beta_array, void **c_array,
                               const blas_int *ldc_array, const blas_int group_count,
                               const blas_int *group_size);

  }  // namespace linalg_internal
}  // namespace cytnx

#endif
