#include "cuGemm_internal.hpp"
#include "cytnx_error.hpp"
#include "Type.hpp"
#include "lapack_wrapper.hpp"

namespace cytnx{

    namespace linalg_internal{
        void cuGemm_Batch_internal_cd(const char *transa_array, const char *transb_array, const blas_int *m_array, const blas_int *n_array, const blas_int *k_array,
                const std::vector<Scalar> &alpha_array, const void **a_array, const blas_int *lda_array, const void **b_array, const blas_int *ldb_array,
                const std::vector<Scalar> &beta_array, void **c_array, const blas_int *ldc_array, const blas_int group_count, const blas_int *group_size){
            cytnx_complex128 alphas[alpha_array.size()];
            cytnx_complex128 betas[beta_array.size()];
            for (size_t i=0; i<alpha_array.size(); i++) alphas[i] = complex128(alpha_array[i]);
            for (size_t i=0; i<beta_array.size(); i++) betas[i] = complex128(beta_array[i]);

            cytnx_uint64 idx = 0;
            for(cytnx_uint64 i=0;i<group_count;i++){
                for(cytnx_uint64 j=0;j<group_size[i];j++){
                    cytnx_int32 blsMl=m_array[i], blsNr=n_array[i], blsComm=k_array[i];
                    cublasOperation_t opl=CUBLAS_OP_N, opr=CUBLAS_OP_N;
                    switch (transa_array[i]) {
                        case 'N':
                            opl = CUBLAS_OP_N;
                            break;
                        case 'T':
                            opl = CUBLAS_OP_T;
                            break;
                        case 'C':
                            opl = CUBLAS_OP_C;
                            break;
                        default:
                            break;
                    }
                    switch (transb_array[i]) {
                        case 'N':
                            opr = CUBLAS_OP_N;
                            break;
                        case 'T':
                            opr = CUBLAS_OP_T;
                            break;
                        case 'C':
                            opr = CUBLAS_OP_C;
                            break;
                        default:
                            break;
                    }
                    // create handles:
                    cublasHandle_t cublasH = NULL;
                    checkCudaErrors(cublasCreate(&cublasH));
                    checkCudaErrors(cublasZgemm(cublasH,opl,opr,blsNr,blsMl,blsComm,(cuDoubleComplex*)&alphas[i],(cuDoubleComplex*)&b_array[idx],blsNr,(cuDoubleComplex*)&a_array[idx],blsComm,(cuDoubleComplex*)&betas[i],(cuDoubleComplex*)&c_array[idx],blsNr));
                    idx++;
                    cublasDestroy(cublasH);
                }
            }            
        }

        void cuGemm_Batch_internal_cf(const char *transa_array, const char *transb_array, const blas_int *m_array, const blas_int *n_array, const blas_int *k_array,
                const std::vector<Scalar> &alpha_array, const void **a_array, const blas_int *lda_array, const void **b_array, const blas_int *ldb_array,
                const std::vector<Scalar> &beta_array, void **c_array, const blas_int *ldc_array, const blas_int group_count, const blas_int *group_size){
            cytnx_complex64 alphas[alpha_array.size()];
            cytnx_complex64 betas[beta_array.size()];
            for (size_t i=0; i<alpha_array.size(); i++) alphas[i] = complex64(alpha_array[i]);
            for (size_t i=0; i<beta_array.size(); i++) betas[i] = complex64(beta_array[i]);

            cytnx_uint64 idx = 0;
            for(cytnx_uint64 i=0;i<group_count;i++){
                for(cytnx_uint64 j=0;j<group_size[i];j++){
                    cytnx_int32 blsMl=m_array[i], blsNr=n_array[i], blsComm=k_array[i];
                    cublasOperation_t opl=CUBLAS_OP_N, opr=CUBLAS_OP_N;
                    switch (transa_array[i]) {
                        case 'N':
                            opl = CUBLAS_OP_N;
                            break;
                        case 'T':
                            opl = CUBLAS_OP_T;
                            break;
                        case 'C':
                            opl = CUBLAS_OP_C;
                            break;
                        default:
                            break;
                    }
                    switch (transb_array[i]) {
                        case 'N':
                            opr = CUBLAS_OP_N;
                            break;
                        case 'T':
                            opr = CUBLAS_OP_T;
                            break;
                        case 'C':
                            opr = CUBLAS_OP_C;
                            break;
                        default:
                            break;
                    }
                    // create handles:
                    cublasHandle_t cublasH = NULL;
                    checkCudaErrors(cublasCreate(&cublasH));
                    checkCudaErrors(cublasCgemm(cublasH,opl,opr,blsNr,blsMl,blsComm,(cuFloatComplex*)&alphas[i],(cuFloatComplex*)&b_array[idx],blsNr,(cuFloatComplex*)&a_array[idx],blsComm,(cuFloatComplex*)&betas[i],(cuFloatComplex*)&c_array[idx],blsNr));
                    idx++;
                    cublasDestroy(cublasH);
                }
            }
        }


        void cuGemm_Batch_internal_d(const char *transa_array, const char *transb_array, const blas_int *m_array, const blas_int *n_array, const blas_int *k_array,
                const std::vector<Scalar> &alpha_array, const void **a_array, const blas_int *lda_array, const void **b_array, const blas_int *ldb_array,
                const std::vector<Scalar> &beta_array, void **c_array, const blas_int *ldc_array, const blas_int group_count, const blas_int *group_size){
            cytnx_double alphas[alpha_array.size()];
            cytnx_double betas[beta_array.size()];
            for (size_t i=0; i<alpha_array.size(); i++) alphas[i] = double(alpha_array[i]);
            for (size_t i=0; i<beta_array.size(); i++) betas[i] = double(beta_array[i]);

            cytnx_uint64 idx = 0;
            for(cytnx_uint64 i=0;i<group_count;i++){
                for(cytnx_uint64 j=0;j<group_size[i];j++){
                    cytnx_int32 blsMl=m_array[i], blsNr=n_array[i], blsComm=k_array[i];
                    cublasOperation_t opl=CUBLAS_OP_N, opr=CUBLAS_OP_N;
                    switch (transa_array[i]) {
                        case 'N':
                            opl = CUBLAS_OP_N;
                            break;
                        case 'T':
                            opl = CUBLAS_OP_T;
                            break;
                        case 'C':
                            opl = CUBLAS_OP_C;
                            break;
                        default:
                            break;
                    }
                    switch (transb_array[i]) {
                        case 'N':
                            opr = CUBLAS_OP_N;
                            break;
                        case 'T':
                            opr = CUBLAS_OP_T;
                            break;
                        case 'C':
                            opr = CUBLAS_OP_C;
                            break;
                        default:
                            break;
                    }
                    // create handles:
                    cublasHandle_t cublasH = NULL;
                    checkCudaErrors(cublasCreate(&cublasH));
                    checkCudaErrors(cublasDgemm(cublasH,opl,opr,blsNr,blsMl,blsComm,(cytnx_double*)&alphas[i],(cytnx_double*)&b_array[idx],blsNr,(cytnx_double*)&a_array[idx],blsComm,(cytnx_double*)&betas[i],(cytnx_double*)&c_array[idx],blsNr));
                    idx++;
                    cublasDestroy(cublasH);
                }
            }
        }
        void cuGemm_Batch_internal_f(const char *transa_array, const char *transb_array, const blas_int *m_array, const blas_int *n_array, const blas_int *k_array,
                const std::vector<Scalar> &alpha_array, const void **a_array, const blas_int *lda_array, const void **b_array, const blas_int *ldb_array,
                const std::vector<Scalar> &beta_array, void **c_array, const blas_int *ldc_array, const blas_int group_count, const blas_int *group_size){
            cytnx_float alphas[alpha_array.size()];
            cytnx_float betas[beta_array.size()];
            for (size_t i=0; i<alpha_array.size(); i++) alphas[i] = float(alpha_array[i]);
            for (size_t i=0; i<beta_array.size(); i++) betas[i] = float(beta_array[i]);

            cytnx_uint64 idx = 0;
            for(cytnx_uint64 i=0;i<group_count;i++){
                for(cytnx_uint64 j=0;j<group_size[i];j++){
                    cytnx_int32 blsMl=m_array[i], blsNr=n_array[i], blsComm=k_array[i];
                    cublasOperation_t opl=CUBLAS_OP_N, opr=CUBLAS_OP_N;
                    switch (transa_array[i]) {
                        case 'N':
                            opl = CUBLAS_OP_N;
                            break;
                        case 'T':
                            opl = CUBLAS_OP_T;
                            break;
                        case 'C':
                            opl = CUBLAS_OP_C;
                            break;
                        default:
                            break;
                    }
                    switch (transb_array[i]) {
                        case 'N':
                            opr = CUBLAS_OP_N;
                            break;
                        case 'T':
                            opr = CUBLAS_OP_T;
                            break;
                        case 'C':
                            opr = CUBLAS_OP_C;
                            break;
                        default:
                            break;
                    }
                    // create handles:
                    cublasHandle_t cublasH = NULL;
                    checkCudaErrors(cublasCreate(&cublasH));
                    checkCudaErrors(cublasSgemm(cublasH,opl,opr,blsNr,blsMl,blsComm,(cytnx_float*)&alphas[i],(cytnx_float*)&b_array[idx],blsNr,(cytnx_float*)&a_array[idx],blsComm,(cytnx_float*)&betas[i],(cytnx_float*)&c_array[idx],blsNr));
                    idx++;
                    cublasDestroy(cublasH);
                }
            }
        }

    }    
}



