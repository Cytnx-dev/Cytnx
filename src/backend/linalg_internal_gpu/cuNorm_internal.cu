#include "cuNorm_internal.hpp"
#include "../utils_internal_interface.hpp"
#include "utils/utils.hpp"
#include "cytnx_error.hpp"
#include "backend/lapack_wrapper.hpp"

#ifdef UNI_OMP
  #include <omp.h>
#endif

namespace cytnx {

  namespace linalg_internal {

    /// cuNorm
    void cuNorm_internal_cd(void *out, const boost::intrusive_ptr<Storage_base> &Rin) {
      cublasHandle_t cublasH = NULL;
      checkCudaErrors(cublasCreate(&cublasH));

      checkCudaErrors(
        cublasDznrm2(cublasH, Rin->len, (cuDoubleComplex *)Rin->Mem, 1, (double *)out));

      cublasDestroy(cublasH);
    }
    void cuNorm_internal_cf(void *out, const boost::intrusive_ptr<Storage_base> &Rin) {
      cublasHandle_t cublasH = NULL;
      checkCudaErrors(cublasCreate(&cublasH));

      checkCudaErrors(cublasScnrm2(cublasH, Rin->len, (cuComplex *)Rin->Mem, 1, (float *)out));

      cublasDestroy(cublasH);
    }
    void cuNorm_internal_d(void *out, const boost::intrusive_ptr<Storage_base> &Rin) {
      cublasHandle_t cublasH = NULL;
      checkCudaErrors(cublasCreate(&cublasH));

      checkCudaErrors(cublasDnrm2(cublasH, Rin->len, (double *)Rin->Mem, 1, (double *)out));

      cublasDestroy(cublasH);
    }
    void cuNorm_internal_f(void *out, const boost::intrusive_ptr<Storage_base> &Rin) {
      cublasHandle_t cublasH = NULL;
      checkCudaErrors(cublasCreate(&cublasH));
      checkCudaErrors(cublasSnrm2(cublasH, Rin->len, (float *)Rin->Mem, 1, (float *)out));
      cublasDestroy(cublasH);
    }

  }  // namespace linalg_internal
}  // namespace cytnx
