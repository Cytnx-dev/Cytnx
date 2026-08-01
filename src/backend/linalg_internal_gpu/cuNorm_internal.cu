#include "cuNorm_internal.hpp"
#include "backend/utils_internal_interface.hpp"
#include "utils/utils.hpp"
#include "cytnx_error.hpp"
#include "backend/lapack_wrapper.hpp"
#include "backend/utils_internal_gpu/cuLibraryHandle_gpu.hpp"

namespace cytnx {

  namespace linalg_internal {

    /// cuNorm
    void cuNorm_internal_cd(void *out, const boost::intrusive_ptr<Storage_base> &Rin) {
      // Shared per-device handle (#1144): cublasCreate costs ~330 us per call,
      // up to 20x the cost of the small GEMMs it wraps.
      cublasHandle_t cublasH = utils_internal::get_cublas_handle();

      checkCudaErrors(
        cublasDznrm2(cublasH, Rin->size(), (cuDoubleComplex *)Rin->data(), 1, (double *)out));
    }
    void cuNorm_internal_cf(void *out, const boost::intrusive_ptr<Storage_base> &Rin) {
      // Shared per-device handle (#1144): cublasCreate costs ~330 us per call,
      // up to 20x the cost of the small GEMMs it wraps.
      cublasHandle_t cublasH = utils_internal::get_cublas_handle();

      checkCudaErrors(
        cublasScnrm2(cublasH, Rin->size(), (cuComplex *)Rin->data(), 1, (float *)out));
    }
    void cuNorm_internal_d(void *out, const boost::intrusive_ptr<Storage_base> &Rin) {
      // Shared per-device handle (#1144): cublasCreate costs ~330 us per call,
      // up to 20x the cost of the small GEMMs it wraps.
      cublasHandle_t cublasH = utils_internal::get_cublas_handle();

      checkCudaErrors(cublasDnrm2(cublasH, Rin->size(), (double *)Rin->data(), 1, (double *)out));
    }
    void cuNorm_internal_f(void *out, const boost::intrusive_ptr<Storage_base> &Rin) {
      // Shared per-device handle (#1144): cublasCreate costs ~330 us per call,
      // up to 20x the cost of the small GEMMs it wraps.
      cublasHandle_t cublasH = utils_internal::get_cublas_handle();
      checkCudaErrors(cublasSnrm2(cublasH, Rin->size(), (float *)Rin->data(), 1, (float *)out));
    }

  }  // namespace linalg_internal
}  // namespace cytnx
