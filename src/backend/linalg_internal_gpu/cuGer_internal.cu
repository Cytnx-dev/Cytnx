#include "cuGer_internal.hpp"
#include "../utils_internal_interface.hpp"

#include "backend/lapack_wrapper.hpp"

#ifdef UNI_OMP
  #include <omp.h>
#endif

namespace cytnx {
  namespace linalg_internal {

    void cuGer_internal_cd(boost::intrusive_ptr<Storage_base> &A,
                           const boost::intrusive_ptr<Storage_base> &x,
                           const boost::intrusive_ptr<Storage_base> &y, const Scalar &a) {
      // create handles:
      cublasHandle_t cublasH = NULL;
      checkCudaErrors(cublasCreate(&cublasH));
      cytnx_complex128 alpha = complex128(a);

      cuDoubleComplex *_A = (cuDoubleComplex *)A->data();
      cuDoubleComplex *_x = (cuDoubleComplex *)x->data();
      cuDoubleComplex *_y = (cuDoubleComplex *)y->data();

      checkCudaErrors(cublasZgeru(cublasH, y->size(), x->size(), (cuDoubleComplex *)&alpha, _y, 1,
                                  _x, 1, _A, y->size()));

      cublasDestroy(cublasH);
    }

    void cuGer_internal_cf(boost::intrusive_ptr<Storage_base> &A,
                           const boost::intrusive_ptr<Storage_base> &x,
                           const boost::intrusive_ptr<Storage_base> &y, const Scalar &a) {
      // create handles:
      cublasHandle_t cublasH = NULL;
      checkCudaErrors(cublasCreate(&cublasH));
      cytnx_complex64 alpha = complex64(a);

      cuFloatComplex *_A = (cuFloatComplex *)A->data();
      cuFloatComplex *_x = (cuFloatComplex *)x->data();
      cuFloatComplex *_y = (cuFloatComplex *)y->data();

      checkCudaErrors(cublasCgeru(cublasH, y->size(), x->size(), (cuFloatComplex *)&alpha, _y, 1,
                                  _x, 1, _A, y->size()));

      cublasDestroy(cublasH);
    }

    void cuGer_internal_d(boost::intrusive_ptr<Storage_base> &A,
                          const boost::intrusive_ptr<Storage_base> &x,
                          const boost::intrusive_ptr<Storage_base> &y, const Scalar &a) {
      // create handles:
      cublasHandle_t cublasH = NULL;
      checkCudaErrors(cublasCreate(&cublasH));
      cytnx_double alpha = cytnx_double(a);

      cytnx_double *_A = (cytnx_double *)A->data();
      cytnx_double *_x = (cytnx_double *)x->data();
      cytnx_double *_y = (cytnx_double *)y->data();

      checkCudaErrors(
        cublasDger(cublasH, y->size(), x->size(), &alpha, _y, 1, _x, 1, _A, y->size()));

      cublasDestroy(cublasH);
    }

    void cuGer_internal_f(boost::intrusive_ptr<Storage_base> &A,
                          const boost::intrusive_ptr<Storage_base> &x,
                          const boost::intrusive_ptr<Storage_base> &y, const Scalar &a) {
      // create handles:
      cublasHandle_t cublasH = NULL;
      checkCudaErrors(cublasCreate(&cublasH));
      cytnx_float alpha = cytnx_float(a);

      cytnx_float *_A = (cytnx_float *)A->data();
      cytnx_float *_x = (cytnx_float *)x->data();
      cytnx_float *_y = (cytnx_float *)y->data();

      checkCudaErrors(
        cublasSger(cublasH, y->size(), x->size(), &alpha, _y, 1, _x, 1, _A, y->size()));

      cublasDestroy(cublasH);
    }

  }  // namespace linalg_internal
}  // namespace cytnx
