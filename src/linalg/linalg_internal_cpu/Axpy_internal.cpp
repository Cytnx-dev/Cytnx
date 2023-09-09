#include "Axpy_internal.hpp"
#include "cytnx_error.hpp"
#include "backend/lapack_wrapper.hpp"

namespace cytnx {
  namespace linalg_internal {

    void Axpy_internal_cd(const boost::intrusive_ptr<Storage_base> &x,
                          boost::intrusive_ptr<Storage_base> &y, const Scalar &a) {
      cytnx_complex128 *_x = (cytnx_complex128 *)x->Mem;
      cytnx_complex128 *_y = (cytnx_complex128 *)y->Mem;
      cytnx_complex128 _a = complex128(a);
      cblas_zaxpy(x->size(), (double *)&_a, (double *)_x, 1, (double *)_y, 1);
    }

    void Axpy_internal_cf(const boost::intrusive_ptr<Storage_base> &x,
                          boost::intrusive_ptr<Storage_base> &y, const Scalar &a) {
      cytnx_complex64 *_x = (cytnx_complex64 *)x->Mem;
      cytnx_complex64 *_y = (cytnx_complex64 *)y->Mem;
      cytnx_complex64 _a = complex64(a);
      cblas_caxpy(x->size(), (float *)&_a, (float *)_x, 1, (float *)_y, 1);
    }

    void Axpy_internal_d(const boost::intrusive_ptr<Storage_base> &x,
                         boost::intrusive_ptr<Storage_base> &y, const Scalar &a) {
      cytnx_double *_x = (cytnx_double *)x->Mem;
      cytnx_double *_y = (cytnx_double *)y->Mem;
      cytnx_double _a = double(a);
      cblas_daxpy(x->size(), _a, _x, 1, _y, 1);
    }

    void Axpy_internal_f(const boost::intrusive_ptr<Storage_base> &x,
                         boost::intrusive_ptr<Storage_base> &y, const Scalar &a) {
      cytnx_float *_x = (cytnx_float *)x->Mem;
      cytnx_float *_y = (cytnx_float *)y->Mem;
      cytnx_float _a = float(a);
      cblas_saxpy(x->size(), _a, _x, 1, _y, 1);
    }

  }  // namespace linalg_internal

};  // namespace cytnx
