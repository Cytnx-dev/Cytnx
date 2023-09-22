#include "Ger_internal.hpp"
#include "cytnx_error.hpp"
#include "backend/lapack_wrapper.hpp"

namespace cytnx {
  namespace linalg_internal {

    void Ger_internal_cd(boost::intrusive_ptr<Storage_base> &A,
                         const boost::intrusive_ptr<Storage_base> &x,
                         const boost::intrusive_ptr<Storage_base> &y, const Scalar &a) {
      cytnx_complex128 *_A = (cytnx_complex128 *)A->Mem;
      cytnx_complex128 *_x = (cytnx_complex128 *)x->Mem;
      cytnx_complex128 *_y = (cytnx_complex128 *)y->Mem;
      cytnx_complex128 _a = complex128(a);

      cblas_zgeru(CblasRowMajor, x->size(), y->size(), (double *)&_a, (double *)_x, 1, (double *)_y,
                  1, (double *)_A, y->size());
    }

    void Ger_internal_cf(boost::intrusive_ptr<Storage_base> &A,
                         const boost::intrusive_ptr<Storage_base> &x,
                         const boost::intrusive_ptr<Storage_base> &y, const Scalar &a) {
      cytnx_complex64 *_A = (cytnx_complex64 *)A->Mem;
      cytnx_complex64 *_x = (cytnx_complex64 *)x->Mem;
      cytnx_complex64 *_y = (cytnx_complex64 *)y->Mem;
      cytnx_complex64 _a = complex64(a);

      cblas_cgeru(CblasRowMajor, x->size(), y->size(), (float *)&_a, (float *)_x, 1, (float *)_y, 1,
                  (float *)_A, y->size());
    }

    void Ger_internal_d(boost::intrusive_ptr<Storage_base> &A,
                        const boost::intrusive_ptr<Storage_base> &x,
                        const boost::intrusive_ptr<Storage_base> &y, const Scalar &a) {
      cytnx_double *_A = (cytnx_double *)A->Mem;
      cytnx_double *_x = (cytnx_double *)x->Mem;
      cytnx_double *_y = (cytnx_double *)y->Mem;
      cytnx_double _a = double(a);

      cblas_dger(CblasRowMajor, x->size(), y->size(), _a, _x, 1, _y, 1, _A, y->size());
    }

    void Ger_internal_f(boost::intrusive_ptr<Storage_base> &A,
                        const boost::intrusive_ptr<Storage_base> &x,
                        const boost::intrusive_ptr<Storage_base> &y, const Scalar &a) {
      cytnx_float *_A = (cytnx_float *)A->Mem;
      cytnx_float *_x = (cytnx_float *)x->Mem;
      cytnx_float *_y = (cytnx_float *)y->Mem;
      cytnx_float _a = float(a);

      cblas_sger(CblasRowMajor, x->size(), y->size(), _a, _x, 1, _y, 1, _A, y->size());
    }

  }  // namespace linalg_internal

};  // namespace cytnx
