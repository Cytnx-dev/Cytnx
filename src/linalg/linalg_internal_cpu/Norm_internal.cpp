#include "Norm_internal.hpp"
#include "utils/utils_internal_interface.hpp"
#include "utils/utils.hpp"
#include "lapack_wrapper.hpp"
#ifdef UNI_OMP
  #include <omp.h>
#endif

namespace cytnx {

  namespace linalg_internal {

    /// Norm
    void Norm_internal_cd(void *out, const boost::intrusive_ptr<Storage_base> &Rin) {
      double *od = static_cast<double *>(out);
      cytnx_complex128 *_Rin = (cytnx_complex128 *)Rin->Mem;

      blas_int incx = 1;
      blas_int Len = Rin->len;
      *od = dznrm2(&Len, _Rin, &incx);
    }
    void Norm_internal_cf(void *out, const boost::intrusive_ptr<Storage_base> &Rin) {
      float *od = static_cast<float *>(out);
      cytnx_complex64 *_Rin = (cytnx_complex64 *)Rin->Mem;

      blas_int incx = 1;
      blas_int Len = Rin->len;
      *od = scnrm2(&Len, _Rin, &incx);
    }
    void Norm_internal_d(void *out, const boost::intrusive_ptr<Storage_base> &Rin) {
      double *od = static_cast<double *>(out);
      cytnx_double *_Rin = (cytnx_double *)Rin->Mem;

      blas_int incx = 1;
      blas_int Len = Rin->len;
      *od = dnrm2(&Len, _Rin, &incx);
    }
    void Norm_internal_f(void *out, const boost::intrusive_ptr<Storage_base> &Rin) {
      float *od = static_cast<float *>(out);
      cytnx_float *_Rin = (cytnx_float *)Rin->Mem;

      blas_int incx = 1;
      blas_int Len = Rin->len;
      *od = snrm2(&Len, _Rin, &incx);
    }

  }  // namespace linalg_internal
}  // namespace cytnx
