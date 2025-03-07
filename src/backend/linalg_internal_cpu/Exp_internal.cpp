#include "Exp_internal.hpp"
#include "cytnx_error.hpp"
#include "backend/lapack_wrapper.hpp"

#ifdef UNI_OMP
  #include <omp.h>
#endif

namespace cytnx {
  namespace linalg_internal {

    void Exp_internal_d(boost::intrusive_ptr<Storage_base> &out,
                        const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem) {
      cytnx_double *_ten = (cytnx_double *)ten->data();
      cytnx_double *_out = (cytnx_double *)out->data();

#pragma omp parallel for schedule(auto)
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[n] = exp(_ten[n]);
      }
    }

    void Exp_internal_f(boost::intrusive_ptr<Storage_base> &out,
                        const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem) {
      cytnx_float *_ten = (cytnx_float *)ten->data();
      cytnx_float *_out = (cytnx_float *)out->data();

#pragma omp parallel for
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[n] = expf(_ten[n]);
      }
    }

    void Exp_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem) {
      cytnx_complex128 *_out = (cytnx_complex128 *)out->data();
      cytnx_complex128 *_ten = (cytnx_complex128 *)ten->data();

#pragma omp parallel for schedule(auto)
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[n] = exp(_ten[n]);
      }
    }

    void Exp_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem) {
      cytnx_complex64 *_out = (cytnx_complex64 *)out->data();
      cytnx_complex64 *_ten = (cytnx_complex64 *)ten->data();

#pragma omp parallel for
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[n] = exp(_ten[n]);
      }
    }

  }  // namespace linalg_internal

}  // namespace cytnx
