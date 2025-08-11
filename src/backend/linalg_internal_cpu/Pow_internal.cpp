#include "backend/linalg_internal_cpu/Pow_internal.hpp"
#include "cytnx_error.hpp"

namespace cytnx {
  namespace linalg_internal {

    void Pow_internal_d(boost::intrusive_ptr<Storage_base> &out,
                        const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                        const cytnx_double &p) {
      cytnx_double *_out = (cytnx_double *)out->data();
      cytnx_double *_ten = (cytnx_double *)ten->data();

      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[n] = pow(_ten[n], p);
      }
    }

    void Pow_internal_f(boost::intrusive_ptr<Storage_base> &out,
                        const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                        const cytnx_double &p) {
      cytnx_float *_out = (cytnx_float *)out->data();
      cytnx_float *_ten = (cytnx_float *)ten->data();

      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[n] = powf(_ten[n], p);
      }
    }

    void Pow_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                         const cytnx_double &p) {
      cytnx_complex128 *_out = (cytnx_complex128 *)out->data();
      cytnx_complex128 *_ten = (cytnx_complex128 *)ten->data();

      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[n] = pow(_ten[n], p);
      }
    }

    void Pow_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                         const cytnx_double &p) {
      cytnx_complex64 *_out = (cytnx_complex64 *)out->data();
      cytnx_complex64 *_ten = (cytnx_complex64 *)ten->data();

      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[n] = pow(_ten[n], p);
      }
    }

  }  // namespace linalg_internal

}  // namespace cytnx
