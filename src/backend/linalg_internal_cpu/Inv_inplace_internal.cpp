#include "Inv_inplace_internal.hpp"
#include "cytnx_error.hpp"
#include "backend/lapack_wrapper.hpp"

namespace cytnx {
  namespace linalg_internal {

    void Inv_inplace_internal_d(boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                                const double &clip) {
      cytnx_double *_ten = (cytnx_double *)ten->data();

      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _ten[n] = std::abs(_ten[n]) < clip ? 0. : 1. / _ten[n];
      }
    }

    void Inv_inplace_internal_f(boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                                const double &clip) {
      cytnx_float *_ten = (cytnx_float *)ten->data();

      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _ten[n] = std::abs(_ten[n]) < clip ? 0.0f : 1.0f / _ten[n];
      }
    }

    void Inv_inplace_internal_cd(boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                                 const double &clip) {
      cytnx_complex128 *_ten = (cytnx_complex128 *)ten->data();

      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _ten[n] =
          std::norm(_ten[n]) < clip ? cytnx_complex128(0., 0.) : cytnx_complex128(1., 0.) / _ten[n];
      }
    }

    void Inv_inplace_internal_cf(boost::intrusive_ptr<Storage_base> &ten, const cytnx_uint64 &Nelem,
                                 const double &clip) {
      cytnx_complex64 *_ten = (cytnx_complex64 *)ten->data();

      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _ten[n] = std::norm(_ten[n]) < clip ? cytnx_complex64(0.0f, 0.0f)
                                            : cytnx_complex64(1.0f, 0.0f) / _ten[n];
      }
    }

  }  // namespace linalg_internal

}  // namespace cytnx
