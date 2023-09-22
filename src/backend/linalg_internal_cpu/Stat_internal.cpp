#include "backend/linalg_internal_cpu/Stat_internal.hpp"
#include "cytnx_error.hpp"

#ifdef UNI_OMP
  #include <omp.h>
#endif

namespace cytnx {
  namespace linalg_internal {

    void Stat_internal_d(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &ten) {
      cytnx_double *_out = (cytnx_double *)out->Mem;
      cytnx_double *_ten = (cytnx_double *)ten->Mem;

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[n] = pow(_ten[n], p);
      }
    }

    void Stat_internal_f(boost::intrusive_ptr<Storage_base> &out,
                         const boost::intrusive_ptr<Storage_base> &ten) {
      cytnx_float *_out = (cytnx_float *)out->Mem;
      cytnx_float *_ten = (cytnx_float *)ten->Mem;

#ifdef UNI_OMP
  #pragma omp parallel for
#endif
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[n] = powf(_ten[n], p);
      }
    }

    void Stat_internal_cd(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten) {
      cytnx_complex128 *_out = (cytnx_complex128 *)out->Mem;
      cytnx_complex128 *_ten = (cytnx_complex128 *)ten->Mem;

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[n] = pow(_ten[n], p);
      }
    }

    void Stat_internal_cf(boost::intrusive_ptr<Storage_base> &out,
                          const boost::intrusive_ptr<Storage_base> &ten) {
      cytnx_complex64 *_out = (cytnx_complex64 *)out->Mem;
      cytnx_complex64 *_ten = (cytnx_complex64 *)ten->Mem;

#ifdef UNI_OMP
  #pragma omp parallel for
#endif
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        _out[n] = pow(_ten[n], p);
      }
    }

  }  // namespace linalg_internal

}  // namespace cytnx
