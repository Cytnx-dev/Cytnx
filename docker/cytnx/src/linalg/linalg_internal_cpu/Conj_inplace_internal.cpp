#include "linalg/linalg_internal_cpu/Conj_inplace_internal.hpp"
#include "cytnx_error.hpp"
#include "utils/lapack_wrapper.hpp"

#ifdef UNI_OMP
  #include <omp.h>
#endif

namespace cytnx {
  namespace linalg_internal {

    void Conj_inplace_internal_cf(boost::intrusive_ptr<Storage_base> &ten,
                                  const cytnx_uint64 &Nelem) {
      cytnx_complex64 *tmp = (cytnx_complex64 *)ten->Mem;

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        tmp[n].imag(-tmp[n].imag());
      }
    }

    void Conj_inplace_internal_cd(boost::intrusive_ptr<Storage_base> &ten,
                                  const cytnx_uint64 &Nelem) {
      cytnx_complex128 *tmp = (cytnx_complex128 *)ten->Mem;

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 n = 0; n < Nelem; n++) {
        tmp[n].imag(-tmp[n].imag());
      }
    }

  }  // namespace linalg_internal

}  // namespace cytnx
