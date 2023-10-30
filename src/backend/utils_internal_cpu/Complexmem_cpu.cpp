#include "Complexmem_cpu.hpp"
#include "backend/Storage.hpp"
#include "backend/lapack_wrapper.hpp"

#ifdef UNI_OMP
  #include <omp.h>
#endif

#ifdef UNI_MKL
  #define __Cpt_dbl cytnx_complex128 *
  #define __Cpt_flt cytnx_complex64 *
#else
  #define __Cpt_dbl __complex__ double *
  #define __Cpt_flt __complex__ float *
#endif

using namespace std;
namespace cytnx {

  namespace utils_internal {

    void Complexmem_cpu_cdtd(void *out, void *in, const cytnx_uint64 &Nelem, const bool get_real) {
      cytnx_double *des = static_cast<cytnx_double *>(out);
      cytnx_complex128 *src = static_cast<cytnx_complex128 *>(in);

      if (get_real) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          des[n] = src[n].real();
        }
      } else {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          des[n] = src[n].imag();
        }
      }
    }

    void Complexmem_cpu_cftf(void *out, void *in, const cytnx_uint64 &Nelem, const bool get_real) {
      cytnx_float *des = static_cast<cytnx_float *>(out);
      cytnx_complex64 *src = static_cast<cytnx_complex64 *>(in);

      if (get_real) {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          des[n] = src[n].real();
        }
      } else {
#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
        for (cytnx_uint64 n = 0; n < Nelem; n++) {
          des[n] = src[n].imag();
        }
      }
    }

    void ComplexMatrix_from_real_cd(void *out, void *in, const cytnx_uint64 &m,
                                    const cytnx_uint64 &n, const bool real_part) {
      if (real_part)
        LAPACKE_zlacp2(LAPACK_ROW_MAJOR, 'A', m, n, (double *)in, n, (__Cpt_dbl)out, n);
      else
        LAPACKE_zlacp2(LAPACK_ROW_MAJOR, 'A', m, n, (double *)in, n,
                       (__Cpt_dbl)(&((cytnx_double *)in)[1]), n);
    }
    void ComplexMatrix_from_real_cf(void *out, void *in, const cytnx_uint64 &m,
                                    const cytnx_uint64 &n, const bool real_part) {
      if (real_part)
        LAPACKE_clacp2(LAPACK_ROW_MAJOR, 'A', m, n, (float *)in, n, (__Cpt_flt)out, n);
      else
        LAPACKE_clacp2(LAPACK_ROW_MAJOR, 'A', m, n, (float *)in, n,
                       (__Cpt_flt)(&((cytnx_float *)in)[1]), n);
    }

  }  // namespace utils_internal

}  // namespace cytnx
