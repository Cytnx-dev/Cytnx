#include "Det_internal.hpp"
#include "utils/utils_internal_interface.hpp"
#include "utils/utils.hpp"
#include "lapack_wrapper.hpp"
#ifdef UNI_OMP
  #include <omp.h>
#endif

namespace cytnx {

  namespace linalg_internal {

    /// Det
    void Det_internal_cd(void *out, const boost::intrusive_ptr<Storage_base> &Rin,
                         const cytnx_uint64 &L) {
      cytnx_complex128 *od = static_cast<cytnx_complex128 *>(out);
      lapack_complex_double *_Rin =
        (lapack_complex_double *)malloc(sizeof(cytnx_complex128) * Rin->len);
      memcpy(_Rin, Rin->Mem, sizeof(cytnx_complex128) * Rin->len);

      lapack_int *ipiv = (lapack_int *)malloc((L + 1) * sizeof(lapack_int));
      lapack_int N = L;
      lapack_int info;
      info = LAPACKE_zgetrf(LAPACK_COL_MAJOR, N, N, _Rin, N, ipiv);
      cytnx_error_msg(
        info != 0, "%s %d",
        "[ERROR][Det_internal] Error in Lapack function 'zgetrf': Lapack INFO = ", info);
      od[0] = 1;
      bool neg = 0;
      for (lapack_int i = 0; i < N; i++) {
        od[0] *= (cytnx_complex128)_Rin[i * N + i];
        if (ipiv[i] != (i + 1)) neg = !neg;
      }
      free(ipiv);
      free(_Rin);
      if (neg) od[0] *= -1;
    }
    void Det_internal_cf(void *out, const boost::intrusive_ptr<Storage_base> &Rin,
                         const cytnx_uint64 &L) {
      cytnx_complex64 *od = static_cast<cytnx_complex64 *>(out);
      lapack_complex_float *_Rin =
        (lapack_complex_float *)malloc(sizeof(cytnx_complex64) * Rin->len);
      memcpy(_Rin, Rin->Mem, sizeof(cytnx_complex64) * Rin->len);

      lapack_int *ipiv = (lapack_int *)malloc((L + 1) * sizeof(lapack_int));
      lapack_int N = L;
      lapack_int info;
      info = LAPACKE_cgetrf(LAPACK_COL_MAJOR, N, N, _Rin, N, ipiv);
      cytnx_error_msg(
        info != 0, "%s %d",
        "[ERROR][Det_internal] Error in Lapack function 'cgetrf': Lapack INFO = ", info);
      od[0] = 1;
      bool neg = 0;
      for (lapack_int i = 0; i < N; i++) {
        od[0] *= (cytnx_complex64)_Rin[i * N + i];
        if (ipiv[i] != (i + 1)) neg = !neg;
      }
      free(ipiv);
      free(_Rin);
      if (neg) od[0] *= -1;
    }
    void Det_internal_d(void *out, const boost::intrusive_ptr<Storage_base> &Rin,
                        const cytnx_uint64 &L) {
      cytnx_double *od = static_cast<cytnx_double *>(out);
      cytnx_double *_Rin = (cytnx_double *)malloc(sizeof(cytnx_double) * Rin->len);
      memcpy(_Rin, Rin->Mem, sizeof(cytnx_double) * Rin->len);

      lapack_int *ipiv = (lapack_int *)malloc((L + 1) * sizeof(lapack_int));
      lapack_int N = L;
      lapack_int info;
      info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, N, N, _Rin, N, ipiv);
      cytnx_error_msg(
        info != 0, "%s %d",
        "[ERROR][Det_internal] Error in Lapack function 'dgetrf': Lapack INFO = ", info);
      od[0] = 1;
      bool neg = 0;
      for (lapack_int i = 0; i < N; i++) {
        od[0] *= _Rin[i * N + i];
        if (ipiv[i] != (i + 1)) neg = !neg;
      }
      free(ipiv);
      free(_Rin);
      if (neg) od[0] *= -1;
    }
    void Det_internal_f(void *out, const boost::intrusive_ptr<Storage_base> &Rin,
                        const cytnx_uint64 &L) {
      float *od = static_cast<float *>(out);
      cytnx_float *_Rin = (cytnx_float *)malloc(sizeof(cytnx_float) * Rin->len);
      memcpy(_Rin, Rin->Mem, sizeof(cytnx_float) * Rin->len);

      lapack_int *ipiv = (lapack_int *)malloc((L + 1) * sizeof(lapack_int));
      lapack_int N = L;
      lapack_int info;
      info = LAPACKE_sgetrf(LAPACK_COL_MAJOR, N, N, _Rin, N, ipiv);
      cytnx_error_msg(
        info != 0, "%s %d",
        "[ERROR][Det_internal] Error in Lapack function 'sgetrf': Lapack INFO = ", info);
      od[0] = 1;
      bool neg = 0;
      for (lapack_int i = 0; i < N; i++) {
        od[0] *= _Rin[i * N + i];
        if (ipiv[i] != (i + 1)) neg = !neg;
      }
      free(ipiv);
      free(_Rin);
      if (neg) od[0] *= -1;
    }

  }  // namespace linalg_internal
}  // namespace cytnx
