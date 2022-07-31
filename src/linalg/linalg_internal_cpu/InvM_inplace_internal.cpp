#include "cytnx_error.hpp"
#include "InvM_inplace_internal.hpp"
#include "lapack_wrapper.hpp"

namespace cytnx {

  namespace linalg_internal {

    void InvM_inplace_internal_d(boost::intrusive_ptr<Storage_base> &iten, const cytnx_int64 &L) {
      lapack_int *ipiv = (lapack_int *)malloc((L + 1) * sizeof(lapack_int));
      lapack_int info;
      info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, L, L, (cytnx_double *)iten->Mem, L, ipiv);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'dgetrf': Lapack INFO = ", info);

      info = LAPACKE_dgetri(LAPACK_COL_MAJOR, L, (cytnx_double *)iten->Mem, L, ipiv);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'dgetri': Lapack INFO = ", info);

      free(ipiv);
    }

    void InvM_inplace_internal_f(boost::intrusive_ptr<Storage_base> &iten, const cytnx_int64 &L) {
      lapack_int *ipiv = (lapack_int *)malloc((L + 1) * sizeof(lapack_int));
      lapack_int info;
      info = LAPACKE_sgetrf(LAPACK_COL_MAJOR, L, L, (cytnx_float *)iten->Mem, L, ipiv);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'sgetrf': Lapack INFO = ", info);

      info = LAPACKE_sgetri(LAPACK_COL_MAJOR, L, (cytnx_float *)iten->Mem, L, ipiv);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'sgetri': Lapack INFO = ", info);
      free(ipiv);
    }

    void InvM_inplace_internal_cd(boost::intrusive_ptr<Storage_base> &iten, const cytnx_int64 &L) {
      lapack_int *ipiv = (lapack_int *)malloc((L + 1) * sizeof(lapack_int));
      lapack_int info;
      info = LAPACKE_zgetrf(LAPACK_COL_MAJOR, L, L, (lapack_complex_double *)iten->Mem, L, ipiv);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'zgetrf': Lapack INFO = ", info);

      info = LAPACKE_zgetri(LAPACK_COL_MAJOR, L, (lapack_complex_double *)iten->Mem, L, ipiv);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'zgetri': Lapack INFO = ", info);

      free(ipiv);
    }

    void InvM_inplace_internal_cf(boost::intrusive_ptr<Storage_base> &iten, const cytnx_int64 &L) {
      lapack_int *ipiv = (lapack_int *)malloc((L + 1) * sizeof(lapack_int));
      lapack_int info;
      info = LAPACKE_cgetrf(LAPACK_COL_MAJOR, L, L, (lapack_complex_float *)iten->Mem, L, ipiv);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'cgetrf': Lapack INFO = ", info);

      info = LAPACKE_cgetri(LAPACK_COL_MAJOR, L, (lapack_complex_float *)iten->Mem, L, ipiv);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'cgetri': Lapack INFO = ", info);

      free(ipiv);
    }

  }  // namespace linalg_internal

}  // namespace cytnx
