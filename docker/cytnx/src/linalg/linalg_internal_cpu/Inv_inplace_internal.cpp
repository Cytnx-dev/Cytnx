#include "cytnx_error.hpp"
#include "Inv_inplace_internal.hpp"
#include "lapack_wrapper.hpp"

namespace cytnx {

  namespace linalg_internal {

    void Inv_inplace_internal_d(boost::intrusive_ptr<Storage_base> &iten, const cytnx_int32 &L) {
      cytnx_int32 *ipiv = (cytnx_int32 *)malloc((L + 1) * sizeof(cytnx_int32));
      cytnx_int32 info;
      dgetrf(&L, &L, (cytnx_double *)iten->Mem, &L, ipiv, &info);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'dgetrf': Lapack INFO = ", info);

      cytnx_int32 lwork = -1;
      cytnx_double worktest = 0.;
      dgetri(&L, (cytnx_double *)iten->Mem, &L, ipiv, &worktest, &lwork, &info);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'dgetri': Lapack INFO = ", info);

      lwork = (cytnx_int32)worktest;
      std::cout << lwork << std::endl;
      cytnx_double *work = (cytnx_double *)malloc(lwork * sizeof(cytnx_double));
      dgetri(&L, (cytnx_double *)iten->Mem, &L, ipiv, work, &lwork, &info);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'dgetri': Lapack INFO = ", info);

      free(ipiv);
      free(work);
    }

    void Inv_inplace_internal_f(boost::intrusive_ptr<Storage_base> &iten, const cytnx_int32 &L) {
      cytnx_int32 *ipiv = (cytnx_int32 *)malloc((L + 1) * sizeof(cytnx_int32));
      cytnx_int32 info;
      sgetrf(&L, &L, (cytnx_float *)iten->Mem, &L, ipiv, &info);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'sgetrf': Lapack INFO = ", info);

      cytnx_int32 lwork = -1;
      cytnx_float worktest = 0.;
      sgetri(&L, (cytnx_float *)iten->Mem, &L, ipiv, &worktest, &lwork, &info);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'sgetri': Lapack INFO = ", info);

      lwork = (cytnx_int32)worktest;
      cytnx_float *work = (cytnx_float *)malloc(lwork * sizeof(cytnx_float));
      sgetri(&L, (cytnx_float *)iten->Mem, &L, ipiv, work, &lwork, &info);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'sgetri': Lapack INFO = ", info);

      free(ipiv);
      free(work);
    }

    void Inv_inplace_internal_cd(boost::intrusive_ptr<Storage_base> &iten, const cytnx_int32 &L) {
      cytnx_int32 *ipiv = (cytnx_int32 *)malloc((L + 1) * sizeof(cytnx_int32));
      cytnx_int32 info;
      zgetrf(&L, &L, (cytnx_complex128 *)iten->Mem, &L, ipiv, &info);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'zgetrf': Lapack INFO = ", info);

      cytnx_int32 lwork = -1;
      cytnx_complex128 worktest = 0.;
      zgetri(&L, (cytnx_complex128 *)iten->Mem, &L, ipiv, &worktest, &lwork, &info);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'zgetri': Lapack INFO = ", info);

      lwork = (cytnx_int32)(worktest.real());
      cytnx_complex128 *work = (cytnx_complex128 *)malloc(lwork * sizeof(cytnx_complex128));
      zgetri(&L, (cytnx_complex128 *)iten->Mem, &L, ipiv, work, &lwork, &info);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'zgetri': Lapack INFO = ", info);

      free(ipiv);
      free(work);
    }

    void Inv_inplace_internal_cf(boost::intrusive_ptr<Storage_base> &iten, const cytnx_int32 &L) {
      cytnx_int32 *ipiv = (cytnx_int32 *)malloc((L + 1) * sizeof(cytnx_int32));
      cytnx_int32 info;
      cgetrf(&L, &L, (cytnx_complex64 *)iten->Mem, &L, ipiv, &info);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'cgetrf': Lapack INFO = ", info);

      cytnx_int32 lwork = -1;
      cytnx_complex64 worktest = 0.;
      cgetri(&L, (cytnx_complex64 *)iten->Mem, &L, ipiv, &worktest, &lwork, &info);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'cgetri': Lapack INFO = ", info);

      lwork = (cytnx_int32)(worktest.real());
      cytnx_complex64 *work = (cytnx_complex64 *)malloc(lwork * sizeof(cytnx_complex64));
      cgetri(&L, (cytnx_complex64 *)iten->Mem, &L, ipiv, work, &lwork, &info);

      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'cgetri': Lapack INFO = ", info);

      free(ipiv);
      free(work);
    }

  }  // namespace linalg_internal

}  // namespace cytnx
