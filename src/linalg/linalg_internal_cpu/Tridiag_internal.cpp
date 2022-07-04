#include "Tridiag_internal.hpp"
#include "cytnx_error.hpp"
#include "lapack_wrapper.hpp"
#include <iostream>
namespace cytnx {

  namespace linalg_internal {

    void Tridiag_internal_d(const boost::intrusive_ptr<Storage_base> &diag,
                            const boost::intrusive_ptr<Storage_base> &s_diag,
                            boost::intrusive_ptr<Storage_base> &S,
                            boost::intrusive_ptr<Storage_base> &U, const cytnx_int64 &L) {
      char job;
      job = (U->dtype == Type.Void) ? 'N' : 'V';
      // std::cout << L << std::endl;
      // copy from in to S[out]
      memcpy(S->Mem, diag->Mem, L * sizeof(cytnx_double));

      // create tmp for sub-diag and cpy in:
      cytnx_double *Dsv = (cytnx_double *)malloc((L - 1) * sizeof(cytnx_double));
      memcpy(Dsv, s_diag->Mem, (L - 1) * sizeof(cytnx_double));

      lapack_int ldz = 1;
      lapack_int info;

      // check if compute eigV
      if (U->dtype != Type.Void) {
        ldz = L;
      }

      info = LAPACKE_dstev(LAPACK_COL_MAJOR, job, L, (cytnx_double *)S->Mem, Dsv,
                           (cytnx_double *)U->Mem, ldz);
      // std::cout << L << std::endl;
      cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'dstev': Lapack INFO = ", info);

      // house keeping
      free(Dsv);
    }
    void Tridiag_internal_f(const boost::intrusive_ptr<Storage_base> &diag,
                            const boost::intrusive_ptr<Storage_base> &s_diag,
                            boost::intrusive_ptr<Storage_base> &S,
                            boost::intrusive_ptr<Storage_base> &U, const cytnx_int64 &L) {
      char job;
      job = (U->dtype == Type.Void) ? 'N' : 'V';
      // std::cout << L << std::endl;
      // copy from in to S[out]
      memcpy(S->Mem, diag->Mem, L * sizeof(cytnx_float));

      // create tmp for sub-diag and cpy in:
      cytnx_float *Dsv = (cytnx_float *)malloc((L - 1) * sizeof(cytnx_float));
      memcpy(Dsv, s_diag->Mem, (L - 1) * sizeof(cytnx_float));

      lapack_int ldz = 1;
      lapack_int info;

      // check if compute eigV
      if (U->dtype != Type.Void) {
        ldz = L;
      }

      info = LAPACKE_sstev(LAPACK_COL_MAJOR, job, L, (cytnx_float *)S->Mem, Dsv,
                           (cytnx_float *)U->Mem, ldz);
      cytnx_error_msg(info != 0, "%s %d", "Error in Lapack function 'sstev': Lapack INFO = ", info);

      // house keeping
      free(Dsv);
    }
  }  // namespace linalg_internal
}  // namespace cytnx
