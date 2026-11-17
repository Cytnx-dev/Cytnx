#include "Tridiag_internal.hpp"
#include "cytnx_error.hpp"
#include "backend/lapack_wrapper.hpp"
#include <iostream>
#include <vector>
namespace cytnx {

  namespace linalg_internal {

    void Tridiag_internal_d(const boost::intrusive_ptr<Storage_base> &diag,
                            const boost::intrusive_ptr<Storage_base> &s_diag,
                            boost::intrusive_ptr<Storage_base> &S,
                            boost::intrusive_ptr<Storage_base> &U, const cytnx_int64 &L,
                            bool throw_excp /*= false*/) {
      char job;
      job = (U->dtype() == Type.Void) ? 'N' : 'V';
      // copy from in to S[out]
      memcpy(S->data(), diag->data(), L * sizeof(cytnx_double));

      // LAPACK overwrites the sub-diagonal, so work on a copy:
      const cytnx_double *sub_diag = static_cast<const cytnx_double *>(s_diag->data());
      std::vector<cytnx_double> Dsv(sub_diag, sub_diag + (L - 1));

      lapack_int ldz = 1;
      lapack_int info;

      // check if compute eigV
      if (U->dtype() != Type.Void) {
        ldz = L;
      }

      info = LAPACKE_dstev(LAPACK_COL_MAJOR, job, L, (cytnx_double *)S->data(), Dsv.data(),
                           (cytnx_double *)U->data(), ldz);
      cytnx_error_msg(info != 0, "Error in Lapack function 'dstev': Lapack INFO = %d\n", info);
    }
    void Tridiag_internal_f(const boost::intrusive_ptr<Storage_base> &diag,
                            const boost::intrusive_ptr<Storage_base> &s_diag,
                            boost::intrusive_ptr<Storage_base> &S,
                            boost::intrusive_ptr<Storage_base> &U, const cytnx_int64 &L,
                            bool throw_excp /*= false*/) {
      char job;
      job = (U->dtype() == Type.Void) ? 'N' : 'V';
      // copy from in to S[out]
      memcpy(S->data(), diag->data(), L * sizeof(cytnx_float));

      // LAPACK overwrites the sub-diagonal, so work on a copy:
      const cytnx_float *sub_diag = static_cast<const cytnx_float *>(s_diag->data());
      std::vector<cytnx_float> Dsv(sub_diag, sub_diag + (L - 1));

      lapack_int ldz = 1;
      lapack_int info;

      // check if compute eigV
      if (U->dtype() != Type.Void) {
        ldz = L;
      }

      info = LAPACKE_sstev(LAPACK_COL_MAJOR, job, L, (cytnx_float *)S->data(), Dsv.data(),
                           (cytnx_float *)U->data(), ldz);
      cytnx_error_msg(info != 0, "Error in Lapack function 'sstev': Lapack INFO = %d\n", info);
    }
  }  // namespace linalg_internal
}  // namespace cytnx
