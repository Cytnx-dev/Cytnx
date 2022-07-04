#include "QR_internal.hpp"
#include "cytnx_error.hpp"
#include "lapack_wrapper.hpp"

namespace cytnx {

  namespace linalg_internal {

    template <class T>
    void GetUpTri(T *out, const T *elem, const cytnx_uint64 &M, const cytnx_uint64 &N) {
      cytnx_uint64 min = M < N ? M : N;
      for (cytnx_uint64 i = 0; i < min; i++) {
        memcpy(out + i * N + i, elem + i * N + i, (N - i) * sizeof(T));
      }
    }

    template <class T>
    void GetDiag(T *out, const T *elem, const cytnx_uint64 &M, const cytnx_uint64 &N,
                 const cytnx_uint64 &diag_N) {
      cytnx_uint64 min = M < N ? M : N;
      min = min < diag_N ? min : diag_N;

#ifdef UNI_OMP
  #pragma omp parallel for schedule(dynamic)
#endif
      for (cytnx_uint64 i = 0; i < min; i++) out[i] = elem[i * N + i];
    }

    /// QR
    void QR_internal_cd(const boost::intrusive_ptr<Storage_base> &in,
                        boost::intrusive_ptr<Storage_base> &Q,
                        boost::intrusive_ptr<Storage_base> &R,
                        boost::intrusive_ptr<Storage_base> &D,
                        boost::intrusive_ptr<Storage_base> &tau, const cytnx_int64 &M,
                        const cytnx_int64 &N, const bool &is_d) {
      // Q should be the same shape as in
      // tau should be the min(M,N)

      cytnx_complex128 *pQ = (cytnx_complex128 *)Q->Mem;
      cytnx_complex128 *pR = (cytnx_complex128 *)R->Mem;
      cytnx_complex128 *ptau = (cytnx_complex128 *)tau->Mem;

      // cytnx_complex128* Mij = (cytnx_complex128*)malloc(M * N * sizeof(cytnx_complex128));
      memcpy(pQ, in->Mem, M * N * sizeof(cytnx_complex128));

      lapack_int ldA = N;
      lapack_int info;
      lapack_int K = N;

      // call linalg:
      info = LAPACKE_zgelqf(LAPACK_COL_MAJOR, N, M, (lapack_complex_double *)pQ, ldA,
                            (lapack_complex_double *)ptau);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'zgelqf': Lapack INFO = ", info);

      // getR:
      GetUpTri(pR, pQ, M, N);

      // getD:
      if (is_d) {
        cytnx_complex128 *pD = (cytnx_complex128 *)D->Mem;
        GetDiag(pD, pR, M, N, N);
        cytnx_uint64 min = M < N ? M : N;
        // normalize:
        for (cytnx_uint64 i = 0; i < min; i++) {
          for (cytnx_uint64 j = 0; j < N - i; j++) {
            pR[i * N + i + j] /= pD[i];
          }
        }
      }

      // getQ:
      // query lwork & alloc
      lapack_int col = M < N ? N : M;

      // call linalg:
      info = LAPACKE_zunglq(LAPACK_COL_MAJOR, N, col, K, (lapack_complex_double *)pQ, ldA,
                            (lapack_complex_double *)ptau);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'zunglq': Lapack INFO = ", info);
    }
    void QR_internal_cf(const boost::intrusive_ptr<Storage_base> &in,
                        boost::intrusive_ptr<Storage_base> &Q,
                        boost::intrusive_ptr<Storage_base> &R,
                        boost::intrusive_ptr<Storage_base> &D,
                        boost::intrusive_ptr<Storage_base> &tau, const cytnx_int64 &M,
                        const cytnx_int64 &N, const bool &is_d) {
      // Q should be the same shape as in
      // tau should be the min(M,N)

      cytnx_complex64 *pQ = (cytnx_complex64 *)Q->Mem;
      cytnx_complex64 *pR = (cytnx_complex64 *)R->Mem;
      cytnx_complex64 *ptau = (cytnx_complex64 *)tau->Mem;

      // cytnx_complex128* Mij = (cytnx_complex128*)malloc(M * N * sizeof(cytnx_complex128));
      memcpy(pQ, in->Mem, M * N * sizeof(cytnx_complex64));

      lapack_int ldA = N;
      lapack_int info;
      lapack_int K = N;

      // call linalg:
      info = LAPACKE_cgelqf(LAPACK_COL_MAJOR, N, M, (lapack_complex_float *)pQ, ldA,
                            (lapack_complex_float *)ptau);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'cgelqf': Lapack INFO = ", info);

      // getR:
      GetUpTri(pR, pQ, M, N);

      // getD:
      if (is_d) {
        cytnx_complex64 *pD = (cytnx_complex64 *)D->Mem;
        GetDiag(pD, pR, M, N, N);
        cytnx_uint64 min = M < N ? M : N;
        // normalize:
        for (cytnx_uint64 i = 0; i < min; i++) {
          for (cytnx_uint64 j = 0; j < N - i; j++) {
            pR[i * N + i + j] /= pD[i];
          }
        }
      }

      // getQ:
      // query lwork & alloc
      lapack_int col = M < N ? N : M;

      // call linalg:
      info = LAPACKE_cunglq(LAPACK_COL_MAJOR, N, col, K, (lapack_complex_float *)pQ, ldA,
                            (lapack_complex_float *)ptau);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'cunglq': Lapack INFO = ", info);
    }
    void QR_internal_d(const boost::intrusive_ptr<Storage_base> &in,
                       boost::intrusive_ptr<Storage_base> &Q, boost::intrusive_ptr<Storage_base> &R,
                       boost::intrusive_ptr<Storage_base> &D,
                       boost::intrusive_ptr<Storage_base> &tau, const cytnx_int64 &M,
                       const cytnx_int64 &N, const bool &is_d) {
      // Q should be the same shape as in
      // tau should be the min(M,N)

      cytnx_double *pQ = (cytnx_double *)Q->Mem;
      cytnx_double *pR = (cytnx_double *)R->Mem;
      cytnx_double *ptau = (cytnx_double *)tau->Mem;

      memcpy(pQ, in->Mem, M * N * sizeof(cytnx_double));

      lapack_int ldA = N;
      lapack_int info;
      lapack_int K = N;

      // call linalg:
      info = LAPACKE_dgelqf(LAPACK_COL_MAJOR, N, M, pQ, ldA, ptau);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'dgelqf': Lapack INFO = ", info);

      // getR:
      GetUpTri(pR, pQ, M, N);

      // getD:
      if (is_d) {
        cytnx_double *pD = (cytnx_double *)D->Mem;
        GetDiag(pD, pR, M, N, N);
        cytnx_uint64 min = M < N ? M : N;
        // normalize:
        for (cytnx_uint64 i = 0; i < min; i++) {
          for (cytnx_uint64 j = 0; j < N - i; j++) {
            pR[i * N + i + j] /= pD[i];
          }
        }
      }

      // getQ:
      // query lwork & alloc
      lapack_int col = M < N ? N : M;
      info = LAPACKE_dorglq(LAPACK_COL_MAJOR, N, col, K, pQ, ldA, ptau);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'dorglq': Lapack INFO = ", info);
    }
    void QR_internal_f(const boost::intrusive_ptr<Storage_base> &in,
                       boost::intrusive_ptr<Storage_base> &Q, boost::intrusive_ptr<Storage_base> &R,
                       boost::intrusive_ptr<Storage_base> &D,
                       boost::intrusive_ptr<Storage_base> &tau, const cytnx_int64 &M,
                       const cytnx_int64 &N, const bool &is_d) {
      // Q should be the same shape as in
      // tau should be the min(M,N)

      cytnx_float *pQ = (cytnx_float *)Q->Mem;
      cytnx_float *pR = (cytnx_float *)R->Mem;
      cytnx_float *ptau = (cytnx_float *)tau->Mem;

      // cytnx_complex128* Mij = (cytnx_complex128*)malloc(M * N * sizeof(cytnx_complex128));
      memcpy(pQ, in->Mem, M * N * sizeof(cytnx_float));

      lapack_int ldA = N;
      lapack_int info;
      lapack_int K = N;

      // call linalg:
      info = LAPACKE_sgelqf(LAPACK_COL_MAJOR, N, M, pQ, ldA, ptau);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'sgelqf': Lapack INFO = ", info);

      // getR:
      GetUpTri(pR, pQ, M, N);

      // getD:
      if (is_d) {
        cytnx_float *pD = (cytnx_float *)D->Mem;
        GetDiag(pD, pR, M, N, N);
        cytnx_uint64 min = M < N ? M : N;
        // normalize:
        for (cytnx_uint64 i = 0; i < min; i++) {
          for (cytnx_uint64 j = 0; j < N - i; j++) {
            pR[i * N + i + j] /= pD[i];
          }
        }
      }

      // getQ:
      // query lwork & alloc
      lapack_int col = M < N ? N : M;
      info = LAPACKE_sorglq(LAPACK_COL_MAJOR, N, col, K, pQ, ldA, ptau);
      cytnx_error_msg(info != 0, "%s %d",
                      "Error in Lapack function 'sorglq': Lapack INFO = ", info);
    }

  }  // namespace linalg_internal
}  // namespace cytnx
