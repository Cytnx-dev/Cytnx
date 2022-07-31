#ifndef __LAPACK_WRAPPER_H__
#define __LAPACK_WRAPPER_H__

#include <stdint.h>

#include <complex>
#include <complex.h>
#include <algorithm>
#include "Type.hpp"

#ifdef UNI_MKL
  #include <mkl.h>

#else
extern "C" {

// BLAS functions
void dgemm_(const char* transa, const char* transb, const int32_t* m, const int32_t* n,
            const int32_t* k, const double* alpha, const double* a, const int32_t* lda,
            const double* b, const int32_t* ldb, const double* beta, double* c, const int32_t* ldc);
void sgemm_(const char* transa, const char* transb, const int32_t* m, const int32_t* n,
            const int32_t* k, const float* alpha, const float* a, const int32_t* lda,
            const float* b, const int32_t* ldb, const float* beta, float* c, const int32_t* ldc);
void zgemm_(const char* transa, const char* transb, const int32_t* m, const int32_t* n,
            const int32_t* k, const std::complex<double>* alpha, const std::complex<double>* a,
            const int32_t* lda, const std::complex<double>* b, const int32_t* ldb,
            const std::complex<double>* beta, std::complex<double>* c, const int32_t* ldc);
void cgemm_(const char* transa, const char* transb, const int32_t* m, const int32_t* n,
            const int32_t* k, const std::complex<float>* alpha, const std::complex<float>* a,
            const int32_t* lda, const std::complex<float>* b, const int32_t* ldb,
            const std::complex<float>* beta, std::complex<float>* c, const int32_t* ldc);

// void   zgemm_(const char *transa, const char *transb, const int32_t *m, const int32_t *n, const
// int32_t *k,
//               const double _Complex *alpha, const double _Complex *a, const int32_t *lda, const
//               double _Complex *b, const int32_t *ldb, const double _Complex *beta, double
//               _Complex *c, const int32_t *ldc);

double dasum_(const int32_t* n, const double* x, const int32_t* incx);

void scopy_(const int32_t* n, const float* x, const int32_t* incx, float* y, const int32_t* incy);
void dcopy_(const int32_t* n, const double* x, const int32_t* incx, double* y, const int32_t* incy);
// void   zcopy_(const int32_t *n, const double *x, const int32_t *incx, std::complex<double> *y,
// const int32_t *incy);

void daxpy_(const int32_t* n, const double* alpha, const double* x, const int32_t* incx, double* y,
            const int32_t* incy);
void zaxpy_(const int32_t* n, const std::complex<double>* alpha, const std::complex<double>* x,
            const int32_t* incx, std::complex<double>* y, const int32_t* incy);

void dscal_(const int32_t* n, const double* a, double* x, const int32_t* incx);
void zscal_(const int32_t* n, const std::complex<double>* a, std::complex<double>* x,
            const int32_t* incx);
void zdscal_(const int32_t* n, const double* a, std::complex<double>* x, const int32_t* incx);

double dnrm2_(const int32_t* n, const double* x, const int32_t* incx);
double dznrm2_(const int32_t* n, const std::complex<double>* x, const int32_t* incx);

void dgemv_(const char* trans, const int32_t* m, const int32_t* n, const double* alpha,
            const double* a, const int32_t* lda, const double* x, const int32_t* incx,
            const double* beta, const double* y, const int32_t* incy);
void zgemv_(const char* trans, const int32_t* m, const int32_t* n,
            const std::complex<double>* alpha, const std::complex<double>* a, const int32_t* lda,
            const std::complex<double>* x, const int32_t* incx, const std::complex<double>* beta,
            const std::complex<double>* y, const int32_t* incy);

double ddot_(const int32_t* n, const double* x, const int32_t* incx, const double* y,
             const int32_t* incy);
void zdotc_(std::complex<double>* res, const int32_t* n, const std::complex<double>* x,
            const int32_t* incx, const std::complex<double>* y, const int32_t* incy);

// D and Z LAPACK functions

void dgeqp3_(const int32_t* m, const int32_t* n, double* a, const int32_t* lda, int* jpvt,
             double* tau, double* work, int* lwork, int32_t* info);
void zgeqp3_(const int32_t* m, const int32_t* n, std::complex<double>* a, const int32_t* lda,
             int* jpvt, std::complex<double>* tau, std::complex<double>* work, int* lwork,
             double* rwork, int32_t* info);
/*
void dgeqpf_( const int32_t* m, const int32_t* n, double* a, const int32_t* lda,
              int* jpvt, double* tau, double* work, int32_t* info );
void zgeqpf_( const int32_t* m, const int32_t* n, std::complex<double>* a, const int32_t* lda,
              int* jpvt, std::complex<double>* tau, std::complex<double>* work,
              double* rwork, int32_t* info );
*/

void dgesvd_(const char* jobu, const char* jobvt, const int32_t* m, const int32_t* n, double* a,
             const int32_t* lda, double* s, double* u, const int32_t* ldu, double* vt,
             const int32_t* ldvt, double* work, const int32_t* lwork, int32_t* info);

void sgesvd_(const char* jobu, const char* jobvt, const int32_t* m, const int32_t* n, float* a,
             const int32_t* lda, float* s, float* u, const int32_t* ldu, float* vt,
             const int32_t* ldvt, float* work, const int32_t* lwork, int32_t* info);

void zgesvd_(const char* jobu, const char* jobvt, const int32_t* m, const int32_t* n,
             std::complex<double>* a, const int32_t* lda, double* s, std::complex<double>* u,
             const int32_t* ldu, std::complex<double>* vt, const int32_t* ldvt,
             std::complex<double>* work, const int32_t* lwork, double* rwork, int32_t* info);

void cgesvd_(const char* jobu, const char* jobvt, const int32_t* m, const int32_t* n,
             std::complex<float>* a, const int32_t* lda, float* s, std::complex<float>* u,
             const int32_t* ldu, std::complex<float>* vt, const int32_t* ldvt,
             std::complex<float>* work, const int32_t* lwork, float* rwork, int32_t* info);

void dgesdd_(const char* jobz, const int32_t* m, const int32_t* n, double* a, const int32_t* lda,
             double* s, double* u, const int32_t* ldu, double* vt, const int32_t* ldvt,
             double* work, const int32_t* lwork, int32_t* iwork, int32_t* info);
void zgesdd_(const char* jobz, const int32_t* m, const int32_t* n, std::complex<double>* a,
             const int32_t* lda, double* s, std::complex<double>* u, const int32_t* ldu,
             std::complex<double>* vt, const int32_t* ldvt, std::complex<double>* work,
             const int32_t* lwork, double* rwork, int32_t* iwork, int32_t* info);

void ssyev_(const char* jobz, const char* uplo, const int32_t* n, float* a, const int32_t* lda,
            float* w, float* work, const int32_t* lwork, int32_t* info);
void dsyev_(const char* jobz, const char* uplo, const int32_t* n, double* a, const int32_t* lda,
            double* w, double* work, const int32_t* lwork, int32_t* info);
void cheev_(const char* jobz, const char* uplo, const int32_t* n, std::complex<float>* a,
            const int32_t* lda, float* w, std::complex<float>* work, const int32_t* lwork,
            const float* rwork, int32_t* info);

void zheev_(const char* jobz, const char* uplo, const int32_t* n, std::complex<double>* a,
            const int32_t* lda, double* w, std::complex<double>* work, const int32_t* lwork,
            const double* rwork, int32_t* info);
void zgeev_(const char* jobvl, const char* jobvr, const int32_t* n, const std::complex<double>* a,
            const int32_t* lda, const std::complex<double>* w, const std::complex<double>* vl,
            const int32_t* ldvl, const std::complex<double>* vr, const int32_t* ldvr,
            const std::complex<double>* work, const int32_t* lwork, const double* rwork,
            int32_t* info);
void cgeev_(const char* jobvl, const char* jobvr, const int32_t* n, const std::complex<float>* a,
            const int32_t* lda, const std::complex<float>* w, const std::complex<float>* vl,
            const int32_t* ldvl, const std::complex<float>* vr, const int32_t* ldvr,
            const std::complex<float>* work, const int32_t* lwork, const float* rwork,
            int32_t* info);

void dstev_(const char* jobz, const int32_t* n, const double* d, const double* e, const double* z,
            const int32_t* ldaz, const double* work, int32_t* info);

void dgetrf_(const int32_t* m, const int32_t* n, const double* a, const int32_t* lda,
             const int32_t* ipiv, int32_t* info);
void sgetrf_(const int32_t* m, const int32_t* n, const float* a, const int32_t* lda,
             const int32_t* ipiv, int32_t* info);

void zgetrf_(const int32_t* m, const int32_t* n, const std::complex<double>* a, const int32_t* lda,
             const int32_t* ipiv, int32_t* info);
void cgetrf_(const int32_t* m, const int32_t* n, const std::complex<float>* a, const int32_t* lda,
             const int32_t* ipiv, int32_t* info);

void dgetri_(const int32_t* n, const double* a, const int32_t* lda, const int32_t* ipiv,
             const double* work, const int32_t* lwork, int32_t* info);
void sgetri_(const int32_t* n, const float* a, const int32_t* lda, const int32_t* ipiv,
             const float* work, const int32_t* lwork, int32_t* info);
void cgetri_(const int32_t* n, const std::complex<float>* a, const int32_t* lda,
             const int32_t* ipiv, const std::complex<float>* work, const int32_t* lwork,
             int32_t* info);
void zgetri_(const int32_t* n, const std::complex<double>* a, const int32_t* lda,
             const int32_t* ipiv, const std::complex<double>* work, const int32_t* lwork,
             int32_t* info);
void sgelqf_(const int32_t* m, const int32_t* n, float* a, const int32_t* lda, float* tau,
             float* work, const int32_t* lwork, int32_t* info);

void dgelqf_(const int32_t* m, const int32_t* n, double* a, const int32_t* lda, double* tau,
             double* work, const int32_t* lwork, int32_t* info);
void zgelqf_(const int32_t* m, const int32_t* n, std::complex<double>* a, const int32_t* lda,
             std::complex<double>* tau, std::complex<double>* work, const int32_t* lwork,
             int32_t* info);
void cgelqf_(const int32_t* m, const int32_t* n, std::complex<float>* a, const int32_t* lda,
             std::complex<float>* tau, std::complex<float>* work, const int32_t* lwork,
             int32_t* info);

void dorglq_(const int32_t* m, const int32_t* n, const int32_t* k, double* a, const int32_t* lda,
             const double* tau, double* work, const int32_t* lwork, int32_t* info);
void sorglq_(const int32_t* m, const int32_t* n, const int32_t* k, float* a, const int32_t* lda,
             const float* tau, float* work, const int32_t* lwork, int32_t* info);
void zunglq_(const int32_t* m, const int32_t* n, const int32_t* k, std::complex<double>* a,
             const int32_t* lda, const std::complex<double>* tau, std::complex<double>* work,
             const int32_t* lwork, int32_t* info);
void cunglq_(const int32_t* m, const int32_t* n, const int32_t* k, std::complex<float>* a,
             const int32_t* lda, const std::complex<float>* tau, std::complex<float>* work,
             const int32_t* lwork, int32_t* info);
void dgeqlf_(const int32_t* m, const int32_t* n, double* a, const int32_t* lda, double* tau,
             double* work, const int32_t* lwork, int32_t* info);
void zgeqlf_(const int32_t* m, const int32_t* n, std::complex<double>* a, const int32_t* lda,
             std::complex<double>* tau, std::complex<double>* work, const int32_t* lwork,
             int32_t* info);

void dorgql_(const int32_t* m, const int32_t* n, const int32_t* k, double* a, const int32_t* lda,
             const double* tau, double* work, const int32_t* lwork, int32_t* info);
void zungql_(const int32_t* m, const int32_t* n, const int32_t* k, std::complex<double>* a,
             const int32_t* lda, const std::complex<double>* tau, std::complex<double>* work,
             const int32_t* lwork, int32_t* info);

void dgeqrf_(const int32_t* m, const int32_t* n, double* a, const int32_t* lda, double* tau,
             double* work, const int32_t* lwork, int32_t* info);
void sgeqrf_(const int32_t* m, const int32_t* n, float* a, const int32_t* lda, float* tau,
             float* work, const int32_t* lwork, int32_t* info);
void zgeqrf_(const int32_t* m, const int32_t* n, std::complex<double>* a, const int32_t* lda,
             std::complex<double>* tau, std::complex<double>* work, const int32_t* lwork,
             int32_t* info);
void cgeqrf_(const int32_t* m, const int32_t* n, std::complex<float>* a, const int32_t* lda,
             std::complex<float>* tau, std::complex<float>* work, const int32_t* lwork,
             int32_t* info);
void dorgqr_(const int32_t* m, const int32_t* n, const int32_t* k, double* a, const int32_t* lda,
             const double* tau, double* work, const int32_t* lwork, int32_t* info);
void sorgqr_(const int32_t* m, const int32_t* n, const int32_t* k, float* a, const int32_t* lda,
             const float* tau, float* work, const int32_t* lwork, int32_t* info);

void zungqr_(const int32_t* m, const int32_t* n, const int32_t* k, std::complex<double>* a,
             const int32_t* lda, const std::complex<double>* tau, std::complex<double>* work,
             const int32_t* lwork, int32_t* info);
void cungqr_(const int32_t* m, const int32_t* n, const int32_t* k, std::complex<float>* a,
             const int32_t* lda, const std::complex<float>* tau, std::complex<float>* work,
             const int32_t* lwork, int32_t* info);
void dgerqf_(const int32_t* m, const int32_t* n, double* a, const int32_t* lda, double* tau,
             double* work, const int32_t* lwork, int32_t* info);
void zgerqf_(const int32_t* m, const int32_t* n, std::complex<double>* a, const int32_t* lda,
             std::complex<double>* tau, std::complex<double>* work, const int32_t* lwork,
             int32_t* info);

void dorgrq_(const int32_t* m, const int32_t* n, const int32_t* k, double* a, const int32_t* lda,
             const double* tau, double* work, const int32_t* lwork, int32_t* info);
void sorgrq_(const int32_t* m, const int32_t* n, const int32_t* k, float* a, const int32_t* lda,
             const float* tau, float* work, const int32_t* lwork, int32_t* info);

void zungrq_(const int32_t* m, const int32_t* n, const int32_t* k, std::complex<double>* a,
             const int32_t* lda, const std::complex<double>* tau, std::complex<double>* work,
             const int32_t* lwork, int32_t* info);

}  // extern C

// Wrappers for BLAS and LAPACK functions
inline void dgemm(const char* transa, const char* transb, const int32_t* m, const int32_t* n,
                  const int32_t* k, const double* alpha, const double* a, const int32_t* lda,
                  const double* b, const int32_t* ldb, const double* beta, double* c,
                  const int32_t* ldc) {
  dgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline void sgemm(const char* transa, const char* transb, const int32_t* m, const int32_t* n,
                  const int32_t* k, const float* alpha, const float* a, const int32_t* lda,
                  const float* b, const int32_t* ldb, const float* beta, float* c,
                  const int32_t* ldc) {
  sgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void zgemm(const char* transa, const char* transb, const int32_t* m, const int32_t* n,
                  const int32_t* k, const std::complex<double>* alpha,
                  const std::complex<double>* a, const int32_t* lda, const std::complex<double>* b,
                  const int32_t* ldb, const std::complex<double>* beta, std::complex<double>* c,
                  const int32_t* ldc) {
  zgemm_(transa, transb, m, n, k, (const std::complex<double>*)alpha,
         (const std::complex<double>*)a, lda, (const std::complex<double>*)b, ldb,
         (const std::complex<double>*)beta, (std::complex<double>*)c, ldc);
}

inline void cgemm(const char* transa, const char* transb, const int32_t* m, const int32_t* n,
                  const int32_t* k, const std::complex<float>* alpha, const std::complex<float>* a,
                  const int32_t* lda, const std::complex<float>* b, const int32_t* ldb,
                  const std::complex<float>* beta, std::complex<float>* c, const int32_t* ldc) {
  cgemm_(transa, transb, m, n, k, (const std::complex<float>*)alpha, (const std::complex<float>*)a,
         lda, (const std::complex<float>*)b, ldb, (const std::complex<float>*)beta,
         (std::complex<float>*)c, ldc);
}

inline double dasum(const int32_t* n, const double* x, const int32_t* incx) {
  return dasum_(n, x, incx);
}

inline void dcopy(const int32_t& n, const double* x, const int32_t& incx, double* y,
                  const int32_t& incy) {
  dcopy_(&n, x, &incx, y, &incy);
}
inline void scopy(const int32_t& n, const float* x, const int32_t& incx, float* y,
                  const int32_t& incy) {
  scopy_(&n, x, &incx, y, &incy);
}

inline void daxpy(const int32_t* n, const double* alpha, const double* x, const int32_t* incx,
                  double* y, const int32_t* incy) {
  daxpy_(n, alpha, x, incx, y, incy);
}

inline void zaxpy(const int32_t* n, const std::complex<double>* alpha,
                  const std::complex<double>* x, const int32_t* incx, std::complex<double>* y,
                  const int32_t* incy) {
  zaxpy_(n, alpha, x, incx, y, incy);
}

inline double dnrm2(const int32_t* n, const double* x, const int32_t* incx) {
  return dnrm2_(n, x, incx);
}

inline double dznrm2(const int32_t* n, const std::complex<double>* x, const int32_t* incx) {
  return dznrm2_(n, x, incx);
}

inline void dscal(const int32_t* n, const double* a, double* x, const int32_t* incx) {
  dscal_(n, a, x, incx);
}
inline void zscal(const int32_t* n, const std::complex<double>* a, std::complex<double>* x,
                  const int32_t* incx) {
  zscal_(n, a, x, incx);
}

inline void zdscal(const int32_t* n, const double* a, std::complex<double>* x,
                   const int32_t* incx) {
  zdscal_(n, a, x, incx);
}
inline void dsyev(const char* jobz, const char* uplo, const int32_t* n, double* a,
                  const int32_t* lda, double* w, double* work, const int32_t* lwork,
                  int32_t* info) {
  dsyev_(jobz, uplo, n, a, lda, w, work, lwork, info);
}
inline void ssyev(const char* jobz, const char* uplo, const int32_t* n, float* a,
                  const int32_t* lda, float* w, float* work, const int32_t* lwork, int32_t* info) {
  ssyev_(jobz, uplo, n, a, lda, w, work, lwork, info);
}

inline void cheev(const char* jobz, const char* uplo, const int32_t* n, std::complex<float>* a,
                  const int32_t* lda, float* w, std::complex<float>* work, const int32_t* lwork,
                  const float* rwork, int32_t* info) {
  cheev_(jobz, uplo, n, a, lda, w, work, lwork, rwork, info);
}

inline void zheev(const char* jobz, const char* uplo, const int32_t* n, std::complex<double>* a,
                  const int32_t* lda, double* w, std::complex<double>* work, const int32_t* lwork,
                  const double* rwork, int32_t* info) {
  zheev_(jobz, uplo, n, a, lda, w, work, lwork, rwork, info);
}

inline void zgeev(const char* jobvl, const char* jobvr, const int32_t* n,
                  const std::complex<double>* a, const int32_t* lda, const std::complex<double>* w,
                  const std::complex<double>* vl, const int32_t* ldvl,
                  const std::complex<double>* vr, const int32_t* ldvr,
                  const std::complex<double>* work, const int32_t* lwork, const double* rwork,
                  int32_t* info) {
  zgeev_(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info);
}

inline void cgeev(const char* jobvl, const char* jobvr, const int32_t* n,
                  const std::complex<float>* a, const int32_t* lda, const std::complex<float>* w,
                  const std::complex<float>* vl, const int32_t* ldvl, const std::complex<float>* vr,
                  const int32_t* ldvr, const std::complex<float>* work, const int32_t* lwork,
                  const float* rwork, int32_t* info) {
  cgeev_(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info);
}

inline void dgesvd(const char* jobu, const char* jobvt, const int32_t* m, const int32_t* n,
                   double* a, const int32_t* lda, double* s, double* u, const int32_t* ldu,
                   double* vt, const int32_t* ldvt, double* work, const int32_t* lwork,
                   int32_t* info) {
  dgesvd_(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
}

inline void sgesvd(const char* jobu, const char* jobvt, const int32_t* m, const int32_t* n,
                   float* a, const int32_t* lda, float* s, float* u, const int32_t* ldu, float* vt,
                   const int32_t* ldvt, float* work, const int32_t* lwork, int32_t* info) {
  sgesvd_(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
}

inline void zgesvd(const char* jobu, const char* jobvt, const int32_t* m, const int32_t* n,
                   std::complex<double>* a, const int32_t* lda, double* s, std::complex<double>* u,
                   const int32_t* ldu, std::complex<double>* vt, const int32_t* ldvt,
                   std::complex<double>* work, const int32_t* lwork, double* rwork, int32_t* info) {
  zgesvd_(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info);
}

inline void cgesvd(const char* jobu, const char* jobvt, const int32_t* m, const int32_t* n,
                   std::complex<float>* a, const int32_t* lda, float* s, std::complex<float>* u,
                   const int32_t* ldu, std::complex<float>* vt, const int32_t* ldvt,
                   std::complex<float>* work, const int32_t* lwork, float* rwork, int32_t* info) {
  cgesvd_(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info);
}

inline void dgesdd(const char* jobz, const int32_t* m, const int32_t* n, double* a,
                   const int32_t* lda, double* s, double* u, const int32_t* ldu, double* vt,
                   const int32_t* ldvt, double* work, const int32_t* lwork, int32_t* iwork,
                   int32_t* info) {
  dgesdd_(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info);
}

inline void zgesdd(const char* jobz, const int32_t* m, const int32_t* n, std::complex<double>* a,
                   const int32_t* lda, double* s, std::complex<double>* u, const int32_t* ldu,
                   std::complex<double>* vt, const int32_t* ldvt, std::complex<double>* work,
                   const int32_t* lwork, double* rwork, int32_t* iwork, int32_t* info) {
  zgesdd_(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, info);
}

inline void dgemv(const char* trans, const int32_t* m, const int32_t* n, const double* alpha,
                  const double* a, const int32_t* lda, const double* x, const int32_t* incx,
                  const double* beta, const double* y, const int32_t* incy) {
  dgemv_(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

inline void zgemv(const char* trans, const int32_t* m, const int32_t* n,
                  const std::complex<double>* alpha, const std::complex<double>* a,
                  const int32_t* lda, const std::complex<double>* x, const int32_t* incx,
                  const std::complex<double>* beta, const std::complex<double>* y,
                  const int32_t* incy) {
  zgemv_(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

inline double ddot(const int32_t* n, const double* x, const int32_t* incx, const double* y,
                   const int32_t* incy) {
  return ddot_(n, x, incx, y, incy);
}

inline void zdotc(std::complex<double>* res, const int32_t* n, const std::complex<double>* x,
                  const int32_t* incx, const std::complex<double>* y, const int32_t* incy) {
  zdotc_(res, n, x, incx, y, incy);
}

inline void dstev(const char* jobz, const int32_t* n, const double* d, const double* e,
                  const double* z, const int32_t* ldaz, const double* work, int32_t* info) {
  dstev_(jobz, n, d, e, z, ldaz, work, info);
}

inline void dgetrf(const int32_t* m, const int32_t* n, const double* a, const int32_t* lda,
                   const int32_t* ipiv, int32_t* info) {
  dgetrf_(m, n, a, lda, ipiv, info);
}
inline void sgetrf(const int32_t* m, const int32_t* n, const float* a, const int32_t* lda,
                   const int32_t* ipiv, int32_t* info) {
  sgetrf_(m, n, a, lda, ipiv, info);
}

inline void zgetrf(const int32_t* m, const int32_t* n, const std::complex<double>* a,
                   const int32_t* lda, const int32_t* ipiv, int32_t* info) {
  zgetrf_(m, n, a, lda, ipiv, info);
}
inline void cgetrf(const int32_t* m, const int32_t* n, const std::complex<float>* a,
                   const int32_t* lda, const int32_t* ipiv, int32_t* info) {
  cgetrf_(m, n, a, lda, ipiv, info);
}

inline void dgetri(const int32_t* n, const double* a, const int32_t* lda, const int32_t* ipiv,
                   const double* work, const int32_t* lwork, int32_t* info) {
  dgetri_(n, a, lda, ipiv, work, lwork, info);
}
inline void sgetri(const int32_t* n, const float* a, const int32_t* lda, const int32_t* ipiv,
                   const float* work, const int32_t* lwork, int32_t* info) {
  sgetri_(n, a, lda, ipiv, work, lwork, info);
}
inline void zgetri(const int32_t* n, const std::complex<double>* a, const int32_t* lda,
                   const int32_t* ipiv, const std::complex<double>* work, const int32_t* lwork,
                   int32_t* info) {
  zgetri_(n, a, lda, ipiv, work, lwork, info);
}
inline void cgetri(const int32_t* n, const std::complex<float>* a, const int32_t* lda,
                   const int32_t* ipiv, const std::complex<float>* work, const int32_t* lwork,
                   int32_t* info) {
  cgetri_(n, a, lda, ipiv, work, lwork, info);
}
/*
inline void dgeqpf( const int32_t* m, const int32_t* n, double* a, const int32_t* lda, int* jpvt,
double* tau, double* work, int32_t* info ){

  dgeqpf_(m, n, a, lda, jpvt, tau, work, info);

}
*/

inline void dgeqp3(const int32_t* m, const int32_t* n, double* a, const int32_t* lda, int* jpvt,
                   double* tau, double* work, int* lwork, int32_t* info) {
  dgeqp3_(m, n, a, lda, jpvt, tau, work, lwork, info);
}

inline void dgelqf(const int32_t* m, const int32_t* n, double* a, const int32_t* lda, double* tau,
                   double* work, const int32_t* lwork, int32_t* info) {
  dgelqf_(m, n, a, lda, tau, work, lwork, info);
}
inline void sgelqf(const int32_t* m, const int32_t* n, float* a, const int32_t* lda, float* tau,
                   float* work, const int32_t* lwork, int32_t* info) {
  sgelqf_(m, n, a, lda, tau, work, lwork, info);
}
inline void dorglq(const int32_t* m, const int32_t* n, const int32_t* k, double* a,
                   const int32_t* lda, const double* tau, double* work, const int32_t* lwork,
                   int32_t* info) {
  dorglq_(m, n, k, a, lda, tau, work, lwork, info);
}
inline void sorglq(const int32_t* m, const int32_t* n, const int32_t* k, float* a,
                   const int32_t* lda, const float* tau, float* work, const int32_t* lwork,
                   int32_t* info) {
  sorglq_(m, n, k, a, lda, tau, work, lwork, info);
}

inline void dgeqlf(const int32_t* m, const int32_t* n, double* a, const int32_t* lda, double* tau,
                   double* work, const int32_t* lwork, int32_t* info) {
  dgeqlf_(m, n, a, lda, tau, work, lwork, info);
}

inline void dorgql(const int32_t* m, const int32_t* n, const int32_t* k, double* a,
                   const int32_t* lda, const double* tau, double* work, const int32_t* lwork,
                   int32_t* info) {
  dorgql_(m, n, k, a, lda, tau, work, lwork, info);
}

inline void dgeqrf(const int32_t* m, const int32_t* n, double* a, const int32_t* lda, double* tau,
                   double* work, const int32_t* lwork, int32_t* info) {
  dgeqrf_(m, n, a, lda, tau, work, lwork, info);
}
inline void sgeqrf(const int32_t* m, const int32_t* n, float* a, const int32_t* lda, float* tau,
                   float* work, const int32_t* lwork, int32_t* info) {
  sgeqrf_(m, n, a, lda, tau, work, lwork, info);
}
inline void dorgqr(const int32_t* m, const int32_t* n, const int32_t* k, double* a,
                   const int32_t* lda, const double* tau, double* work, const int32_t* lwork,
                   int32_t* info) {
  dorgqr_(m, n, k, a, lda, tau, work, lwork, info);
}
inline void sorgqr(const int32_t* m, const int32_t* n, const int32_t* k, float* a,
                   const int32_t* lda, const float* tau, float* work, const int32_t* lwork,
                   int32_t* info) {
  sorgqr_(m, n, k, a, lda, tau, work, lwork, info);
}

inline void dgerqf(const int32_t* m, const int32_t* n, double* a, const int32_t* lda, double* tau,
                   double* work, const int32_t* lwork, int32_t* info) {
  dgerqf_(m, n, a, lda, tau, work, lwork, info);
}

inline void dorgrq(const int32_t* m, const int32_t* n, const int32_t* k, double* a,
                   const int32_t* lda, const double* tau, double* work, const int32_t* lwork,
                   int32_t* info) {
  dorgrq_(m, n, k, a, lda, tau, work, lwork, info);
}
inline void sorgrq(const int32_t* m, const int32_t* n, const int32_t* k, float* a,
                   const int32_t* lda, const float* tau, float* work, const int32_t* lwork,
                   int32_t* info) {
  sorgrq_(m, n, k, a, lda, tau, work, lwork, info);
}
inline void zgeqrf(const int32_t* m, const int32_t* n, std::complex<double>* a, const int32_t* lda,
                   std::complex<double>* tau, std::complex<double>* work, const int32_t* lwork,
                   int32_t* info) {
  zgeqrf_(m, n, a, lda, tau, work, lwork, info);
}
inline void cgeqrf(const int32_t* m, const int32_t* n, std::complex<float>* a, const int32_t* lda,
                   std::complex<float>* tau, std::complex<float>* work, const int32_t* lwork,
                   int32_t* info) {
  cgeqrf_(m, n, a, lda, tau, work, lwork, info);
}
inline void zungqr(const int32_t* m, const int32_t* n, const int32_t* k, std::complex<double>* a,
                   const int32_t* lda, const std::complex<double>* tau, std::complex<double>* work,
                   const int32_t* lwork, int32_t* info) {
  zungqr_(m, n, k, a, lda, tau, work, lwork, info);
}
inline void cungqr(const int32_t* m, const int32_t* n, const int32_t* k, std::complex<float>* a,
                   const int32_t* lda, const std::complex<float>* tau, std::complex<float>* work,
                   const int32_t* lwork, int32_t* info) {
  cungqr_(m, n, k, a, lda, tau, work, lwork, info);
}

inline void zgerqf(const int32_t* m, const int32_t* n, std::complex<double>* a, const int32_t* lda,
                   std::complex<double>* tau, std::complex<double>* work, const int32_t* lwork,
                   int32_t* info) {
  zgerqf_(m, n, a, lda, tau, work, lwork, info);
}

inline void zungrq(const int32_t* m, const int32_t* n, const int32_t* k, std::complex<double>* a,
                   const int32_t* lda, const std::complex<double>* tau, std::complex<double>* work,
                   const int32_t* lwork, int32_t* info) {
  zungrq_(m, n, k, a, lda, tau, work, lwork, info);
}

inline void zgeqp3(const int32_t* m, const int32_t* n, std::complex<double>* a, const int32_t* lda,
                   int* jpvt, std::complex<double>* tau, std::complex<double>* work, int* lwork,
                   double* rwork, int32_t* info) {
  zgeqp3_(m, n, a, lda, jpvt, tau, work, lwork, rwork, info);
}
/*
inline void zgeqpf( const int32_t* m, const int32_t* n, std::complex<double>* a, const int32_t* lda,
              int* jpvt, std::complex<double>* tau, std::complex<double>* work,
              double* rwork, int32_t* info ){
  zgeqpf_(m, n, a, lda, jpvt, tau, work, rwork, info);
}
*/
inline void zgelqf(const int32_t* m, const int32_t* n, std::complex<double>* a, const int32_t* lda,
                   std::complex<double>* tau, std::complex<double>* work, const int32_t* lwork,
                   int32_t* info) {
  zgelqf_(m, n, a, lda, tau, work, lwork, info);
}
inline void cgelqf(const int32_t* m, const int32_t* n, std::complex<float>* a, const int32_t* lda,
                   std::complex<float>* tau, std::complex<float>* work, const int32_t* lwork,
                   int32_t* info) {
  cgelqf_(m, n, a, lda, tau, work, lwork, info);
}

inline void zunglq(const int32_t* m, const int32_t* n, const int32_t* k, std::complex<double>* a,
                   const int32_t* lda, const std::complex<double>* tau, std::complex<double>* work,
                   const int32_t* lwork, int32_t* info) {
  zunglq_(m, n, k, a, lda, tau, work, lwork, info);
}
inline void cunglq(const int32_t* m, const int32_t* n, const int32_t* k, std::complex<float>* a,
                   const int32_t* lda, const std::complex<float>* tau, std::complex<float>* work,
                   const int32_t* lwork, int32_t* info) {
  cunglq_(m, n, k, a, lda, tau, work, lwork, info);
}
inline void zgeqlf(const int32_t* m, const int32_t* n, std::complex<double>* a, const int32_t* lda,
                   std::complex<double>* tau, std::complex<double>* work, const int32_t* lwork,
                   int32_t* info) {
  zgeqlf_(m, n, a, lda, tau, work, lwork, info);
}

inline void zungql(const int32_t* m, const int32_t* n, const int32_t* k, std::complex<double>* a,
                   const int32_t* lda, const std::complex<double>* tau, std::complex<double>* work,
                   const int32_t* lwork, int32_t* info) {
  zungql_(m, n, k, a, lda, tau, work, lwork, info);
}

#endif  // UNI_MKL

#endif
