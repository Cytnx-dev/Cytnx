#ifndef __LAPACK_WRAPPER_H__
#define __LAPACK_WRAPPER_H__

#include <stdint.h>

#include <complex>
#include <complex.h>
#include <algorithm>
#include "Type.hpp"

#ifdef UNI_MKL
  #include <mkl.h>
typedef MKL_INT blas_int;

#else
//#include <cblas.h>
//#ifdef BLAS_ILP64
//    typedef int64_t blas_int;
//#else
typedef int32_t blas_int;
  //#endif
  #include <lapacke.h>

extern "C" {

// BLAS functions
void dgemm_(const char *transa, const char *transb, const blas_int *m, const blas_int *n,
            const blas_int *k, const double *alpha, const double *a, const blas_int *lda,
            const double *b, const blas_int *ldb, const double *beta, double *c,
            const blas_int *ldc);
void sgemm_(const char *transa, const char *transb, const blas_int *m, const blas_int *n,
            const blas_int *k, const float *alpha, const float *a, const blas_int *lda,
            const float *b, const blas_int *ldb, const float *beta, float *c, const blas_int *ldc);
void zgemm_(const char *transa, const char *transb, const blas_int *m, const blas_int *n,
            const blas_int *k, const std::complex<double> *alpha, const std::complex<double> *a,
            const blas_int *lda, const std::complex<double> *b, const blas_int *ldb,
            const std::complex<double> *beta, std::complex<double> *c, const blas_int *ldc);
void cgemm_(const char *transa, const char *transb, const blas_int *m, const blas_int *n,
            const blas_int *k, const std::complex<float> *alpha, const std::complex<float> *a,
            const blas_int *lda, const std::complex<float> *b, const blas_int *ldb,
            const std::complex<float> *beta, std::complex<float> *c, const blas_int *ldc);

// void   zgemm_(const char *transa, const char *transb, const blas_int *m, const blas_int *n, const
// blas_int *k,
//               const double _Complex *alpha, const double _Complex *a, const blas_int *lda, const
//               double _Complex *b, const blas_int *ldb, const double _Complex *beta, double
//               _Complex *c, const blas_int *ldc);

double dasum_(const blas_int *n, const double *x, const blas_int *incx);

void scopy_(const blas_int *n, const float *x, const blas_int *incx, float *y,
            const blas_int *incy);
void dcopy_(const blas_int *n, const double *x, const blas_int *incx, double *y,
            const blas_int *incy);
// void   zcopy_(const blas_int *n, const double *x, const blas_int *incx, std::complex<double> *y,
// const blas_int *incy);

void daxpy_(const blas_int *n, const double *alpha, const double *x, const blas_int *incx,
            double *y, const blas_int *incy);
void saxpy_(const blas_int *n, const float *alpha, const float *x, const blas_int *incx, float *y,
            const blas_int *incy);
void zaxpy_(const blas_int *n, const std::complex<double> *alpha, const std::complex<double> *x,
            const blas_int *incx, std::complex<double> *y, const blas_int *incy);
void caxpy_(const blas_int *n, const std::complex<float> *alpha, const std::complex<float> *x,
            const blas_int *incx, std::complex<float> *y, const blas_int *incy);
void dscal_(const blas_int *n, const double *a, double *x, const blas_int *incx);
void sscal_(const blas_int *n, const float *a, float *x, const blas_int *incx);
void zscal_(const blas_int *n, const std::complex<double> *a, std::complex<double> *x,
            const blas_int *incx);
void cscal_(const blas_int *n, const std::complex<float> *a, std::complex<float> *x,
            const blas_int *incx);
void zdscal_(const blas_int *n, const double *a, std::complex<double> *x, const blas_int *incx);

float snrm2_(const blas_int *n, const float *x, const blas_int *incx);
float scnrm2_(const blas_int *n, const std::complex<float> *x, const blas_int *incx);
double dnrm2_(const blas_int *n, const double *x, const blas_int *incx);
double dznrm2_(const blas_int *n, const std::complex<double> *x, const blas_int *incx);

void sgemv_(const char *trans, const blas_int *m, const blas_int *n, const float *alpha,
            const float *a, const blas_int *lda, const float *x, const blas_int *incx,
            const float *beta, const float *y, const blas_int *incy);

void dgemv_(const char *trans, const blas_int *m, const blas_int *n, const double *alpha,
            const double *a, const blas_int *lda, const double *x, const blas_int *incx,
            const double *beta, const double *y, const blas_int *incy);

void zgemv_(const char *trans, const blas_int *m, const blas_int *n,
            const std::complex<double> *alpha, const std::complex<double> *a, const blas_int *lda,
            const std::complex<double> *x, const blas_int *incx, const std::complex<double> *beta,
            const std::complex<double> *y, const blas_int *incy);

void cgemv_(const char *trans, const blas_int *m, const blas_int *n,
            const std::complex<float> *alpha, const std::complex<float> *a, const blas_int *lda,
            const std::complex<float> *x, const blas_int *incx, const std::complex<float> *beta,
            const std::complex<float> *y, const blas_int *incy);

double ddot_(const blas_int *n, const double *x, const blas_int *incx, const double *y,
             const blas_int *incy);
float sdot_(const blas_int *n, const float *x, const blas_int *incx, const float *y,
            const blas_int *incy);

  #ifndef FORTRAN_COMPLEX_FUNCTIONS_RETURN_VOID
std::complex<double> zdotc_(const blas_int *n, const std::complex<double> *x, const blas_int *incx,
                            const std::complex<double> *y, const blas_int *incy);
std::complex<double> zdotu_(const blas_int *n, const std::complex<double> *x, const blas_int *incx,
                            const std::complex<double> *y, const blas_int *incy);
std::complex<float> cdotc_(const blas_int *n, const std::complex<float> *x, const blas_int *incx,
                           const std::complex<float> *y, const blas_int *incy);
std::complex<float> cdotu_(const blas_int *n, const std::complex<float> *x, const blas_int *incx,
                           const std::complex<float> *y, const blas_int *incy);
  #else
void zdotc_(std::complex<double> *res, const blas_int *n, const std::complex<double> *x,
            const blas_int *incx, const std::complex<double> *y, const blas_int *incy);
void zdotu_(std::complex<double> *res, const blas_int *n, const std::complex<double> *x,
            const blas_int *incx, const std::complex<double> *y, const blas_int *incy);
void cdotc_(std::complex<float> *res, const blas_int *n, const std::complex<float> *x,
            const blas_int *incx, const std::complex<float> *y, const blas_int *incy);
void cdotu_(std::complex<float> *res, const blas_int *n, const std::complex<float> *x,
            const blas_int *incx, const std::complex<float> *y, const blas_int *incy);
  #endif

// D and Z LAPACK functions
/*
void dgeqp3_( const blas_int* m, const blas_int* n, double* a, const blas_int* lda,
              int* jpvt, double* tau, double* work, int* lwork, blas_int* info );
void zgeqp3_( const blas_int* m, const blas_int* n, std::complex<double>* a, const blas_int* lda,
              int* jpvt, std::complex<double>* tau, std::complex<double>* work,
              int* lwork, double* rwork, blas_int* info );

//void dgeqpf_( const blas_int* m, const blas_int* n, double* a, const blas_int* lda,
//              int* jpvt, double* tau, double* work, blas_int* info );
//void zgeqpf_( const blas_int* m, const blas_int* n, std::complex<double>* a, const blas_int* lda,
//              int* jpvt, std::complex<double>* tau, std::complex<double>* work,
//              double* rwork, blas_int* info );
//

void dgesvd_( const char* jobu, const char* jobvt, const blas_int* m,
              const blas_int* n, double* a, const blas_int* lda, double* s,
              double* u, const blas_int* ldu, double* vt, const blas_int* ldvt,
              double* work, const blas_int* lwork, blas_int* info );

void sgesvd_( const char* jobu, const char* jobvt, const blas_int* m,
              const blas_int* n, float* a, const blas_int* lda, float* s,
              float* u, const blas_int* ldu, float* vt, const blas_int* ldvt,
              float* work, const blas_int* lwork, blas_int* info );

void zgesvd_( const char* jobu, const char* jobvt, const blas_int* m,
              const blas_int* n, std::complex<double>* a, const blas_int* lda, double* s,
              std::complex<double>* u, const blas_int* ldu, std::complex<double>* vt, const
blas_int* ldvt, std::complex<double>* work, const blas_int* lwork, double* rwork, blas_int* info );

void cgesvd_( const char* jobu, const char* jobvt, const blas_int* m,
              const blas_int* n, std::complex<float>* a, const blas_int* lda, float* s,
              std::complex<float>* u, const blas_int* ldu, std::complex<float>* vt, const blas_int*
ldvt, std::complex<float>* work, const blas_int* lwork, float* rwork, blas_int* info );

void dgesdd_( const char* jobz, const blas_int* m, const blas_int* n, double* a,
              const blas_int* lda, double* s, double* u, const blas_int* ldu, double* vt, const
blas_int* ldvt, double* work, const blas_int* lwork, blas_int* iwork, blas_int* info ); void
zgesdd_( const char* jobz, const blas_int* m, const blas_int* n, std::complex<double>* a, const
blas_int* lda, double* s, std::complex<double>* u, const blas_int* ldu, std::complex<double>* vt,
const blas_int* ldvt, std::complex<double>* work, const blas_int* lwork, double* rwork, blas_int*
iwork, blas_int *info );

void ssyev_( const char* jobz, const char* uplo, const blas_int* n, float* a,
             const blas_int* lda, float* w, float* work, const blas_int* lwork,
             blas_int* info );
void dsyev_( const char* jobz, const char* uplo, const blas_int* n, double* a,
             const blas_int* lda, double* w, double* work, const blas_int* lwork,
             blas_int* info );
void cheev_( const char* jobz, const char* uplo, const blas_int* n, std::complex<float>* a,
             const blas_int* lda, float* w, std::complex<float>* work, const blas_int* lwork,
             const float* rwork, blas_int* info );

void zheev_( const char* jobz, const char* uplo, const blas_int* n, std::complex<double>* a,
             const blas_int* lda, double* w, std::complex<double>* work, const blas_int* lwork,
             const double* rwork, blas_int* info );


void zgeev_( const char* jobvl, const char* jobvr, const blas_int* n,  std::complex<double>* a,
             const blas_int* lda, std::complex<double>* w, std::complex<double> *vl, const blas_int
*ldvl, std::complex<double> *vr, const blas_int *ldvr, std::complex<double> *work, const blas_int*
lwork, double *rwork, blas_int* info ); void cgeev_( const char* jobvl, const char* jobvr, const
blas_int* n,  std::complex<float>* a, const blas_int* lda, std::complex<float>* w,
std::complex<float> *vl, const blas_int *ldvl, std::complex<float> *vr, const blas_int *ldvr,
std::complex<float> *work, const blas_int* lwork, float *rwork, blas_int* info ); void dgeev_( const
char* jobvl, const char* jobvr, const blas_int* n,  double* a, const blas_int* lda, double* wr,
double *wl, double *vl, const blas_int *ldvl, double *vr, const blas_int *ldvr, double *work, const
blas_int* lwork, blas_int* info ); void sgeev_( const char* jobvl, const char* jobvr, const
blas_int* n,  float* a, const blas_int* lda, float* wr, float *wl, float *vl, const blas_int *ldvl,
             float *vr, const blas_int *ldvr, float *work, const blas_int* lwork,
             blas_int* info );

void sstev_( const char* jobz, const blas_int* n, const float* d, const float* e, const float* z,
             const blas_int* ldaz, const float* work, blas_int* info );
void dstev_( const char* jobz, const blas_int* n, const double* d, const double* e, const double* z,
             const blas_int* ldaz, const double* work, blas_int* info );

void dgetrf_( const blas_int *m, const blas_int *n, const double *a,
              const blas_int *lda, const blas_int *ipiv, blas_int* info );
void sgetrf_( const blas_int *m, const blas_int *n, const float *a,
              const blas_int *lda, const blas_int *ipiv, blas_int* info );

void zgetrf_( const blas_int *m, const blas_int *n, const std::complex<double> *a,
              const blas_int *lda, const blas_int *ipiv, blas_int* info );
void cgetrf_( const blas_int *m, const blas_int *n, const std::complex<float> *a,
              const blas_int *lda, const blas_int *ipiv, blas_int* info );

void dgetri_( const blas_int *n, const double *a,  const blas_int *lda,
              const blas_int *ipiv, const double* work, const blas_int* lwork, blas_int* info );
void sgetri_( const blas_int *n, const float *a,  const blas_int *lda,
              const blas_int *ipiv, const float* work, const blas_int* lwork, blas_int* info );
void cgetri_( const blas_int *n, const std::complex<float> *a, const blas_int *lda,
              const blas_int *ipiv, const std::complex<float> *work, const blas_int *lwork, blas_int
*info ); void zgetri_( const blas_int *n, const std::complex<double> *a, const blas_int *lda, const
blas_int *ipiv, const std::complex<double> *work, const blas_int *lwork, blas_int *info ); void
sgelqf_( const blas_int* m, const blas_int* n, float* a, const blas_int* lda, float* tau, float*
work, const blas_int* lwork, blas_int* info );

void dgelqf_( const blas_int* m, const blas_int* n, double* a,
              const blas_int* lda, double* tau, double* work, const blas_int* lwork, blas_int* info
); void zgelqf_( const blas_int* m, const blas_int* n, std::complex<double>* a, const blas_int* lda,
std::complex<double>* tau, std::complex<double>* work, const blas_int* lwork, blas_int* info ); void
cgelqf_( const blas_int* m, const blas_int* n, std::complex<float>* a, const blas_int* lda,
std::complex<float>* tau, std::complex<float>* work, const blas_int* lwork, blas_int* info );

void dorglq_( const blas_int* m, const blas_int* n, const blas_int* k, double* a,
              const blas_int* lda, const double* tau, double* work, const blas_int* lwork, blas_int*
info ); void sorglq_( const blas_int* m, const blas_int* n, const blas_int* k, float* a, const
blas_int* lda, const float* tau, float* work, const blas_int* lwork, blas_int* info ); void zunglq_(
const blas_int* m, const blas_int* n, const blas_int* k, std::complex<double>* a, const blas_int*
lda, const std::complex<double>* tau, std::complex<double>* work, const blas_int* lwork, blas_int*
info ); void cunglq_( const blas_int* m, const blas_int* n, const blas_int* k, std::complex<float>*
a, const blas_int* lda, const std::complex<float>* tau, std::complex<float>* work, const blas_int*
lwork, blas_int* info ); void dgeqlf_( const blas_int* m, const blas_int* n, double* a, const
blas_int* lda, double* tau, double* work, const blas_int* lwork, blas_int* info ); void zgeqlf_(
const blas_int* m, const blas_int* n, std::complex<double>* a, const blas_int* lda,
std::complex<double>* tau, std::complex<double>* work, const blas_int* lwork, blas_int* info );

void dorgql_( const blas_int* m, const blas_int* n, const blas_int* k, double* a,
              const blas_int* lda, const double* tau, double* work, const blas_int* lwork, blas_int*
info ); void zungql_( const blas_int* m, const blas_int* n, const blas_int* k, std::complex<double>*
a, const blas_int* lda, const std::complex<double>* tau, std::complex<double>* work, const blas_int*
lwork, blas_int* info );

void dgeqrf_( const blas_int* m, const blas_int* n, double* a,
              const blas_int* lda, double* tau, double* work, const blas_int* lwork, blas_int* info
); void sgeqrf_( const blas_int* m, const blas_int* n, float* a, const blas_int* lda, float* tau,
float* work, const blas_int* lwork, blas_int* info ); void zgeqrf_( const blas_int* m, const
blas_int* n, std::complex<double>* a, const blas_int* lda, std::complex<double>* tau,
std::complex<double>* work, const blas_int* lwork, blas_int* info ); void cgeqrf_( const blas_int*
m, const blas_int* n, std::complex<float>* a, const blas_int* lda, std::complex<float>* tau,
std::complex<float>* work, const blas_int* lwork, blas_int* info ); void dorgqr_( const blas_int* m,
const blas_int* n, const blas_int* k, double* a, const blas_int* lda, const double* tau, double*
work, const blas_int* lwork, blas_int* info ); void sorgqr_( const blas_int* m, const blas_int* n,
const blas_int* k, float* a, const blas_int* lda, const float* tau, float* work, const blas_int*
lwork, blas_int* info );

void zungqr_( const blas_int* m, const blas_int* n, const blas_int* k, std::complex<double>* a,
              const blas_int* lda, const std::complex<double>* tau, std::complex<double>* work,
const blas_int* lwork, blas_int* info ); void cungqr_( const blas_int* m, const blas_int* n, const
blas_int* k, std::complex<float>* a, const blas_int* lda, const std::complex<float>* tau,
std::complex<float>* work, const blas_int* lwork, blas_int* info ); void dgerqf_( const blas_int* m,
const blas_int* n, double* a, const blas_int* lda, double* tau, double* work, const blas_int* lwork,
blas_int* info ); void zgerqf_( const blas_int* m, const blas_int* n, std::complex<double>* a, const
blas_int* lda, std::complex<double>* tau, std::complex<double>* work, const blas_int* lwork,
blas_int* info );

void dorgrq_( const blas_int* m, const blas_int* n, const blas_int* k, double* a,
              const blas_int* lda, const double* tau, double* work, const blas_int* lwork, blas_int*
info ); void sorgrq_( const blas_int* m, const blas_int* n, const blas_int* k, float* a, const
blas_int* lda, const float* tau, float* work, const blas_int* lwork, blas_int* info );

void zungrq_( const blas_int* m, const blas_int* n, const blas_int* k, std::complex<double>* a,
              const blas_int* lda, const std::complex<double>* tau, std::complex<double>* work,
const blas_int* lwork, blas_int* info );

void dstemr_( const char *jobz, const char *range, const blas_int *N, double *D, double *E,const
double *vl, const double *vu, const blas_int *il, const blas_int *iu, blas_int *M, double* W, double
*Z, const blas_int *ldz, const blas_int *nzc, blas_int *isuppz, bool *tryrac, double*work, blas_int*
lwork, blas_int*iwork, blas_int* liwork, blas_int* info);
*/
}  // extern C

// Wrappers for BLAS and LAPACK functions
inline void dgemm(const char *transa, const char *transb, const blas_int *m, const blas_int *n,
                  const blas_int *k, const double *alpha, const double *a, const blas_int *lda,
                  const double *b, const blas_int *ldb, const double *beta, double *c,
                  const blas_int *ldc) {
  dgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline void sgemm(const char *transa, const char *transb, const blas_int *m, const blas_int *n,
                  const blas_int *k, const float *alpha, const float *a, const blas_int *lda,
                  const float *b, const blas_int *ldb, const float *beta, float *c,
                  const blas_int *ldc) {
  sgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void zgemm(const char *transa, const char *transb, const blas_int *m, const blas_int *n,
                  const blas_int *k, const std::complex<double> *alpha,
                  const std::complex<double> *a, const blas_int *lda, const std::complex<double> *b,
                  const blas_int *ldb, const std::complex<double> *beta, std::complex<double> *c,
                  const blas_int *ldc) {
  zgemm_(transa, transb, m, n, k, (const std::complex<double> *)alpha,
         (const std::complex<double> *)a, lda, (const std::complex<double> *)b, ldb,
         (const std::complex<double> *)beta, (std::complex<double> *)c, ldc);
}

inline void cgemm(const char *transa, const char *transb, const blas_int *m, const blas_int *n,
                  const blas_int *k, const std::complex<float> *alpha, const std::complex<float> *a,
                  const blas_int *lda, const std::complex<float> *b, const blas_int *ldb,
                  const std::complex<float> *beta, std::complex<float> *c, const blas_int *ldc) {
  cgemm_(transa, transb, m, n, k, (const std::complex<float> *)alpha,
         (const std::complex<float> *)a, lda, (const std::complex<float> *)b, ldb,
         (const std::complex<float> *)beta, (std::complex<float> *)c, ldc);
}

inline double dasum(const blas_int *n, const double *x, const blas_int *incx) {
  return dasum_(n, x, incx);
}

inline void dcopy(const blas_int &n, const double *x, const blas_int &incx, double *y,
                  const blas_int &incy) {
  dcopy_(&n, x, &incx, y, &incy);
}
inline void scopy(const blas_int &n, const float *x, const blas_int &incx, float *y,
                  const blas_int &incy) {
  scopy_(&n, x, &incx, y, &incy);
}

inline void daxpy(const blas_int *n, const double *alpha, const double *x, const blas_int *incx,
                  double *y, const blas_int *incy) {
  daxpy_(n, alpha, x, incx, y, incy);
}
inline void saxpy(const blas_int *n, const float *alpha, const float *x, const blas_int *incx,
                  float *y, const blas_int *incy) {
  saxpy_(n, alpha, x, incx, y, incy);
}

inline void zaxpy(const blas_int *n, const std::complex<double> *alpha,
                  const std::complex<double> *x, const blas_int *incx, std::complex<double> *y,
                  const blas_int *incy) {
  zaxpy_(n, alpha, x, incx, y, incy);
}
inline void caxpy(const blas_int *n, const std::complex<float> *alpha, const std::complex<float> *x,
                  const blas_int *incx, std::complex<float> *y, const blas_int *incy) {
  caxpy_(n, alpha, x, incx, y, incy);
}

inline double dnrm2(const blas_int *n, const double *x, const blas_int *incx) {
  return dnrm2_(n, x, incx);
}

inline double dznrm2(const blas_int *n, const std::complex<double> *x, const blas_int *incx) {
  return dznrm2_(n, x, incx);
}

inline float snrm2(const blas_int *n, const float *x, const blas_int *incx) {
  return snrm2_(n, x, incx);
}

inline float scnrm2(const blas_int *n, const std::complex<float> *x, const blas_int *incx) {
  return scnrm2_(n, x, incx);
}

inline void dscal(const blas_int *n, const double *a, double *x, const blas_int *incx) {
  dscal_(n, a, x, incx);
}
inline void sscal(const blas_int *n, const float *a, float *x, const blas_int *incx) {
  sscal_(n, a, x, incx);
}
inline void zscal(const blas_int *n, const std::complex<double> *a, std::complex<double> *x,
                  const blas_int *incx) {
  zscal_(n, a, x, incx);
}
inline void cscal(const blas_int *n, const std::complex<float> *a, std::complex<float> *x,
                  const blas_int *incx) {
  cscal_(n, a, x, incx);
}
inline void zdscal(const blas_int *n, const double *a, std::complex<double> *x,
                   const blas_int *incx) {
  zdscal_(n, a, x, incx);
}
/*
inline void dsyev( const char* jobz, const char* uplo, const blas_int* n, double* a,
             const blas_int* lda, double* w, double* work, const blas_int* lwork,
             blas_int* info )
{ dsyev_(  jobz,  uplo,  n,  a, lda, w,  work,  lwork, info ); }
inline void ssyev( const char* jobz, const char* uplo, const blas_int* n, float* a,
             const blas_int* lda, float* w, float* work, const blas_int* lwork,
             blas_int* info )
{ ssyev_(  jobz,  uplo,  n,  a, lda, w,  work,  lwork, info ); }

inline void cheev( const char* jobz, const char* uplo, const blas_int* n, std::complex<float>* a,
             const blas_int* lda, float* w, std::complex<float>* work, const blas_int* lwork,
             const float* rwork, blas_int* info )
{ cheev_(  jobz,  uplo,  n,  a, lda, w,  work,  lwork, rwork, info ); }

inline void zheev( const char* jobz, const char* uplo, const blas_int* n, std::complex<double>* a,
             const blas_int* lda, double* w, std::complex<double>* work, const blas_int* lwork,
             const double* rwork, blas_int* info )
{ zheev_(  jobz,  uplo,  n,  a, lda, w,  work,  lwork, rwork, info ); }

inline void zgeev( const char* jobvl, const char* jobvr, const blas_int* n, std::complex<double>* a,
    const blas_int* lda, std::complex<double>* w, std::complex<double> *vl, const blas_int *ldvl,
    std::complex<double> *vr, const blas_int *ldvr, std::complex<double> *work, const blas_int*
lwork, double *rwork, blas_int* info )
{
  zgeev_(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info);
}

inline void cgeev( const char* jobvl, const char* jobvr, const blas_int* n, std::complex<float>* a,
    const blas_int* lda, std::complex<float>* w, std::complex<float> *vl, const blas_int *ldvl,
    std::complex<float> *vr, const blas_int *ldvr, std::complex<float> *work, const blas_int* lwork,
    float *rwork, blas_int* info )
{
  cgeev_(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info);
}



inline void dgesvd( const char* jobu, const char* jobvt, const blas_int* m,
              const blas_int* n, double* a, const blas_int* lda, double* s,
              double* u, const blas_int* ldu, double* vt, const blas_int* ldvt,
              double* work, const blas_int* lwork, blas_int* info )
{
  dgesvd_( jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info );
}

inline void sgesvd( const char* jobu, const char* jobvt, const blas_int* m,
              const blas_int* n, float* a, const blas_int* lda, float* s,
              float* u, const blas_int* ldu, float* vt, const blas_int* ldvt,
              float* work, const blas_int* lwork, blas_int* info )
{
  sgesvd_( jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info );
}

inline void zgesvd( const char* jobu, const char* jobvt, const blas_int* m,
              const blas_int* n, std::complex<double>* a, const blas_int* lda, double* s,
              std::complex<double>* u, const blas_int* ldu, std::complex<double>* vt, const
blas_int* ldvt, std::complex<double>* work, const blas_int* lwork, double* rwork, blas_int* info )
{
  zgesvd_( jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info );
}

inline void cgesvd( const char* jobu, const char* jobvt, const blas_int* m,
              const blas_int* n, std::complex<float>* a, const blas_int* lda, float* s,
              std::complex<float>* u, const blas_int* ldu, std::complex<float>* vt, const blas_int*
ldvt, std::complex<float>* work, const blas_int* lwork, float* rwork, blas_int* info )
{
  cgesvd_( jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info );
}


inline void dgesdd( const char* jobz, const blas_int* m, const blas_int* n, double* a,
              const blas_int* lda, double* s, double* u, const blas_int* ldu, double* vt, const
blas_int* ldvt, double* work, const blas_int* lwork, blas_int* iwork, blas_int* info )
{
  dgesdd_(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info);
}

inline void zgesdd( const char* jobz, const blas_int* m, const blas_int* n,
              std::complex<double>* a, const blas_int* lda, double* s,
              std::complex<double>* u, const blas_int* ldu,
              std::complex<double>* vt, const blas_int* ldvt,
              std::complex<double>* work, const blas_int* lwork,
              double* rwork, blas_int* iwork, blas_int *info )
{
  zgesdd_(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, info);
}
*/

inline void sgemv(const char *trans, const blas_int *m, const blas_int *n, const float *alpha,
                  const float *a, const blas_int *lda, const float *x, const blas_int *incx,
                  const float *beta, const float *y, const blas_int *incy) {
  sgemv_(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
inline void dgemv(const char *trans, const blas_int *m, const blas_int *n, const double *alpha,
                  const double *a, const blas_int *lda, const double *x, const blas_int *incx,
                  const double *beta, const double *y, const blas_int *incy) {
  dgemv_(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
inline void zgemv(const char *trans, const blas_int *m, const blas_int *n,
                  const std::complex<double> *alpha, const std::complex<double> *a,
                  const blas_int *lda, const std::complex<double> *x, const blas_int *incx,
                  const std::complex<double> *beta, const std::complex<double> *y,
                  const blas_int *incy) {
  zgemv_(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
inline void cgemv(const char *trans, const blas_int *m, const blas_int *n,
                  const std::complex<float> *alpha, const std::complex<float> *a,
                  const blas_int *lda, const std::complex<float> *x, const blas_int *incx,
                  const std::complex<float> *beta, const std::complex<float> *y,
                  const blas_int *incy) {
  cgemv_(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

inline double ddot(const blas_int *n, const double *x, const blas_int *incx, const double *y,
                   const blas_int *incy) {
  return ddot_(n, x, incx, y, incy);
}
inline float sdot(const blas_int *n, const float *x, const blas_int *incx, const float *y,
                  const blas_int *incy) {
  return sdot_(n, x, incx, y, incy);
}

inline void zdotc(std::complex<double> *res, const blas_int *n, const std::complex<double> *x,
                  const blas_int *incx, const std::complex<double> *y, const blas_int *incy) {
  #ifndef FORTRAN_COMPLEX_FUNCTIONS_RETURN_VOID
  *res = zdotc_(n, x, incx, y, incy);
  #else
  zdotc_(res, n, x, incx, y, incy);
  #endif
}
inline void zdotu(std::complex<double> *res, const blas_int *n, const std::complex<double> *x,
                  const blas_int *incx, const std::complex<double> *y, const blas_int *incy) {
  #ifndef FORTRAN_COMPLEX_FUNCTIONS_RETURN_VOID
  *res = zdotu_(n, x, incx, y, incy);
  #else
  zdotu_(res, n, x, incx, y, incy);
  #endif
}
inline void cdotc(std::complex<float> *res, const blas_int *n, const std::complex<float> *x,
                  const blas_int *incx, const std::complex<float> *y, const blas_int *incy) {
  #ifndef FORTRAN_COMPLEX_FUNCTIONS_RETURN_VOID
  *res = cdotc_(n, x, incx, y, incy);
  #else
  cdotc_(res, n, x, incx, y, incy);
  #endif
}
inline void cdotu(std::complex<float> *res, const blas_int *n, const std::complex<float> *x,
                  const blas_int *incx, const std::complex<float> *y, const blas_int *incy) {
  #ifndef FORTRAN_COMPLEX_FUNCTIONS_RETURN_VOID
  *res = cdotu_(n, x, incx, y, incy);
  #else
  cdotu_(res, n, x, incx, y, incy);
  #endif
}

  /*
  inline void dstev( const char* jobz, const blas_int* n, const double* d, const double* e, const
  double* z, const blas_int* ldaz, const double* work, blas_int* info )
  {
    dstev_( jobz, n, d, e, z, ldaz, work, info );
  }

  inline void sstev( const char* jobz, const blas_int* n, const float* d, const float* e, const
  float* z, const blas_int* ldaz, const float* work, blas_int* info )
  {
    sstev_( jobz, n, d, e, z, ldaz, work, info );
  }

  inline void dgetrf( const blas_int *m, const blas_int *n, const double *a,  const blas_int *lda,
  const blas_int *ipiv, blas_int* info )
  {
    dgetrf_( m, n, a, lda, ipiv, info );
  }
  inline void sgetrf( const blas_int *m, const blas_int *n, const float *a,  const blas_int *lda,
  const blas_int *ipiv, blas_int* info )
  {
    sgetrf_( m, n, a, lda, ipiv, info );
  }

  inline void zgetrf( const blas_int *m, const blas_int *n, const std::complex<double> *a,  const
  blas_int *lda, const blas_int *ipiv, blas_int* info )
  {
    zgetrf_(m, n, a, lda, ipiv, info);
  }
  inline void cgetrf( const blas_int *m, const blas_int *n, const std::complex<float> *a,  const
  blas_int *lda, const blas_int *ipiv, blas_int* info )
  {
    cgetrf_(m, n, a, lda, ipiv, info);
  }

  inline void dgetri( const blas_int *n, const double *a,  const blas_int *lda, const blas_int
  *ipiv, const double* work, const blas_int* lwork, blas_int* info )
  {
    dgetri_(n, a, lda, ipiv, work, lwork, info);
  }
  inline void sgetri( const blas_int *n, const float *a,  const blas_int *lda, const blas_int *ipiv,
  const float* work, const blas_int* lwork, blas_int* info )
  {
    sgetri_(n, a, lda, ipiv, work, lwork, info);
  }
  inline void zgetri( const blas_int *n, const std::complex<double> *a, const blas_int *lda, const
  blas_int *ipiv, const std::complex<double> *work, const blas_int *lwork, blas_int *info )
  {
    zgetri_(n, a, lda, ipiv, work, lwork, info);
  }
  inline void cgetri( const blas_int *n, const std::complex<float> *a, const blas_int *lda, const
  blas_int *ipiv, const std::complex<float> *work, const blas_int *lwork, blas_int *info )
  {
    cgetri_(n, a, lda, ipiv, work, lwork, info);
  }

  //inline void dgeqpf( const blas_int* m, const blas_int* n, double* a, const blas_int* lda, int*
  jpvt, double* tau, double* work, blas_int* info ){
  //
  //dgeqpf_(m, n, a, lda, jpvt, tau, work, info);
  //
  //}


  inline void dgeqp3( const blas_int* m, const blas_int* n, double* a, const blas_int* lda, int*
  jpvt, double* tau, double* work, int* lwork, blas_int* info ){

    dgeqp3_(m, n, a, lda, jpvt, tau, work, lwork,info);

  }

  inline void dgelqf( const blas_int* m, const blas_int* n, double* a,
                      const blas_int* lda, double* tau, double* work, const blas_int* lwork,
  blas_int* info )
  {
    dgelqf_(m, n, a, lda, tau, work, lwork, info );
  }
  inline void sgelqf( const blas_int* m, const blas_int* n, float* a,
                      const blas_int* lda, float* tau, float* work, const blas_int* lwork, blas_int*
  info )
  {
    sgelqf_(m, n, a, lda, tau, work, lwork, info );
  }
  inline void dorglq( const blas_int* m, const blas_int* n, const blas_int* k, double* a,
                      const blas_int* lda, const double* tau, double* work, const blas_int* lwork,
  blas_int* info )
  {
    dorglq_(m, n, k, a, lda, tau, work, lwork, info );
  }
  inline void sorglq( const blas_int* m, const blas_int* n, const blas_int* k, float* a,
                      const blas_int* lda, const float* tau, float* work, const blas_int* lwork,
  blas_int* info )
  {
    sorglq_(m, n, k, a, lda, tau, work, lwork, info );
  }



  inline void dgeqlf( const blas_int* m, const blas_int* n, double* a,
                      const blas_int* lda, double* tau, double* work, const blas_int* lwork,
  blas_int* info )
  {
    dgeqlf_(m, n, a, lda, tau, work, lwork, info );
  }

  inline void dorgql( const blas_int* m, const blas_int* n, const blas_int* k, double* a,
                      const blas_int* lda, const double* tau, double* work, const blas_int* lwork,
  blas_int* info )
  {
    dorgql_(m, n, k, a, lda, tau, work, lwork, info );
  }

  inline void dgeqrf( const blas_int* m, const blas_int* n, double* a,
                      const blas_int* lda, double* tau, double* work, const blas_int* lwork,
  blas_int* info )
  {
    dgeqrf_(m, n, a, lda, tau, work, lwork, info );
  }
  inline void sgeqrf( const blas_int* m, const blas_int* n, float* a,
                      const blas_int* lda, float* tau, float* work, const blas_int* lwork, blas_int*
  info )
  {
    sgeqrf_(m, n, a, lda, tau, work, lwork, info );
  }
  inline void dorgqr( const blas_int* m, const blas_int* n, const blas_int* k, double* a,
                      const blas_int* lda, const double* tau, double* work, const blas_int* lwork,
  blas_int* info )
  {
    dorgqr_(m, n, k, a, lda, tau, work, lwork, info );
  }
  inline void sorgqr( const blas_int* m, const blas_int* n, const blas_int* k, float* a,
                      const blas_int* lda, const float* tau, float* work, const blas_int* lwork,
  blas_int* info )
  {
    sorgqr_(m, n, k, a, lda, tau, work, lwork, info );
  }

  inline void dgerqf( const blas_int* m, const blas_int* n, double* a,
                      const blas_int* lda, double* tau, double* work, const blas_int* lwork,
  blas_int* info )
  {
    dgerqf_(m, n, a, lda, tau, work, lwork, info );
  }

  inline void dorgrq(const blas_int* m, const blas_int* n, const blas_int* k, double* a,
                     const blas_int* lda, const double* tau, double* work, const blas_int* lwork,
  blas_int* info )
  {
    dorgrq_(m, n, k, a, lda, tau, work, lwork, info );
  }
  inline void sorgrq(const blas_int* m, const blas_int* n, const blas_int* k, float* a,
                     const blas_int* lda, const float* tau, float* work, const blas_int* lwork,
  blas_int* info )
  {
    sorgrq_(m, n, k, a, lda, tau, work, lwork, info );
  }
  inline void zgeqrf( const blas_int* m, const blas_int* n, std::complex<double>* a,
                      const blas_int* lda, std::complex<double>* tau, std::complex<double>* work,
  const blas_int* lwork, blas_int* info )
  {
    zgeqrf_(m, n, a, lda, tau, work, lwork, info );
  }
  inline void cgeqrf( const blas_int* m, const blas_int* n, std::complex<float>* a,
                      const blas_int* lda, std::complex<float>* tau, std::complex<float>* work,
  const blas_int* lwork, blas_int* info )
  {
    cgeqrf_(m, n, a, lda, tau, work, lwork, info );
  }
  inline void zungqr( const blas_int* m, const blas_int* n, const blas_int* k, std::complex<double>*
  a, const blas_int* lda, const std::complex<double>* tau, std::complex<double>* work, const
  blas_int* lwork, blas_int* info )
  {
    zungqr_(m, n, k, a, lda, tau, work, lwork, info );
  }
  inline void cungqr( const blas_int* m, const blas_int* n, const blas_int* k, std::complex<float>*
  a, const blas_int* lda, const std::complex<float>* tau, std::complex<float>* work, const blas_int*
  lwork, blas_int* info )
  {
    cungqr_(m, n, k, a, lda, tau, work, lwork, info );
  }

  inline void zgerqf( const blas_int* m, const blas_int* n, std::complex<double>* a,
                      const blas_int* lda, std::complex<double>* tau, std::complex<double>* work,
  const blas_int* lwork, blas_int* info )
  {
    zgerqf_(m, n, a, lda, tau, work, lwork, info );
  }

  inline void zungrq( const blas_int* m, const blas_int* n, const blas_int* k, std::complex<double>*
  a, const blas_int* lda, const std::complex<double>* tau, std::complex<double>* work, const
  blas_int* lwork, blas_int* info )
  {
    zungrq_(m, n, k, a, lda, tau, work, lwork, info );
  }

  inline void zgeqp3( const blas_int* m, const blas_int* n, std::complex<double>* a, const blas_int*
  lda, int* jpvt, std::complex<double>* tau, std::complex<double>* work, int* lwork, double* rwork,
  blas_int* info ){ zgeqp3_(m, n, a, lda, jpvt, tau, work, lwork,rwork, info);
  }

  //inline void zgeqpf( const blas_int* m, const blas_int* n, std::complex<double>* a, const
  blas_int* lda,
  //              int* jpvt, std::complex<double>* tau, std::complex<double>* work,
  //              double* rwork, blas_int* info ){
  //  zgeqpf_(m, n, a, lda, jpvt, tau, work, rwork, info);
  //}

  inline void zgelqf( const blas_int* m, const blas_int* n, std::complex<double>* a,
                      const blas_int* lda, std::complex<double>* tau, std::complex<double>* work,
  const blas_int* lwork, blas_int* info )
  {
    zgelqf_(m, n, a, lda, tau, work, lwork, info );
  }
  inline void cgelqf( const blas_int* m, const blas_int* n, std::complex<float>* a,
                      const blas_int* lda, std::complex<float>* tau, std::complex<float>* work,
  const blas_int* lwork, blas_int* info )
  {
    cgelqf_(m, n, a, lda, tau, work, lwork, info );
  }

  inline void zunglq( const blas_int* m, const blas_int* n, const blas_int* k, std::complex<double>*
  a, const blas_int* lda, const std::complex<double>* tau, std::complex<double>* work, const
  blas_int* lwork, blas_int* info )
  {
    zunglq_(m, n, k, a, lda, tau, work, lwork, info );
  }
  inline void cunglq( const blas_int* m, const blas_int* n, const blas_int* k, std::complex<float>*
  a, const blas_int* lda, const std::complex<float>* tau, std::complex<float>* work, const blas_int*
  lwork, blas_int* info )
  {
    cunglq_(m, n, k, a, lda, tau, work, lwork, info );
  }
  inline void zgeqlf( const blas_int* m, const blas_int* n, std::complex<double>* a,
                      const blas_int* lda, std::complex<double>* tau, std::complex<double>* work,
  const blas_int* lwork, blas_int* info )
  {
    zgeqlf_(m, n, a, lda, tau, work, lwork, info );
  }

  inline void zungql( const blas_int* m, const blas_int* n, const blas_int* k, std::complex<double>*
  a, const blas_int* lda, const std::complex<double>* tau, std::complex<double>* work, const
  blas_int* lwork, blas_int* info )
  {
    zungql_(m, n, k, a, lda, tau, work, lwork, info );
  }
  */
#endif  // UNI_MKL

#endif
