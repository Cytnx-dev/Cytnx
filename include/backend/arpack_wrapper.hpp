#ifndef __ARPACK_WRAPPER_H_INCLUDED__
#define __ARPACK_WRAPPER_H_INCLUDED__

#include "lapack_wrapper.hpp"
#include <complex>

namespace arpack {

extern "C" {

// === Real Symmetric (double precision) ===
void dsaupd_(int *ido, char *bmat, int *n, char *which, int *nev, double *tol,
             double *resid, int *ncv, double *v, int *ldv, int *iparam, int *ipntr,
             double *workd, double *workl, int *lworkl, int *info);

void dseupd_(int *rvec, char *howmny, int *select, double *d, double *z, int *ldz,
             double *sigma, char *bmat, int *n, char *which, int *nev, double *tol,
             double *resid, int *ncv, double *v, int *ldv, int *iparam, int *ipntr,
             double *workd, double *workl, int *lworkl, int *info);

// === Real Symmetric (single precision) ===
void ssaupd_(int *ido, char *bmat, int *n, char *which, int *nev, float *tol,
             float *resid, int *ncv, float *v, int *ldv, int *iparam, int *ipntr,
             float *workd, float *workl, int *lworkl, int *info);

void sseupd_(int *rvec, char *howmny, int *select, float *d, float *z, int *ldz,
             float *sigma, char *bmat, int *n, char *which, int *nev, float *tol,
             float *resid, int *ncv, float *v, int *ldv, int *iparam, int *ipntr,
             float *workd, float *workl, int *lworkl, int *info);

// === Real Non-Symmetric (double precision) ===
void dnaupd_(int *ido, char *bmat, int *n, char *which, int *nev, double *tol,
             double *resid, int *ncv, double *v, int *ldv, int *iparam, int *ipntr,
             double *workd, double *workl, int *lworkl, int *info);

void dneupd_(int *rvec, char *howmny, int *select, double *dr, double *di,
             double *z, int *ldz, double *sigmar, double *sigmai, double *workev,
             char *bmat, int *n, char *which, int *nev, double *tol,
             double *resid, int *ncv, double *v, int *ldv, int *iparam,
             int *ipntr, double *workd, double *workl, int *lworkl, int *info);

// === Real Non-Symmetric (single precision) ===
void snaupd_(int *ido, char *bmat, int *n, char *which, int *nev, float *tol,
             float *resid, int *ncv, float *v, int *ldv, int *iparam, int *ipntr,
             float *workd, float *workl, int *lworkl, int *info);

void sneupd_(int *rvec, char *howmny, int *select, float *dr, float *di,
             float *z, int *ldz, float *sigmar, float *sigmai, float *workev,
             char *bmat, int *n, char *which, int *nev, float *tol,
             float *resid, int *ncv, float *v, int *ldv, int *iparam,
             int *ipntr, float *workd, float *workl, int *lworkl, int *info);

// === Complex General (non-Hermitian, double precision) ===
void znaupd_(int *ido, char *bmat, int *n, char *which, int *nev, double *tol,
             std::complex<double> *resid, int *ncv, std::complex<double> *v, int *ldv,
             int *iparam, int *ipntr, std::complex<double> *workd, std::complex<double> *workl,
             int *lworkl, double *rwork, int *info);

void zneupd_(int *rvec, char *howmny, int *select, std::complex<double> *d,
             std::complex<double> *z, int *ldz, std::complex<double> *sigma,
             std::complex<double> *workev, char *bmat, int *n, char *which,
             int *nev, double *tol, std::complex<double> *resid, int *ncv,
             std::complex<double> *v, int *ldv, int *iparam, int *ipntr,
             std::complex<double> *workd, std::complex<double> *workl,
             int *lworkl, double *rwork, int *info);

// === Complex General (non-Hermitian, single precision) ===
void cnaupd_(int *ido, char *bmat, int *n, char *which, int *nev, float *tol,
             std::complex<float> *resid, int *ncv, std::complex<float> *v, int *ldv,
             int *iparam, int *ipntr, std::complex<float> *workd, std::complex<float> *workl,
             int *lworkl, float *rwork, int *info);

void cneupd_(int *rvec, char *howmny, int *select, std::complex<float> *d,
             std::complex<float> *z, int *ldz, std::complex<float> *sigma,
             std::complex<float> *workev, char *bmat, int *n, char *which,
             int *nev, float *tol, std::complex<float> *resid, int *ncv,
             std::complex<float> *v, int *ldv, int *iparam, int *ipntr,
             std::complex<float> *workd, std::complex<float> *workl,
             int *lworkl, float *rwork, int *info);

} // extern "C"

} //namespace arpack

#endif // __ARPACK_WRAPPER_H_INCLUDED__

