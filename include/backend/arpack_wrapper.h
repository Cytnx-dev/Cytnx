#ifndef __ARPACK_WRAPPER_H_INCLUDED__
#define __ARPACK_WRAPPER_H_INCLUDED__

#include "lapack_wrapper.hpp"
extern "C" {

void dsaupd_(int *ido, char *bmat, int *n, char *which, int *nev, double *tol, double *resid, int *ncv, double *v, int *ldv, int *iparam, int *ipntr, double *workd, double *workl, int *lworkl, int *info);

void dseupd_(int *rvec, char *All, int *select, double *d, double *z, int *ldz, double *sigma, char *bmat, int *n, char *which, int *nev, double *tol, double *resid, int *ncv, double *v, int *ldv, int *iparam, int *ipntr, double *workd, double *workl, int *lworkl, int *info);

void dnaupd_(int *ido, char *bmat, int *n, char *which, int *nev, double *tol, double *resid, int *ncv, double *v, int *ldv, int *iparam, int *ipntr, double *workd, double *workl, int *lworkl, int *info);

void dneupd_(int *rvec, char *howmny, int *select, double *dr, double *di, double *z, int *ldz, double *sigmar, double *sigmai, double *workev, char *bmat, int *n, char *which, int *nev, double *tol, double *resid, int *ncv, double *v, int *ldv, int *iparam, int *ipntr, double *workd, double *workl, int *lworkl, int *info);

void znaupd_(int *ido, char *bmat, int *n, char *which, int *nev, double *tol, std::complex<double> *resid, int *ncv, std::complex<double> *v, int *ldv, int *iparam, int *ipntr, std::complex<double> *workd, std::complex<double> *workl, int *lworkl, double *rwork, int *info);

//void zneupd_(int *rvec, char *All, int *select, std::complex<double> *d, std::complex<double> *z, int *ldz, std::complex<double> *sigma, std::complex<double> *workev, char *bmat, int *n, char *which, int *nev, double *tol, std::complex<double> *resid, int *ncv, std::complex<double> *v, int *ldv, int *iparam, int *ipntr, std::complex<double> *workd, std::complex<double> *workl, int *lworkl, double *rwork, int *info);

void zneupd_(int *rvec, char *howmny, int *select, std::complex<double> *d, std::complex<double> *z, int *ldz, std::complex<double> *sigma, std::complex<double> *workev, char *bmat, int *n, char *which, int *nev, double *tol, std::complex<double> *resid, int *ncv, std::complex<double> *v, int *ldv, int *iparam, int *ipntr, std::complex<double> *workd, std::complex<double> *workl, int *lworkl, double *rwork, int *info);
}

//void arpack_for_corner ( const uni10::UniTensor<double> &target, uni10::Matrix<double> &corner, unsigned int & max_iter, double err_tol );
#endif
