#ifndef BLASExt_h
#define BLASExt_h

#include <complex.h>

//double V*D*V^T
void dvdvt_naive(double complex* G, const double* V, const double complex* D, const int L, const int M);
void dvdvt_mem_align(double complex* G, const double* V, const double complex* D, const int L, const int M);
void dvdvt_blas_gemm(double complex* G, const double* V, const double complex* D, const int L, const int M);
void dvdvt_blas_syrk(double complex* G, const double* V, const double complex* D, const int L, const int M);
void dvdvt(double complex* G, const double* V, const double complex* D, const int L, const int M);

//double complex V*D*V^H
void zvdvh_naive(double complex* G, const double complex* V, const double complex* D, const int L, const int M);
void zvdvh_mem_align(double complex* G, const double complex* V, const double complex* D, const int L, const int M);
void zvdvh_blas_gemm(double complex* G, const double complex* V, const double complex* D, const int L, const int M);
void zvdvh(double complex* G, const double complex* V, const double complex* D, const int L, const int M);


#endif
