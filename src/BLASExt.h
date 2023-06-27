#ifndef BLASExt_h
#define BLASExt_h

#include <complex.h>


//Naive version
void dvdvt_naive(double complex* G, const double* V, const double complex* D, const int L, const int M);
void zvdvh_naive(double complex* G, const double complex* V, const double complex* D, const int L, const int M);


#ifdef HAVE_AVX2
void dvdvt(double complex* G, const double* V, const double complex* D, const int L, const int M);
void kernel_dvdvt(double complex* G, const double* V, const double complex* D, const int x, const int y, const int l, const int r, const int M, const int L);
void kernel_dvdvt_hor(double complex* G, const double* V, const double complex* D, const int x, const int y, const int l, const int r, const int M, const int L);

void zvdvh(double complex* G, const double complex* V, const double complex* D, const int L, const int M);
void kernel_zvdvh(double complex* G, const double complex* V, const double complex* D, const int x, const int y, const int l, const int r, const int M, const int L);
void kernel_zvdvh_hor(double complex* G, const double complex* V, const double complex* D, const int x, const int y, const int l, const int r, const int M, const int L);
#endif

#endif
