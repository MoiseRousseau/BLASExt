#ifndef BLASExt_h
#define BLASExt_h

#include <complex.h>


//Naive version
void dvdvt_naive(double complex* G, const double* V, const double complex* D, const int L, const int M);
//void zvdvh_naive(double complex* G, const double complex* V, const double complex* D, const int L, const int M);


#ifdef HAVE_AVX2
// AVX2 VERSION
void dvdvt(double complex* G, const double* V, const double complex* D, const int L, const int M);
void kernel_dvdvt(double complex* G, const double* V, const double complex* D, const int x, const int y, const int l, const int r, const int M, const int L);
void kernel_dvdvt_hor(double complex* G, const double* V, const double complex* D, const int x, const int y, const int l, const int r, const int M, const int L);
#endif

#if 0
// AVX512 VERSION
void VDVH_kernel_avx512(std::vector<Complex> &G, const std::vector<double> &V, const std::vector<Complex> &D, const int L, const int M);
void kernel_avx512(Complex* G, const double* V, const Complex* D, const int x, const int y, const int l, const int r, const int M, const int L);

void VDVH_kernel_avx512(std::vector<Complex> &G, const std::vector<Complex> &V, const std::vector<Complex> &D, const int L, const int M);
void kernel_avx512(void* G, Complex* V, Complex* D, int x, int y, int l, int r, int M, int L);
#endif



#endif
