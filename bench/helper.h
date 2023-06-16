#ifndef helper_h
#define helper_h

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <sys/time.h>

/* 
Generate a random floating point number from -1 to 1
https://stackoverflow.com/questions/33058848/generate-a-random-double-between-1-and-1
*/
double doublerand() 
{
    double div = RAND_MAX / 2.;
    return -1. + (rand() / div);
}

/*
Create a random matrix (double format)
*/
double* generate_random_dmatrix(int L, int M) {
  double* A = (double*) malloc(L * M * sizeof(double));
  for (int i=0; i<L*M; i++) {
      A[i] = doublerand();
  }
  return A;
}

/*
Create a random matrix (double complex format)
*/
double complex* generate_random_zmatrix(int L, int M) {
  double* A = (double*) malloc(2 * L * M * sizeof(double));
  for (int i=0; i<2*L*M; i++) {
      A[i] = doublerand();
  }
  return (double complex*) A;
}

/*
Create a zero-filled matrix (double complex format)
*/
double complex* generate_zero_zmatrix(int L, int M) {
  double* A = (double*) malloc(2 * L * M * sizeof(double));
  for (int i=0; i<L*M; i++) {
      A[i] = 0.;
  }
  return (double complex*) A;
}

/*
Return time in microsecond
*/
long micro_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * (long)1000000 + tv.tv_usec;
}

#endif
