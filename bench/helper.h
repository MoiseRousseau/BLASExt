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
Create a column oriented random matrix (double format) of L row and M column
*/
double* generate_random_dmatrix(int L, int M) {
  double* A = (double*) malloc(L * M * sizeof(double));
  for (int i=0; i<L*M; i++) {
      A[i] = doublerand();
  }
  return A;
}

/*
Create a column oriented random matrix (double complex format) of L row and M column
*/
double complex* generate_random_zmatrix(int L, int M) {
  double* A = (double*) malloc(2 * L * M * sizeof(double));
  for (int i=0; i<2*L*M; i++) {
      A[i] = doublerand();
  }
  return (double complex*) A;
}

/*
Create a column oriented zero-filled matrix (double complex format)  of L row and M column
*/
double complex* generate_zero_zmatrix(int L, int M) {
  double* A = (double*) malloc(2 * L * M * sizeof(double));
  for (int i=0; i<L*M; i++) {
      A[i] = 0.;
  }
  return (double complex*) A;
}

/*
Print a matrix to screen
*/
const void print_matrix(const double complex* A, const int L, const int M) {
  printf("(Row,Column) Value\n");
  for (long int j=0; j<M; j++)
    for (long int i=0; i<L; i++)
      printf("(%li,%li) %.3f + %.3fi\n", i, j, creal(A[i+j*L]), cimag(A[i+j*L]));
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
