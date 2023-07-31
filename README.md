# BLASExt

Some pseudo-BLAS function for matrix operation not supported in canonical BLAS optimized for AVX2 processors.

## Getting started

### Installation

1. Clone the repository with
```
git clone https://github.com/MoiseRousseau/BLASExt
```

2. Compile with
```
mkdir build && cd build
cmake .. -DBUILD_BENCHMARKS=1
make
```

### Use

1. Include the library header ``BLASExt.h`` in your C or C++ source file.
2. Link your executables / libraries against `libBlasExt.so`.


## New BLAS functions

List of the new pseudo-BLAS function implemented in this library:

| Function     | Operation |
|--------------|-----------|
| `dvdvt(double complex* C, double* A, double complex* D, int L, int M)` |``C = A*D*A^T`` |
| `zvdvh(double complex* C, double complex* A, double complex* D, int L, int M)` |``C = A*D*A^H`` |

Note `A`, `B` and `C` are general matrices and `D` is a diagonal matrix.
`L` represents the size of the square `C` matrix and `M` is the size of inner matrix product.

For a complete list of the new function including naive, memory aligned and BLAS-original implementations developped for comparison, see the ``include/BLASExt.h`` header file.


## Performance versus OpenBLAS

Speedup of the new BLAS functions are compared to an OpenBLAS implementation based on GEMM (see `X_blas_gemm` function in source code):

| Function / Matrix Size (L,M=4*L) | 12 | 32 | 64 | 128 | 256 |
|----------------------------------|----|----|----|-----|-----|
| `dvdvt` | **9.4** | **4.3** | **2.3** | **1.5** | 0.6 |
| `zvdvh` | **2.5** | **1.2** | 0.7 | 0.5 | 0.2 |


## Complete minimal example

The below C source code is the operation ``C = A*D*A^T`` for A and D random real and complex matrices respectively:

```
#include <BLASExt.h>
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>

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
Compute a matrix norm defined as the squared sum of matrix element modulus
*/
double matrix_norm(const double complex* A, const int L, const int M) {
  double norm = 0;
  for (long int i=0; i<L*M; i++) norm += conj(A[i])*A[i];
  return norm;
}

int main() {
    const int L = 64, M = 256;
    double* A = generate_random_dmatrix(L,M);
    double complex* D = (double complex*) generate_random_dmatrix(1,2*M); //diag matrix
    double complex* G = (double complex*) calloc(L * L, sizeof(double complex));
    dvdvt(G, A, D, L, M);
    printf("Matrix norm (squared) = %.6e\n", matrix_norm(G,L,L));
    return 0;
}
```

Save it as `test.c`.

Compilation:
```
export BLASEXTDIR=[Path to BLASExt library root directory]
gcc -o test_blasext test.c -lBLASExt -I$BLASEXTDIR/include -L$BLASEXTDIR/build/
```

Run:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$BLASEXTDIR/build/
./test_blasext
```


## TODO

* What if D matrix is real ?
* CMake Install target ?
* Autodetection of AVX2 capability
* `dvdvt_blas_syrk` not working for L<=32 (nan norm)
