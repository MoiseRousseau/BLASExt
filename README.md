# BLASExt

Some pseudo-BLAS function for matrix operation not supported in canonical BLAS optimized for AVX2 processors.

## Installation

1. Clone the repository with
```
git clone https://github.com/MoiseRousseau/BLASExt
```

2. Compile and install with
```
mkdir build && cd build
cmake .. -DBUILD_BENCHMARKS=1
make
make install
```

## New BLAS function

List of the new pseudo-BLAS function implemented in this library:

| Function     | Operation | Speedup |
|--------------|-----------|---------|
| `dvdvt(double complex* C, double* A, double complex* D, int L, int M)` |``C = A*D*A^T`` | 13x |
| `zvdvh(double complex* C, double complex* A, double complex* D, int L, int M)` |``C = A*D*A^H`` | 6x |

Note `A`, `B` and `C` are general matrices and `D` is a diagonal matrix.
`L` represents the size of the square `C` matrix and `M` is the size of inner matrix product.
Speedup was determined for a random matrix of size `L=67`, `M=500` and compared to a naive implementation.
See the `bench` folder which contains the source file used to compute the speedup.


## TODO

* Autodetection of AVX2 capability
* Add install target in CMake
