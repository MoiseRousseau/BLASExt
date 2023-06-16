# BLASExt

Some pseudo-BLAS function for matrix operation not supported in canonical BLAS

## Installation

1. Clone the repository with
```
git clone https://github.com/MoiseRousseau/BLASExt
```

2. Compile and install with
```
mkdir build && cd build
cmake ..
make
make install
```

## New BLAS function

List of the new BLAS function implemented in this library:

| Function     | Operation | Status |
|--------------|-----------|--------|
| `dvdvt(double complex* C, double* A, double complex*D, int l, int m)` |``C = A*D*A^T`` | OK |
| `zvdvh(double complex* C, double complex* C A, double complex*D, int l, int m)` |``C = A*D*A^H`` | TODO |

Note `A`, `B` and `C` are general matrices and `D` is a diagonal matrix.
`l` represent the size of the square `C` matrix and `m` the size inner matrix product.


## TODO

* Correct warning during compilation
* Autodetection of AVX2 capability
* Add install target in CMake
