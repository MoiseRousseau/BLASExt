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

| Function     | ``type(A)`` | ``type(B)`` | ``type(D)`` | Status |
|--------------|-------------|-------------|-------------|--------|
| ``C = A * D * A^T`` | D |  | Z | OK |
| ``C = A * D * A^H`` | Z |  | Z | TODO |
| ``C = A * B * A^T`` | D | D |  | TODO |
| ``C = A * B * A^H`` | Z | Z |  | TODO |

Note `A`, `B` and `C` are general matrices and `D` is a diagonal matrix.
`D` and `Z` represent double and complex double type.


## TODO

* Correct warning during compilation
* Autodetection of AVX2 capability
