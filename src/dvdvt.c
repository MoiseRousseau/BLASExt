#include "BLASExt.h"
#include <complex.h>
#include <string.h>
#include <cblas.h>

#ifdef HAVE_AVX2
#include <immintrin.h>
#endif

/*
Some refs:
https://en.algorithmica.org/hpc/algorithms/matmul/
https://sci-hub.st/https://dl.acm.org/doi/10.1145/1356052.1356053
https://ia601407.us.archive.org/23/items/cnx-org-col11136/high-performance-computing.pdf
https://people.freebsd.org/~lstewart/articles/cpumemory.pdf
*/


// 
// NAIVE
//
void dvdvt_naive(double complex* G, const double* V, const double complex* D, const int L, const int M) {
  for (int i=0; i<L*L; ++i) G[i] = 0;
  for (int a=0; a<L; ++a) //rows
    for (int b=0; b<L; ++b) //column
      for (int k=0; k<M; ++k)
        G[a+b*L] += V[a+k*L] * D[k] * V[b+k*L]; 
}

//
// MEMORY ALIGNED VERSION
//
void dvdvt_mem_align(double complex* G, const double* V, const double complex* D, const int L, const int M) {
  for(int i=0; i<L*L; ++i) G[i] = 0;
  for(int i=0; i<M; ++i){
    for(int a=0; a<L; ++a){
      double complex vai = V[a+i*L]*D[i];
      for(int b=0; b<L; ++b){
        G[b + a*L] += vai*V[b+i*L]; // original was G(a,b) but was wrong with complex operators
      }
    }
  }
}

//
// BLAS VERSION
//
void dvdvt_blas_gemm(double complex* G, const double* V, const double complex* D, const int L, const int M) {
  double* VD = malloc(L*M*sizeof(double));
  double* G_ = malloc(L*L*sizeof(double));
  //REAL PART
  //Create VD matrix
  cblas_dcopy(L*M, V, 1, VD, 1);
  //Perform VD = V*D
  for (int i=0; i<M; ++i) {
    cblas_dscal(L, creal(D[i]), &VD[i*L], 1);
  }
  //Perform G_ = VD*V^T
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, L, L, M, 1., VD, L, V, L, 0., G_, L);
  //Copy back to G
  cblas_dcopy(L*L, G_, 1, (double*) G, 2);
  
  //IMAG PART
  cblas_dcopy(L*M, V, 1, VD, 1);
  for (int i=0; i<M; ++i) {
    cblas_dscal(L, cimag(D[i]), &VD[i*L], 1);
  }
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, L, L, M, 1., VD, L, V, L, 0., G_, L);
  cblas_dcopy(L*L, G_, 1, &((double*) G)[1], 2);
}

void dvdvt_blas_syrk(double complex* G, const double* V, const double complex* D, const int L, const int M) {
  //Convert V to complex
  double complex* VD = malloc(L*M * sizeof(double complex));
  cblas_dcopy(L*M, V, 1, (double*)VD, 2);
  //Compute V D^1/2
  for (int i=0; i<M; ++i) {
    const double complex temp = csqrt(D[i]);
    cblas_zscal(L, &temp, &VD[i*L], 1);
  }
  const double complex one = 1.+0.I, zero = 0.+0.I;
  cblas_zsyrk(CblasColMajor, CblasLower, CblasNoTrans, L, M, &one, VD, L, &zero, G, L);
}



#ifdef HAVE_AVX2

void kernel_dvdvt(double complex* G, const double* V, const double complex* D, const int x, const int y, const int l, const int r, const int M, const int L);
void kernel_dvdvt_hor(double complex* G, const double* V, const double complex* D, const int x, const int y, const int l, const int r, const int M, const int L);

//
// KERNEL AVX2 version
//
#define KERNEL_HEIGH_D 4
#define KERNEL_WIDTH_D 2
void kernel_dvdvt(double complex* G, const double* V, const double complex* D, const int x, const int y, const int l, const int r, const int M, const int L)
{
  __m256d res[KERNEL_HEIGH_D][KERNEL_WIDTH_D] = {0.}; //hold two complex type
  __m256d reg_temp = {0.}, reg_temp2 = {0.};
  double complex temp;
  
  for(int k=l; k<r; k++) { //k inner dim to reduce (V column, square size of D)
    //loops must be unrooled
    for (int i = 0; i<KERNEL_HEIGH_D; i++) {
      //broadcast lines of V(x+i,k) * D(k) into a register
      temp = V[x+i + k*L] * D[k];
      reg_temp = _mm256_set_pd(cimag(temp),creal(temp),cimag(temp),creal(temp));
      //now multiply the temp register by column of B
      for (int j = 0; j < KERNEL_WIDTH_D; j++) {
        //we should take indice V^T(k,y+j) and V^T(k,y+j+1)
        //so for V:             V(y+j,k)   and V(y+j+1,k)
        int index = y+2*j + k*L;
        reg_temp2 = _mm256_set_pd(V[index+1],V[index+1],V[index],V[index]);
        //res[i][j] += reg_temp2 * reg_temp; // as a vec register and FMA
        res[i][j] = _mm256_fmadd_pd(reg_temp2, reg_temp, res[i][j]);
      }
    }
  }
  
  // write the results back to G considering symmetry
  double* _G = (double*) G;
  for (int j = 0; j < KERNEL_WIDTH_D; j++) {
    for (int i = 0; i < KERNEL_HEIGH_D; i++) {
      if (x+i > y+2*j+1) {
        _G[2*(x+i + (y+2*j+1)*L)] += res[i][j][2]; //lower triangle
        _G[2*(x+i + (y+2*j+1)*L)+1] += res[i][j][3];
        _G[2*(y+2*j+1 + (x+i)*L)] += res[i][j][2]; //upper triangle
        _G[2*(y+2*j+1 + (x+i)*L)+1] += res[i][j][3];
        _G[2*(x+i + (y+2*j)*L)] += res[i][j][0]; //lower triangle
        _G[2*(x+i + (y+2*j)*L)+1] += res[i][j][1];
        _G[2*(y+2*j + (x+i)*L)] += res[i][j][0]; //upper triangle
        _G[2*(y+2*j + (x+i)*L)+1] += res[i][j][1];
      }
      else if (x+i == y+2*j+1) {
        _G[2*(x+i + (y+2*j+1)*L)] += res[i][j][2]; //diagonal
        _G[2*(x+i + (y+2*j+1)*L)+1] += res[i][j][3];
        _G[2*(x+i + (y+2*j)*L)] += res[i][j][0]; //lower triangle
        _G[2*(x+i + (y+2*j)*L)+1] += res[i][j][1];
        _G[2*(y+2*j + (x+i)*L)] += res[i][j][0]; //upper triangle
        _G[2*(y+2*j + (x+i)*L)+1] += res[i][j][1];
      }
      else if (x+i == y+2*j) {
        _G[2*(x+i + (y+2*j)*L)] += res[i][j][0]; //diagonal
        _G[2*(x+i + (y+2*j)*L)+1] += res[i][j][1];
      }
    }
  }
}

void kernel_dvdvt_hor(double complex* G, const double* V, const double complex* D, const int x, const int y, const int l, const int r, const int M, const int L)
{
  __m256d res[KERNEL_WIDTH_D] = {0.}; //hold two complex type
  __m256d reg_temp = {0.}, reg_temp2 = {0.};
  double complex temp;
  
  for(int k=l; k<r; k++) {
    temp = V[x + k*L] * D[k];
    reg_temp = reg_temp = _mm256_set_pd(cimag(temp),creal(temp),cimag(temp),creal(temp));
    for (int j = 0; j < KERNEL_WIDTH_D; j++) {
      int index = y+2*j + k*L;
      reg_temp2 = _mm256_set_pd(V[index+1],V[index+1],V[index],V[index]);
      res[j] = _mm256_fmadd_pd(reg_temp2, reg_temp, res[j]);
    }
  }
  
  // write the results back to G considering symmetry
  double* _G = (double*) G;
  for (int j = 0; j < KERNEL_WIDTH_D; j++) {
    if (x > y+2*j+1) {
      _G[2*(x + (y+2*j+1)*L)] += res[j][2]; //lower triangle
      _G[2*(x + (y+2*j+1)*L)+1] += res[j][3];
      _G[2*(y+2*j+1 + x*L)] += res[j][2]; //upper triangle
      _G[2*(y+2*j+1 + x*L)+1] += res[j][3];
      _G[2*(x + (y+2*j)*L)] += res[j][0]; //lower triangle
      _G[2*(x + (y+2*j)*L)+1] += res[j][1];
      _G[2*(y+2*j + x*L)] += res[j][0]; //upper triangle
      _G[2*(y+2*j + x*L)+1] += res[j][1];
    }
    else if (x == y+2*j+1) {
      _G[2*(x + (y+2*j+1)*L)] += res[j][2]; //diagonal
      _G[2*(x + (y+2*j+1)*L)+1] += res[j][3];
      _G[2*(x + (y+2*j)*L)] += res[j][0]; //lower triangle
      _G[2*(x + (y+2*j)*L)+1] += res[j][1];
      _G[2*(y+2*j + x*L)] += res[j][0]; //upper triangle
      _G[2*(y+2*j + x*L)+1] += res[j][1];
    }
    else if (x == y+2*j) {
      _G[2*(x + (y+2*j)*L)] += res[j][0]; //diagonal
      _G[2*(x + (y+2*j)*L)+1] += res[j][1];
    }
  }
}



void dvdvt(double complex* G, const double* V, const double complex* D, const int L, const int M) {
  //G is LpadH * L
  const int LpadH = (L + 2*KERNEL_WIDTH_D-1) / (2*KERNEL_WIDTH_D) * (2*KERNEL_WIDTH_D);
  
  //padding the output matrix to fit the kernel (to remove later)
  double complex* _G = calloc(L * LpadH, sizeof(double complex));
  
  //using the main kernel
  for (int x = 0; x <= L-KERNEL_HEIGH_D; x += KERNEL_HEIGH_D)
    for (int y = 0; y < x+KERNEL_HEIGH_D; y += 2*KERNEL_WIDTH_D)
      kernel_dvdvt(_G, V, D, x, y, 0, M, M, L);
  
  //using the 1xKERNEL_WIDTH_D kernel to finish
  for (int x = L/KERNEL_HEIGH_D*KERNEL_HEIGH_D; x < L; x += 1)
    for (int y = 0; y <= x; y += 2*KERNEL_WIDTH_D)
      kernel_dvdvt_hor(_G, V, D, x, y, 0, M, M, L);
  
  for (int i = 0; i < L; i++) memcpy(&G[i*L], &_G[i*L], L*sizeof(double complex));
  
  free(_G); // every allocated pointer must be freed
}

#endif


#if 0

//
// KERNEL AVX512 version
// Note the AVX512 kernel seems slower than the AVX2 version...
//

// Double version
#define KERNEL_HEIGH_D_AVX512 4
#define KERNEL_WIDTH_D_AVX512 2
void kernel_avx512(__restrict__ Complex* G, const double* V, const Complex* D, const int x, const int y, const int l, const int r, const int M, const int L)
{
  __m512d res[KERNEL_HEIGH_D_AVX512][KERNEL_WIDTH_D_AVX512]{}; //hold four complex type
  __m512d reg_temp, reg_temp2;
  Complex temp;
  
  for(int k=l; k<r; k++) { //k inner dim to reduce (V column, square size of D)
    //loops must be unrooled
    for (int i = 0; i<KERNEL_HEIGH_D_AVX512; i++) {
      //broadcast lines of V(x+i,k) * D(k) into a register
      temp = V[x+i + k*L] * D[k];
      reg_temp = _mm512_set_pd(
          temp.imag(),temp.real(),temp.imag(),temp.real(),
          temp.imag(),temp.real(),temp.imag(),temp.real()
      );
      //now multiply the temp register by column of B
      for (int j = 0; j < KERNEL_WIDTH_D_AVX512; j++) {
        //we should take indice V^T(k,y+j) and V^T(k,y+j+1)
        //so for V:             V(y+j,k)   and V(y+j+1,k)
        int index = y+4*j + k*L;
        reg_temp2 = _mm512_set_pd(V[index+3],V[index+3],V[index+2],V[index+2],V[index+1],V[index+1],V[index],V[index]);
        //res[i][j] += reg_temp2 * reg_temp; // as a vec register and FMA
        res[i][j] = _mm512_fmadd_pd(reg_temp2, reg_temp, res[i][j]);
      }
    }
  }
  // write the results back to G considering symmetry
  double* _G = (double*) G;
  for (int j = 0; j < KERNEL_WIDTH_D_AVX512; j++) {
    for (int i = 0; i < KERNEL_HEIGH_D_AVX512; i++) {
      for (int k=0; k < 4; k++) { //loop over the 4 complex in a register
        if (x+i > y+4*j+k) {
          _G[2*(x+i + (y+4*j+k)*L)] += res[i][j][2*k]; //lower triangle
          _G[2*(x+i + (y+4*j+k)*L)+1] += res[i][j][2*k+1];
          _G[2*(y+4*j+k + (x+i)*L)] += res[i][j][2*k]; //upper triangle
          _G[2*(y+4*j+k + (x+i)*L)+1] += res[i][j][2*k+1];
        }
        else if (x+i == y+4*j+k) {
          _G[2*(x+i + (y+4*j+k)*L)] += res[i][j][2*k]; //diagonal
          _G[2*(x+i + (y+4*j+k)*L)+1] += res[i][j][2*k+1];
        }
      }
    }
  }
}

void kernel_avx512_hor(__restrict__ Complex* G, const double* V, const Complex* D, const int x, const int y, const int l, const int r, const int M, const int L)
{
  __m512d res[KERNEL_WIDTH_D_AVX512]{}; //hold four complex type
  __m512d reg_temp, reg_temp2;
  Complex temp;
  
  for(int k=l; k<r; k++) { //k inner dim to reduce (V column, square size of D)
    //loops must be unrooled
    temp = V[x + k*L] * D[k];
    reg_temp = _mm512_set_pd(
      temp.imag(),temp.real(),temp.imag(),temp.real(),
      temp.imag(),temp.real(),temp.imag(),temp.real()
    );
    for (int j = 0; j < KERNEL_WIDTH_D_AVX512; j++) {
      int index = y+4*j + k*L;
      reg_temp2 = _mm512_set_pd(
        V[index+3],V[index+3],V[index+2],V[index+2],
        V[index+1],V[index+1],V[index],V[index]
      );
      res[j] = _mm512_fmadd_pd(reg_temp2, reg_temp, res[j]);
    }
  }
  // write the results back to G considering symmetry
  double* _G = (double*) G;
  for (int j = 0; j < KERNEL_WIDTH_D_AVX512; j++) {
    for (int k=0; k < 4; k++) { //loop over the 4 complex in a register
      if (x > y+4*j+k) {
        _G[2*(x + (y+4*j+k)*L)] += res[j][2*k]; //lower triangle
        _G[2*(x + (y+4*j+k)*L)+1] += res[j][2*k+1];
        _G[2*(y+4*j+k + x*L)] += res[j][2*k]; //upper triangle
        _G[2*(y+4*j+k + x*L)+1] += res[j][2*k+1];
      }
      else if (x == y+4*j+k) {
        _G[2*(x + (y+4*j+k)*L)] += res[j][2*k]; //diagonal
        _G[2*(x + (y+4*j+k)*L)+1] += res[j][2*k+1];
      }
    }
  }
}

void VDVH_kernel_avx512(std::vector<Complex> &G, const std::vector<double> &V, const std::vector<Complex> &D, const int L, const int M) {
  //note Mpad is the size of the inner padded matrix
  //G is LpadH * L
  const int LpadH = (L + 4*KERNEL_WIDTH_D_AVX512-1) / (4*KERNEL_WIDTH_D_AVX512) * (4*KERNEL_WIDTH_D_AVX512);
  
  //padding the input matrix to fit the kernel (to remove later)
  std::vector<Complex> _G; _G.resize(L * LpadH);
    
  for (int x = 0; x <= L-KERNEL_HEIGH_D_AVX512; x += KERNEL_HEIGH_D_AVX512)
    for (int y = 0; y < x+KERNEL_HEIGH_D_AVX512; y += 4*KERNEL_WIDTH_D_AVX512)
      kernel_avx512(_G.data(), V.data(), D.data(), x, y, 0, M, M, L);
  
  //using the 1xKERNEL_WIDTH_D kernel to finish
  for (int x = L/KERNEL_HEIGH_D_AVX512*KERNEL_HEIGH_D_AVX512; x < L; x += 1)
    for (int y = 0; y <= x; y += 4*KERNEL_WIDTH_D)
      kernel_avx2_hor(_G.data(), V.data(), D.data(), x, y, 0, M, M, L);
 
  for (int i = 0; i < L; i++) std::copy(_G.begin()+ i*L, _G.begin()+i*L+L, G.begin()+i*L);
  
  _G.resize(0);
}

#endif


