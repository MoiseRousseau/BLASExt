#include "BLASExt.h"
#include <complex.h>
#include <string.h>

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
void zvdvh_naive(double complex* G, const complex* V, const complex* D, const int L, const int M) {
  for(int i=0; i<L*L; ++i) G[i] = 0;
  for (int a=0; a<L; ++a) //rows
    for (int b=0; b<L; ++b) //column
      for (int k=0; k<M; ++k)
        G[a+b*L] += V[a+k*L] * D[k] * conj(V[b+k*L]);
}


//
// MEMORY ALIGNED VERSION
//
void zvdvh_mem_align(double complex* G, const complex* V, const complex* D, const int L, const int M) {
  for(int i=0; i<L*L; ++i) G[i] = 0;
  for(int i=0; i<M; ++i){
    for(int a=0; a<L; ++a){
      double complex vai = conj(V[a+i*L])*D[i];
      for(int b=0; b<L; ++b){
        G[b + a*L] += vai*V[b+i*L];
      }
    }
  }
}



#ifdef HAVE_AVX2

//
// KERNEL AVX2 version
//
#define KERNEL_HEIGH_C 4
#define KERNEL_WIDTH_C 2
void kernel_zvdvh(double complex* G, const double complex* V, const double complex* D, const int x, const int y, const int l, const int r, const int M, const int L)
{
  __m256d res[KERNEL_HEIGH_C][KERNEL_WIDTH_C] = {0.}; //hold two complex type
  __m256d reg_temp = {0.}, reg_temp2 = {0.};
  double complex temp;
  double* _V = (double*) V;
  
  for(int k=l; k<r; k++) { //k inner dim to reduce (V column, square size of D)
    //loops must be unrooled
    for (int i = 0; i<KERNEL_HEIGH_C; i++) {
      //broadcast lines of V(x+i,k) * D(k) into a register
      temp = V[x+i + k*L] * D[k];
      reg_temp = _mm256_set_pd(cimag(temp),creal(temp),cimag(temp),creal(temp));
      //now multiply the temp register by column of B
      for (int j = 0; j < KERNEL_WIDTH_C; j++) {
        //we should take indice V^T(k,y+j) and V^T(k,y+j+1)
        //so for V:             V(y+j,k)   and V(y+j+1,k)
        int index = 2*(y+2*j + k*L);
        reg_temp2 = _mm256_set_pd(_V[index+2],_V[index+2],_V[index],_V[index]); //real part
        res[i][j] = _mm256_fmadd_pd(reg_temp2, reg_temp, res[i][j]);
        reg_temp2 = _mm256_set_pd(_V[index+3],-_V[index+3],_V[index+1],-_V[index+1]); //imag part (conjugate)
        reg_temp2 = _mm256_mul_pd(reg_temp2, reg_temp);
        reg_temp2 = _mm256_permute_pd(reg_temp2, 0b0101);
        res[i][j] = _mm256_add_pd(reg_temp2, res[i][j]);
      }
    }
  }
  
  // write the results back to G
  for (int j = 0; j < KERNEL_WIDTH_C; j++) {
    for (int i = 0; i < KERNEL_HEIGH_C; i++) {
      G[(x+i) + (y+2*j)*L] = res[i][j][0] + res[i][j][1]*1.I;
      G[(x+i) + (y+2*j+1)*L] += res[i][j][2] + res[i][j][3]*1.I;
    }
  }
}

void kernel_zvdvh_hor(double complex* G, const double complex* V, const double complex* D, const int x, const int y, const int l, const int r, const int M, const int L)
{
  __m256d res[KERNEL_WIDTH_C] = {0.}; //hold two complex type
  __m256d reg_temp = {0.}, reg_temp2 = {0.};
  double complex temp;
  double* _V = (double*) V;
  
  for(int k=l; k<r; k++) { //k inner dim to reduce (V column, square size of D)
    //broadcast lines of V(x+i,k) * D(k) into a register
    temp = V[x + k*L] * D[k];
    reg_temp = _mm256_set_pd(cimag(temp),creal(temp),cimag(temp),creal(temp));
    //now multiply the temp register by column of B
    for (int j = 0; j < KERNEL_WIDTH_C; j++) {
      //we should take indice V^T(k,y+j) and V^T(k,y+j+1)
      //so for V:             V(y+j,k)   and V(y+j+1,k)
      int index = 2*(y+2*j + k*L);
      reg_temp2 = _mm256_set_pd(_V[index+2],_V[index+2],_V[index],_V[index]); //real part
      res[j] = _mm256_fmadd_pd(reg_temp2, reg_temp, res[j]);
      reg_temp2 = _mm256_set_pd(_V[index+3],-_V[index+3],_V[index+1],-_V[index+1]); //imag part (conjugate)
      reg_temp2 = _mm256_mul_pd(reg_temp2, reg_temp);
      reg_temp2 = _mm256_permute_pd(reg_temp2, 0b0101);
      res[j] = _mm256_add_pd(reg_temp2, res[j]);
    }
  }
  
  // write the results back to G
  for (int j = 0; j < KERNEL_WIDTH_C; j++) {
    G[x + (y+2*j)*L] = res[j][0] + res[j][1]*1.I;
    G[x + (y+2*j+1)*L] += res[j][2] + res[j][3]*1.I;
  }
}



void zvdvh(double complex* G, const double complex* V, const complex* D, const int L, const int M) {
  //G is LpadH * L
  const int LpadH = (L + 2*KERNEL_WIDTH_C-1) / (2*KERNEL_WIDTH_C) * (2*KERNEL_WIDTH_C);
  
  //padding the output matrix to fit the kernel (to remove later)
  double complex* _G = calloc(LpadH * L, sizeof(double complex));
  
  //using the main kernel
  for (int x = 0; x <= L-KERNEL_HEIGH_C; x += KERNEL_HEIGH_C)
    for (int y = 0; y < LpadH; y += 2*KERNEL_WIDTH_C)
      kernel_zvdvh(_G, V, D, x, y, 0, M, M, L);
  
  //using the 1xKERNEL_WIDTH_C kernel to finish
  for (int x = L/KERNEL_HEIGH_C*KERNEL_HEIGH_C; x < L; x += 1)
    for (int y = 0; y < LpadH; y += 2*KERNEL_WIDTH_C)
      kernel_zvdvh_hor(_G, V, D, x, y, 0, M, M, L);
  
  for (int i = 0; i < L; i++) memcpy(&G[i*L], &_G[i*L], L*sizeof(double complex));
  
  free(_G); // every allocated pointer must be freed
}

#endif


