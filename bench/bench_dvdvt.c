#include "helper.h"
#include "../src/BLASExt.h"
#include <time.h>

/* 
Generate a random floating point number from -1 to 1
https://stackoverflow.com/questions/33058848/generate-a-random-double-between-1-and-1
*/
int main(int argc, char **argv) {
    if (argc != 4) {
        printf("Usage:\n  ");
        printf("%s",argv[0]);
        printf(" [output matrix square size] [inner matrix size] [repeat]\n");
        exit(1);
    }
    int L = atoi(argv[1]);
    int M = atoi(argv[2]);
    int repeat = atoi(argv[3]);
    double* A = generate_random_dmatrix(L,M);
    double complex* D = generate_random_zmatrix(1,M); //diag matrix
    double complex* G = generate_zero_zmatrix(L,L);
    
    //Test optimized version
    long start, end;
    start = micro_time();
    for (int i=0; i<repeat; i++) {
        dvdvt(G, A, D, L, M);
    }
    end = micro_time();
    printf("dvdvt %.3f s\n", (end - start) / 1e6);
    printf("Matrix norm (for comparison) = %.6e\n", matrix_norm(G,L,L));
    
    //Test memory aligned version
    start = micro_time();
    for (int i=0; i<repeat; i++) {
        dvdvt_mem_align(G, A, D, L, M);
    }
    end = micro_time();
    printf("---------------------\n");
    printf("dvdvt_mem_align %.3f s\n", (end - start) / 1e6);
    printf("Matrix norm (for comparison) = %.6e\n", matrix_norm(G,L,L));
    
    //compare to BLAS_GEMM version
    start = micro_time();
    for (int i=0; i<repeat; i++) {
        dvdvt_blas_gemm(G, A, D, L, M);
    }
    end = micro_time();
    printf("---------------------\n");
    printf("dvdvt_blas_gemm %.3f s\n", (end - start) / 1e6);
    printf("Matrix norm (for comparison) = %.6e\n", matrix_norm(G,L,L));
    
    
    //compare to BLAS_SYRK version
    start = micro_time();
    for (int i=0; i<repeat; i++) {
        dvdvt_blas_syrk(G, A, D, L, M);
    }
    end = micro_time();
    printf("---------------------\n");
    printf("dvdvt_blas_syrk %.3f s\n", (end - start) / 1e6);
    printf("Matrix norm (for comparison) = %.6e\n", matrix_norm(G,L,L));
    
    //compare to naive version
    start = micro_time();
    for (int i=0; i<repeat; i++) {
        dvdvt_naive(G, A, D, L, M);
    }
    end = micro_time();
    printf("---------------------\n");
    printf("dvdvt_naive %.3f s\n", (end - start) / 1e6);
    printf("Matrix norm (for comparison) = %.6e\n", matrix_norm(G,L,L));
}
