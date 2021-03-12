/* 
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 
CC = cc
OPT = -O3
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/

#include <immintrin.h>


const char* dgemm_desc = "Blocked, three-loop dgemm.";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */    
void square_dgemm (int n, double* A, double* B, double* C)
{
  // TODO: Implement the blocking optimization
  __m256d ymm0, ymm1, ymm2; 
  int s = 8; 
  int b = n/s+1; 
  __attribute__((aligned(32))) double as[s]; 
  __attribute__((aligned(32))) double buff[4]; 
  for (int i = 0; i < b; i++)
  {
    for (int j = 0; j < b; j++)
    {
      for (int k = 0; k < b; k++)
      {
        for (int si = 0; si < s && (si+s*i) < n; si++)
        {
          for (int sj = 0; sj < s && (sj+s*j) < n; sj++)
          {
            double cij = C[s*i+si+n*(s*j+sj)]; 
            int sk, ssk; 
            for (sk = 0; sk < s && (sk+s*k) < n ; sk++)
            { 
              as[sk] = A[i*s+si+n*(s*k+sk)]; 
            }
            for (ssk = 0; ssk+4 < sk; ssk = ssk+4)
            {
              ymm0 = _mm256_load_pd(&as[ssk]); 
              ymm1 = _mm256_loadu_pd(&B[k*s+ssk+n*(sj+s*j)]); 
              //ymm1 = _mm256_mul_pd(ymm0, ymm1); 
              _mm256_store_pd(buff, _mm256_mul_pd(ymm0, ymm1)); 
            }
            cij += buff[0]+buff[1]+buff[2]+buff[3]; 
            for (; ssk < sk ; ssk++)
            {
              cij += as[ssk]*B[k*s+ssk+n*(sj+s*j)]; 
            }
            C[s*i+si+n*(sj+s*j)]=cij; 
          }
        } 
      }
    }
  }
}





/*

void square_dgemm (int n, double* A, double* B, double* C)
{
  // TODO: Implement the blocking optimization
  __m256 ymm0, ymm1, ymm2; 
  int s = 8; 
  int b = n/s+1; 
  double bs[s]; 
  for (int i = 0; i < b; i++)
  {
    for (int j = 0; j < b; j++)
    {
      for (int k = 0; k < b; k++)
      {
        for (int si = 0; si < s && (si+s*i) < n; si++)
        {
          for (int sj = 0; sj < s && (sj+s*j) < n; sj++)
          {
            double cij = C[s*i+si+n*(s*j+sj)]; 
            for (int sk = 0; sk < s && (sk+s*k) < n ; sk++)
            { 
              bs[sk] = B[k*s+sk+n*(sj+s*j)]; 
              for (int skk = 0; skk < 4; skk++)
              {
                ymm0 = _mm256_loadu_pd(&A[i*s+si+n*(s*k+skk)]); 
                ymm1 = _mm256_loadu_pd(&B[k*s+skk+n*(sj+s*j)]); 
              }
              cij += A[i*s+si+n*(s*k+sk)]*B[k*s+sk+n*(sj+s*j)]; 
            }
            C[s*i+si+n*(sj+s*j)]=cij;   
          }
        } 
      }
    }
  }
}
*/

/*
for (int si = 0; si < s && (si+s*i) < n; si++)
      {
        for (int sj = 0; sj < s && (sj+j*s) < n; sj++)
        {
          double cij = C[s*i+si+n*(s*j+sj)]; 
          for (int k = 0; k < b; k++)
          {
            for (int sk = 0; sk < s && (sk+s*k)< n; sk++)
            {
              cij += A[i*s+si+n*(s*k+sk)]*B[k*s+sk+n*(sj+s*j)]; 
            }
          }
          C[s*i+si+n*(sj+s*j)]=cij; 
        }
      }

*/ 

