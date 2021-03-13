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
inline double hsum_double_avx(__m256d v) {
    __m128d vlow  = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
            vlow  = _mm_add_pd(vlow, vhigh);     // reduce down to 128

    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return  _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar
}

void square_dgemm(int n, double* A, double* B, double* C)
{
  // TODO: Implement the blocking optimization
  __m256d ymm0, ymm1;
  __m256i zero = _mm256_setzero_si256();
  __m256d zerod = _mm256_setzero_pd();
  __m256i mask1 = _mm256_setr_epi32(1,1,0,0,0,0,0,0);
  __m256i mask2 = _mm256_setr_epi32(1,1,1,1,0,0,0,0);
  __m256i mask3 = _mm256_setr_epi32(1,1,1,1,1,1,0,0);
  __m256i masks[4] = {zero, mask1, mask2, mask3};
  __m128i idx = _mm_setr_epi32(0, n, 2*n, 3*n);
  int s = 32; 
  int res = (n - (n/s)*s)%4;
  __m256i mask = (masks[res] != mask);
  __m256d maskd =  _mm256_castsi256_pd(mask);
  //#pragma omp parallel for
  for (int i = 0; i < n; i = i+s)
  {
    for (int j = 0; j < n; j = j+s)
    {
      for (int k = 0; k < n; k = k+s)
      {
        int limi = i+s; 
        int limj = j+s; 
        int limk = k+s; 
        limi = limi < n ? limi:n; 
        limj = limj < n ? limj:n; 
        limk = limk < n ? limk:n; 
        for (int sj = j; sj < limj; sj++)
        {
          //#pragma GCC unroll 4
          for (int si = i; si < limi; si++)
          {
            double cij = C[si+sj*n]; 
            int ssk; 
            //#pragma GCC unroll 2
            for (ssk = 0; ssk+4 < limk-k+1; ssk = ssk+4)
            {
              ymm0 = _mm256_i32gather_pd(&A[si+n*(k+ssk)], idx, 8);
              ymm1 = _mm256_loadu_pd(&B[ssk+k+n*sj]); 
              //ymm1 = _mm256_mul_pd(ymm0, ymm1); 
              cij += hsum_double_avx(_mm256_mul_pd(ymm0, ymm1));
            }
            if (k+ssk < limk) {
                ymm0 = _mm256_mask_i32gather_pd(zerod, &A[si+n*(k+ssk)], idx, maskd, 8);
                ymm1 = _mm256_maskload_pd(&B[ssk+k+n*sj], mask);
                cij += hsum_double_avx(_mm256_mul_pd(ymm0, ymm1));
            }
            C[si+n*sj]=cij;
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

