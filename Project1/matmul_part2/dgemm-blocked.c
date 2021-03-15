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

#include <omp.h>
#include <immintrin.h>

const char* dgemm_desc = "Blocked-openmp,  three-loop dgemm.";

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

void square_dgemm_unroll(int n, double* A, double* B, double* C)
{
  // TODO: Implement the blocking optimization
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
  #pragma omp parallel for schedule(dynamic) default(none) shared(n, A, B, C, masks, idx, s, res, mask, maskd, zero, zerod) 
  for (int i = 0; i < n; i = i+s)
  {
    for (int j = 0; j < n; j = j+s)
    {
      for (int k = 0; k < n; k = k+s)
      {
        __m256d ymm0, ymm1; 
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

#define _MM_HINT_T0 1
#define _MM_HINT_T1 2
#define _MM_HINT_T2 3
#define _MM_HINT_NTA 0
#define _MM_HINT_ENTA 4
#define _MM_HINT_ET0 5
#define _MM_HINT_ET1 6
#define _MM_HINT_ET2 7


/* This routine performs a dgemm operation
 *  C = C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */  


void square_dgemm(int n, double* A, double* B, double* C)
{
  // TODO: Implement the blocking optimization
  __m256i zero = _mm256_setzero_si256();
  __m256d zerod = _mm256_setzero_pd();
  __m256i mask1 = _mm256_setr_epi32(1,1,0,0,0,0,0,0);
  __m256i mask2 = _mm256_setr_epi32(1,1,1,1,0,0,0,0);
  __m256i mask3 = _mm256_setr_epi32(1,1,1,1,1,1,0,0);
  __m256i masks[4] = {zero, mask1, mask2, mask3};
  __m128i idx = _mm_setr_epi32(0, n, 2*n, 3*n);
  const int s = 8; 
  int res = (n - (n/s)*s)%4;
  __m256i mask = (masks[res] != zero);
  __m256d maskd =  _mm256_castsi256_pd(mask);
  #pragma omp parallel for schedule(static) default(none) shared(A, B, C) firstprivate(n, zero, zerod, idx, res, mask, maskd)
  for (int j = 0; j < n; j = j+s)
  {
    int limj = j+s; 
    limj = limj < n ? limj:n; 
    for (int k = 0; k < n; k = k+s)
    {
      int limk = k+s; 
      limk = limk < n ? limk:n; 
      for (int i = 0; i < n; i = i+s)
      {
        int limi = i+s; 
        limi = limi < n ? limi:n; 
        if (limk < n && limi < n && limj < n) {
          _mm_prefetch(&C[i+8+n*j], _MM_HINT_ENTA);
          _mm_prefetch(&C[i+8+n*(j+1)], _MM_HINT_ENTA);
          _mm_prefetch(&C[i+8+n*(j+2)], _MM_HINT_ENTA);
          _mm_prefetch(&C[i+8+n*(j+3)], _MM_HINT_ENTA);
          _mm_prefetch(&C[i+8+n*(j+4)], _MM_HINT_ENTA);
          _mm_prefetch(&C[i+8+n*(j+5)], _MM_HINT_ENTA);
          _mm_prefetch(&C[i+8+n*(j+6)], _MM_HINT_ENTA);
          _mm_prefetch(&C[i+8+n*(j+7)], _MM_HINT_ENTA);
          /*
          _mm_prefetch(&A[i+8+n*k], _MM_HINT_T0);
          _mm_prefetch(&A[i+8+n*(k+1)], _MM_HINT_T0);
          _mm_prefetch(&A[i+8+n*(k+2)], _MM_HINT_T0);
          _mm_prefetch(&A[i+8+n*(k+3)], _MM_HINT_T0);
          _mm_prefetch(&A[i+8+n*(k+4)], _MM_HINT_T0);
          _mm_prefetch(&A[i+8+n*(k+5)], _MM_HINT_T0);
          _mm_prefetch(&A[i+8+n*(k+6)], _MM_HINT_T0);
          _mm_prefetch(&A[i+8+n*(k+7)], _MM_HINT_T0);
          */
          for (int sj = j; sj < j+s; sj += 4)
            for (int si = i; si < i+s; si += 4)
              for (int sk = k; sk < k+s; sk += 4) {
                __m256d b00, b01, b02, b03, b10, b11, b12, b13, b20, b21, b22, b23, b30, b31, b32, b33;
                __m256d c0, c1, c2, c3, a0, a1, a2, a3;
                b00 = _mm256_set1_pd(B[sk+n*(sj+0)]);
                b10 = _mm256_set1_pd(B[sk+1+n*(sj+0)]);
                b20 = _mm256_set1_pd(B[sk+2+n*(sj+0)]);
                b30 = _mm256_set1_pd(B[sk+3+n*(sj+0)]);
                b01 = _mm256_set1_pd(B[sk+n*(sj+1)]);
                b11 = _mm256_set1_pd(B[sk+1+n*(sj+1)]);
                b21 = _mm256_set1_pd(B[sk+2+n*(sj+1)]);
                b31 = _mm256_set1_pd(B[sk+3+n*(sj+1)]);
                b02 = _mm256_set1_pd(B[sk+n*(sj+2)]);
                b12 = _mm256_set1_pd(B[sk+1+n*(sj+2)]);
                b22 = _mm256_set1_pd(B[sk+2+n*(sj+2)]);
                b32 = _mm256_set1_pd(B[sk+3+n*(sj+2)]);
                b03 = _mm256_set1_pd(B[sk+n*(sj+3)]);
                b13 = _mm256_set1_pd(B[sk+1+n*(sj+3)]);
                b23 = _mm256_set1_pd(B[sk+2+n*(sj+3)]);
                b33 = _mm256_set1_pd(B[sk+3+n*(sj+3)]);
                a0 = _mm256_loadu_pd(&A[si+n*sk]);
                a1 = _mm256_loadu_pd(&A[si+n*(sk+1)]);
                a2 = _mm256_loadu_pd(&A[si+n*(sk+2)]);
                a3 = _mm256_loadu_pd(&A[si+n*(sk+3)]);
                c0 = _mm256_loadu_pd(&C[si+sj*n]);
                c1 = _mm256_loadu_pd(&C[si+(sj+1)*n]);
                c2 = _mm256_loadu_pd(&C[si+(sj+2)*n]);
                c3 = _mm256_loadu_pd(&C[si+(sj+3)*n]);
                c0 = _mm256_add_pd(_mm256_fmadd_pd(a0, b00, _mm256_fmadd_pd(a1, b10, c0)), _mm256_fmadd_pd(a2, b20, _mm256_mul_pd(a3, b30)));
                c1 = _mm256_add_pd(_mm256_fmadd_pd(a0, b01, _mm256_fmadd_pd(a1, b11, c1)), _mm256_fmadd_pd(a2, b21, _mm256_mul_pd(a3, b31)));
                c2 = _mm256_add_pd(_mm256_fmadd_pd(a0, b02, _mm256_fmadd_pd(a1, b12, c2)), _mm256_fmadd_pd(a2, b22, _mm256_mul_pd(a3, b32)));
                c3 = _mm256_add_pd(_mm256_fmadd_pd(a0, b03, _mm256_fmadd_pd(a1, b13, c3)), _mm256_fmadd_pd(a2, b23, _mm256_mul_pd(a3, b33)));
                _mm256_storeu_pd(&C[si+sj*n], c0);
                _mm256_storeu_pd(&C[si+(sj+1)*n], c1);
                _mm256_storeu_pd(&C[si+(sj+2)*n], c2);
                _mm256_storeu_pd(&C[si+(sj+3)*n], c3);
              }
        } else
        for (int sj = j; sj < limj; sj++)
        {
          for (int si = i; si < limi; si++)
          {
            __m256d ymm0, ymm1;
            double cij = C[si+sj*n]; 
            int ssk; 
            for (ssk = 0; ssk+4 < limk-k+1; ssk = ssk+4)
            {
                ymm0 = _mm256_i32gather_pd(&A[si+n*(k+ssk)], idx, 8);
                ymm1 = _mm256_loadu_pd(&B[ssk+k+n*sj]); 
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

