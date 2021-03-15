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
  int s = 8; 
  int res = (n - (n/s)*s)%4;
  __m256i mask = (masks[res] != zero);
  __m256d maskd =  _mm256_castsi256_pd(mask);
  //#pragma omp parallel for
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
        for (int si = i; si < limi; si++)
        {
          //#pragma GCC unroll 4
          if (limj < n && limk < n) {
            double ci0 = C[si+j*n];
            double ci1 = C[si+(j+1)*n]; 
            double ci2 = C[si+(j+2)*n];
            double ci3 = C[si+(j+3)*n];
            __m256d ymm00 = _mm256_i32gather_pd(&A[si+n*(k+0)], idx, 8);
            __m256d ymm01 = _mm256_loadu_pd(&B[k+0+n*(j+0)]); 
            __m256d ymm11 = _mm256_loadu_pd(&B[k+0+n*(j+1)]); 
            __m256d ymm21 = _mm256_loadu_pd(&B[k+0+n*(j+2)]); 
            __m256d ymm31 = _mm256_loadu_pd(&B[k+0+n*(j+3)]); 
            ci0 += hsum_double_avx(_mm256_mul_pd(ymm00, ymm01)); 
            ci1 += hsum_double_avx(_mm256_mul_pd(ymm00, ymm11)); 
            ci2 += hsum_double_avx(_mm256_mul_pd(ymm00, ymm21)); 
            ci3 += hsum_double_avx(_mm256_mul_pd(ymm00, ymm31)); 
            ymm00 = _mm256_i32gather_pd(&A[si+n*(k+4)], idx, 8);
            ymm01 = _mm256_loadu_pd(&B[k+4+n*(j+0)]); 
            ymm11 = _mm256_loadu_pd(&B[k+4+n*(j+1)]); 
            ymm21 = _mm256_loadu_pd(&B[k+4+n*(j+2)]); 
            ymm31 = _mm256_loadu_pd(&B[k+4+n*(j+3)]); 
            ci0 += hsum_double_avx(_mm256_mul_pd(ymm00, ymm01)); 
            ci1 += hsum_double_avx(_mm256_mul_pd(ymm00, ymm11)); 
            ci2 += hsum_double_avx(_mm256_mul_pd(ymm00, ymm21)); 
            ci3 += hsum_double_avx(_mm256_mul_pd(ymm00, ymm31)); 

            C[si+(j+0)*n] = ci0;
            C[si+(j+1)*n] = ci1;
            C[si+(j+2)*n] = ci2;
            C[si+(j+3)*n] = ci3;

            double ci4 = C[si+(j+4)*n];
            double ci5 = C[si+(j+5)*n]; 
            double ci6 = C[si+(j+6)*n];
            double ci7 = C[si+(j+7)*n];
            ymm00 = _mm256_i32gather_pd(&A[si+n*(k+0)], idx, 8);
            ymm01 = _mm256_loadu_pd(&B[k+0+n*(j+4)]); 
            ymm11 = _mm256_loadu_pd(&B[k+0+n*(j+5)]); 
            ymm21 = _mm256_loadu_pd(&B[k+0+n*(j+6)]); 
            ymm31 = _mm256_loadu_pd(&B[k+0+n*(j+7)]); 
            ci4 += hsum_double_avx(_mm256_mul_pd(ymm00, ymm01)); 
            ci5 += hsum_double_avx(_mm256_mul_pd(ymm00, ymm11)); 
            ci6 += hsum_double_avx(_mm256_mul_pd(ymm00, ymm21)); 
            ci7 += hsum_double_avx(_mm256_mul_pd(ymm00, ymm31)); 
            ymm00 = _mm256_i32gather_pd(&A[si+n*(k+4)], idx, 8);
            ymm01 = _mm256_loadu_pd(&B[k+4+n*(j+4)]); 
            ymm11 = _mm256_loadu_pd(&B[k+4+n*(j+5)]); 
            ymm21 = _mm256_loadu_pd(&B[k+4+n*(j+6)]); 
            ymm31 = _mm256_loadu_pd(&B[k+4+n*(j+7)]); 
            ci4 += hsum_double_avx(_mm256_mul_pd(ymm00, ymm01)); 
            ci5 += hsum_double_avx(_mm256_mul_pd(ymm00, ymm11)); 
            ci6 += hsum_double_avx(_mm256_mul_pd(ymm00, ymm21)); 
            ci7 += hsum_double_avx(_mm256_mul_pd(ymm00, ymm31)); 
            C[si+(j+4)*n] = ci4;
            C[si+(j+5)*n] = ci5;
            C[si+(j+6)*n] = ci6;
            C[si+(j+7)*n] = ci7;
          } else 
          for (int sj = j; sj < limj; sj++)
          {
            double cij = C[si+sj*n]; 
            int ssk; 
            __m256d ymm0; 
            __m256d ymm1; 
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
  __m256d ymm0, ymm1;
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
  //#pragma omp parallel for
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

// https://stackoverflow.com/questions/16737298/what-is-the-fastest-way-to-transpose-a-matrix-in-c
#define _MM256_TRANSPOSE4_PD(row0, row1, row2, row3) { \
  __m256d tmp3, tmp2, tmp1, tmp0, v; \
  tmp0 = _mm256_unpacklo_pd((row0), (row2)); \
  tmp2 = _mm256_unpacklo_pd((row1), (row3)); \
  tmp1 = _mm256_unpackhi_pd((row0), (row2)); \
  tmp3 = _mm256_unpackhi_pd((row1), (row3)); \
  tmp0 = _mm256_castsi256_pd(_mm256_permute4x64_epi64(_mm256_castpd_si256(tmp0), _MM_SHUFFLE(3,1,2,0))); \
  tmp1 = _mm256_castsi256_pd(_mm256_permute4x64_epi64(_mm256_castpd_si256(tmp1), _MM_SHUFFLE(3,1,2,0))); \
  tmp2 = _mm256_castsi256_pd(_mm256_permute4x64_epi64(_mm256_castpd_si256(tmp2), _MM_SHUFFLE(3,1,2,0))); \
  tmp3 = _mm256_castsi256_pd(_mm256_permute4x64_epi64(_mm256_castpd_si256(tmp3), _MM_SHUFFLE(3,1,2,0))); \
  row0 = _mm256_shuffle_pd(tmp0, tmp2, 0x0); \
  row2 = _mm256_shuffle_pd(tmp0, tmp2, 0xF); \
  row1 = _mm256_shuffle_pd(tmp1, tmp3, 0x0); \
  row3 = _mm256_shuffle_pd(tmp1, tmp3, 0xF); \
}

inline void load_buf_transpose(int n, int i, int j, double* M, __m256d buf[16]) {
  buf[0] = _mm256_loadu_pd(&M[i+j*n]);
  buf[1] = _mm256_loadu_pd(&M[i+(j+1)*n]);
  buf[2] = _mm256_loadu_pd(&M[i+(j+2)*n]);
  buf[3] = _mm256_loadu_pd(&M[i+(j+3)*n]);
  _MM256_TRANSPOSE4_PD(buf[0], buf[1], buf[2], buf[3]);
  buf[4] = _mm256_loadu_pd(&M[i+4+j*n]);
  buf[5] = _mm256_loadu_pd(&M[i+4+(j+1)*n]);
  buf[6] = _mm256_loadu_pd(&M[i+4+(j+2)*n]);
  buf[7] = _mm256_loadu_pd(&M[i+4+(j+3)*n]);
  _MM256_TRANSPOSE4_PD(buf[4], buf[5], buf[6], buf[7]);
  buf[8] = _mm256_loadu_pd(&M[i+(j+4)*n]);
  buf[9] = _mm256_loadu_pd(&M[i+(j+5)*n]);
  buf[10] = _mm256_loadu_pd(&M[i+(j+6)*n]);
  buf[11] = _mm256_loadu_pd(&M[i+(j+7)*n]);
  _MM256_TRANSPOSE4_PD(buf[8], buf[9], buf[10], buf[11]);
  buf[12] = _mm256_loadu_pd(&M[i+4+(j+4)*n]);
  buf[13] = _mm256_loadu_pd(&M[i+4+(j+5)*n]);
  buf[14] = _mm256_loadu_pd(&M[i+4+(j+6)*n]);
  buf[15] = _mm256_loadu_pd(&M[i+4+(j+7)*n]);
  _MM256_TRANSPOSE4_PD(buf[12], buf[13], buf[14], buf[15]);
}

inline void store_buf_transpose(int n, int i, int j, double* M, __m256d buf[16]) {
  _MM256_TRANSPOSE4_PD(buf[0], buf[1], buf[2], buf[3]);
  _mm256_storeu_pd(&M[i+j*n], buf[0]);
  _mm256_storeu_pd(&M[i+(j+1)*n], buf[1]);
  _mm256_storeu_pd(&M[i+(j+2)*n], buf[2]);
  _mm256_storeu_pd(&M[i+(j+3)*n], buf[3]);
  _MM256_TRANSPOSE4_PD(buf[4], buf[5], buf[6], buf[7]);
  _mm256_storeu_pd(&M[i+4+j*n], buf[4]);
  _mm256_storeu_pd(&M[i+4+(j+1)*n], buf[5]);
  _mm256_storeu_pd(&M[i+4+(j+2)*n], buf[6]);
  _mm256_storeu_pd(&M[i+4+(j+3)*n], buf[7]);
  _MM256_TRANSPOSE4_PD(buf[8], buf[9], buf[10], buf[11]);
  _mm256_storeu_pd(&M[i+(j+4)*n], buf[8]);
  _mm256_storeu_pd(&M[i+(j+5)*n], buf[9]);
  _mm256_storeu_pd(&M[i+(j+6)*n], buf[10]);
  _mm256_storeu_pd(&M[i+(j+7)*n], buf[11]);
  _MM256_TRANSPOSE4_PD(buf[12], buf[13], buf[14], buf[15]);
  _mm256_storeu_pd(&M[i+4+(j+4)*n], buf[12]);
  _mm256_storeu_pd(&M[i+4+(j+5)*n], buf[13]);
  _mm256_storeu_pd(&M[i+4+(j+6)*n], buf[14]);
  _mm256_storeu_pd(&M[i+4+(j+7)*n], buf[15]);
}

inline void store_buf_transpose_add(int n, int i, int j, double* M, __m256d buf[16]) {
  _MM256_TRANSPOSE4_PD(buf[0], buf[1], buf[2], buf[3]);
  _mm256_storeu_pd(&M[i+j*n], _mm256_add_pd(_mm256_loadu_pd(&M[i+(j+0)*n]),buf[0]));
  _mm256_storeu_pd(&M[i+(j+1)*n], _mm256_add_pd(_mm256_loadu_pd(&M[i+(j+1)*n]),buf[1]));
  _mm256_storeu_pd(&M[i+(j+2)*n], _mm256_add_pd(_mm256_loadu_pd(&M[i+(j+2)*n]),buf[2]));
  _mm256_storeu_pd(&M[i+(j+3)*n], _mm256_add_pd(_mm256_loadu_pd(&M[i+(j+3)*n]),buf[3]));
  _MM256_TRANSPOSE4_PD(buf[4], buf[5], buf[6], buf[7]);
  _mm256_storeu_pd(&M[i+4+j*n], _mm256_add_pd(_mm256_loadu_pd(&M[i+4+(j+0)*n]),buf[4]));
  _mm256_storeu_pd(&M[i+4+(j+1)*n], _mm256_add_pd(_mm256_loadu_pd(&M[i+4+(j+1)*n]),buf[5]));
  _mm256_storeu_pd(&M[i+4+(j+2)*n], _mm256_add_pd(_mm256_loadu_pd(&M[i+4+(j+2)*n]),buf[6]));
  _mm256_storeu_pd(&M[i+4+(j+3)*n], _mm256_add_pd(_mm256_loadu_pd(&M[i+4+(j+3)*n]),buf[7]));
  _MM256_TRANSPOSE4_PD(buf[8], buf[9], buf[10], buf[11]);
  _mm256_storeu_pd(&M[i+(j+4)*n], _mm256_add_pd(_mm256_loadu_pd(&M[i+(j+4)*n]),buf[8]));
  _mm256_storeu_pd(&M[i+(j+5)*n], _mm256_add_pd(_mm256_loadu_pd(&M[i+(j+5)*n]),buf[9]));
  _mm256_storeu_pd(&M[i+(j+6)*n], _mm256_add_pd(_mm256_loadu_pd(&M[i+(j+6)*n]),buf[10]));
  _mm256_storeu_pd(&M[i+(j+7)*n], _mm256_add_pd(_mm256_loadu_pd(&M[i+(j+7)*n]),buf[11]));
  _MM256_TRANSPOSE4_PD(buf[12], buf[13], buf[14], buf[15]);
  _mm256_storeu_pd(&M[i+4+(j+4)*n], _mm256_add_pd(_mm256_loadu_pd(&M[i+4+(j+4)*n]),buf[12]));
  _mm256_storeu_pd(&M[i+4+(j+5)*n], _mm256_add_pd(_mm256_loadu_pd(&M[i+4+(j+5)*n]),buf[13]));
  _mm256_storeu_pd(&M[i+4+(j+6)*n], _mm256_add_pd(_mm256_loadu_pd(&M[i+4+(j+6)*n]),buf[14]));
  _mm256_storeu_pd(&M[i+4+(j+7)*n], _mm256_add_pd(_mm256_loadu_pd(&M[i+4+(j+7)*n]),buf[15]));
}

inline void reset_buf(__m256d buf[16]) {
  for (int i = 0; i < 16; i++) {
    buf[i] = _mm256_setzero_pd();
  }
}

void square_dgemm_transpose(int n, double* A, double* B, double* C)
{
  // TODO: Implement the blocking optimization
  __m256i zero = _mm256_setzero_si256();
  __m256d zerod = _mm256_setzero_pd();
  __m256i mask1 = _mm256_setr_epi32(1,1,0,0,0,0,0,0);
  __m256i mask2 = _mm256_setr_epi32(1,1,1,1,0,0,0,0);
  __m256i mask3 = _mm256_setr_epi32(1,1,1,1,1,1,0,0);
  __m256i masks[4] = {zero, mask1, mask2, mask3};
  __m128i idx = _mm_setr_epi32(0, n, 2*n, 3*n);
  int s = 8; 
  int res = (n - (n/s)*s)%4;
  __m256i mask = (masks[res] != zero);
  __m256d maskd =  _mm256_castsi256_pd(mask);
  //double __attribute__((aligned(32))) abuf[64], cbuf[64]; 
  __m256d cbuf[16];
  //#pragma omp parallel for
  for (int j = 0; j < n; j = j+s)
  {
    int limj = j+s; 
    limj = limj < n ? limj:n; 
    for (int i = 0; i < n; i = i+s)
    {
      int limi = i+s; 
      limi = limi < n ? limi:n;
      if (limi < n && limj < n) {
        reset_buf(cbuf);
      }
      for (int k = 0; k < n; k = k+s)
      {
        int limk = k+s; 
        limk = limk < n ? limk:n; 
        if (limi < n && limj < n && limk < n) 
        {
          for (int si = i; si < i+s; si++)
          {
            __m256d ymm00 = _mm256_i32gather_pd(&A[si+n*(k+0)], idx, 8);
            __m256d ymm01 = _mm256_loadu_pd(&B[k+0+n*(j+0)]); 
            __m256d ymm11 = _mm256_loadu_pd(&B[k+0+n*(j+1)]); 
            __m256d ymm21 = _mm256_loadu_pd(&B[k+0+n*(j+2)]); 
            __m256d ymm31 = _mm256_loadu_pd(&B[k+0+n*(j+3)]);
            ymm01 = _mm256_mul_pd(ymm00, ymm01);
            ymm11 = _mm256_mul_pd(ymm00, ymm11);
            ymm21 = _mm256_mul_pd(ymm00, ymm21);
            ymm31 = _mm256_mul_pd(ymm00, ymm31);
            _MM256_TRANSPOSE4_PD(ymm01, ymm11, ymm21, ymm31);
            int _i = si - i;
            cbuf[_i] = _mm256_add_pd(cbuf[_i], _mm256_add_pd(_mm256_add_pd(ymm01,ymm11),_mm256_add_pd(ymm21,ymm31)));
            __m256d ymm10 = _mm256_i32gather_pd(&A[si+n*(k+4)], idx, 8);
            ymm01 = _mm256_loadu_pd(&B[k+4+n*(j+0)]); 
            ymm11 = _mm256_loadu_pd(&B[k+4+n*(j+1)]); 
            ymm21 = _mm256_loadu_pd(&B[k+4+n*(j+2)]); 
            ymm31 = _mm256_loadu_pd(&B[k+4+n*(j+3)]); 
            ymm01 = _mm256_mul_pd(ymm10, ymm01);
            ymm11 = _mm256_mul_pd(ymm10, ymm11);
            ymm21 = _mm256_mul_pd(ymm10, ymm21);
            ymm31 = _mm256_mul_pd(ymm10, ymm31);
            _MM256_TRANSPOSE4_PD(ymm01, ymm11, ymm21, ymm31);
            cbuf[_i] = _mm256_add_pd(cbuf[_i], _mm256_add_pd(_mm256_add_pd(ymm01,ymm11),_mm256_add_pd(ymm21,ymm31)));

            ymm01 = _mm256_loadu_pd(&B[k+0+n*(j+4)]); 
            ymm11 = _mm256_loadu_pd(&B[k+0+n*(j+5)]); 
            ymm21 = _mm256_loadu_pd(&B[k+0+n*(j+6)]); 
            ymm31 = _mm256_loadu_pd(&B[k+0+n*(j+7)]); 
            ymm01 = _mm256_mul_pd(ymm00, ymm01);
            ymm11 = _mm256_mul_pd(ymm00, ymm11);
            ymm21 = _mm256_mul_pd(ymm00, ymm21);
            ymm31 = _mm256_mul_pd(ymm00, ymm31);
            _MM256_TRANSPOSE4_PD(ymm01, ymm11, ymm21, ymm31);
            cbuf[_i+8] = _mm256_add_pd(cbuf[_i+8], _mm256_add_pd(_mm256_add_pd(ymm01,ymm11),_mm256_add_pd(ymm21,ymm31)));
            ymm01 = _mm256_loadu_pd(&B[k+4+n*(j+4)]); 
            ymm11 = _mm256_loadu_pd(&B[k+4+n*(j+5)]); 
            ymm21 = _mm256_loadu_pd(&B[k+4+n*(j+6)]); 
            ymm31 = _mm256_loadu_pd(&B[k+4+n*(j+7)]); 
            ymm01 = _mm256_mul_pd(ymm10, ymm01);
            ymm11 = _mm256_mul_pd(ymm10, ymm11);
            ymm21 = _mm256_mul_pd(ymm10, ymm21);
            ymm31 = _mm256_mul_pd(ymm10, ymm31);
            _MM256_TRANSPOSE4_PD(ymm01, ymm11, ymm21, ymm31);
            cbuf[_i+8] = _mm256_add_pd(cbuf[_i+8], _mm256_add_pd(_mm256_add_pd(ymm01,ymm11),_mm256_add_pd(ymm21,ymm31)));          
          }
        }
        else
        {
          for (int si = i; si < limi; si++)
            for (int sj = j; sj < limj; sj++)
            {
              double cij = C[si+sj*n]; 
              int ssk; 
              __m256d ymm0; 
              __m256d ymm1; 
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
      if (limi < n && limj < n) {
        store_buf_transpose_add(n, i, j, C, cbuf);
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

