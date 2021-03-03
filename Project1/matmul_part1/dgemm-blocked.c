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

const char* dgemm_desc = "Blocked, three-loop dgemm.";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */    
void square_dgemm (int n, double* A, double* B, double* C)
{
  // TODO: Implement the blocking optimization
  int s = 32; 
  int b = n/s+1; 
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
              cij += A[i*s+si+n*(s*k+sk)]*B[k*s+sk+n*(sj+s*j)]; 
            }
            C[s*i+si+n*(sj+s*j)]=cij;   
          }
        } 
      }
    }
  }
}



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

