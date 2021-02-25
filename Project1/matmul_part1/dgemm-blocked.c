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
  int s = 10; 
    for (int i = 0; i < n/s; i = i+s)
    {
      for (int j = 0; j < n/s; j = j+s)
      {
        double cij = C[i+j*n]; 
        for (int k = 0; k < n/s; k = k+s)
        {
          cij += A[i+k*n]*B[k+j*n]; 
          C[i+j*n] = cij; 
        }
      }
    }
  }





/* 
void square_dgemm_blocked (int n, double* A, double* B, double* C){
  for (int si = 1; si < 1025; si = si*2)
  {
    square_dgemm(n, si, A, B, C); 
  }
  
}

*/ 

