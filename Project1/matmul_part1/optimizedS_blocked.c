#include <stdlib.h>
#include <stdio.h>
#ifdef GETTIMEOFDAY
#include <sys/time.h>
#else
#include <time.h>
#endif


double wall_time ()
{
#ifdef GETTIMEOFDAY
  struct timeval t;
  gettimeofday (&t, NULL);
  return 1.*t.tv_sec + 1.e-6*t.tv_usec;
#else
  struct timespec t;
  clock_gettime (CLOCK_MONOTONIC, &t);
  return 1.*t.tv_sec + 1.e-9*t.tv_nsec;
#endif

}

int main (int argc, char **argv)
{
    /* use n = 769(the largest test size in benchmark.c) as the test size in finding the optimzied S
    in blocked-algorithm*/
    int n = 769; 
    double* buf = NULL; 
    buf = (double*) malloc (3*n*n*sizeof(double)); 
    
    double* A = buf+0; 
    double* B = A + n*n; 
    double* C = B + n*n; 

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i+n*j] = 2*drand48()-1; 
            B[i+n*j] = 2*drand48()-1; 
            C[i+n*j] = 2*drand48()-1; 
        }
        
    }

    double seconds = -1.0; 
    int s[8] = {2, 4, 8, 16, 32, 64, 128, 256}; 

    for (int is = 0; is < 8; is++)
    {
        seconds = -wall_time(); 
        int ss = s[is]; 
        int b = n/ss+1; 
        for (int i = 0; i < b; i++)
        {
            for (int j = 0; j < b; j++)
            {
                for (int  si = 0; si < ss && (si+ss*i) < n; si++)
                {
                    for (int sj = 0; sj < ss && (sj+ss*j)<n ; sj++)
                    {
                        double cij = C[ss*i+si+n*(ss*j+sj)]; 
                        for (int k = 0; k < b; k++)
                        {
                            for (int sk = 0; sk < ss && (sk+ss*k)<n; sk++)
                            {
                                cij += A[i*ss+si+n*(ss*k+sk)]*B[k*ss+sk+n*(sj+ss*j)]; 
                            }
                            
                        }
                        C[ss*i+si+n*(sj+ss*j)]=cij; 
                     
                    }
                    
                }
                
            }
            
        }
        seconds += wall_time(); 
        printf("%f ", seconds);      
           
    }
}
