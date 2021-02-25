#!/bin/bash -l

#SBATCH --job-name=matrixmult
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --output=matrixmult-%j.out
#SBATCH --error=matrixmult-%j.err

# load modules
if command -v module 1>/dev/null 2>&1; then
   module load gcc mkl
fi

export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10

echo "==== benchmark-naive ======================"
./benchmark-naive | tee timing_basic_dgemm.data
echo
echo "==== benchmark-blas ======================="
./benchmark-blas | tee timing_blas_dgemm.data
echo
echo "==== benchmark-blocked ===================="
./benchmark-blocked | tee timing_blocked_dgemm.data

echo
echo "==== plot results ========================="
gnuplot timing.gp
