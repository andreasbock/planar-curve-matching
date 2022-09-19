#!/bin/zsh

export OMP_NUM_THREADS=1
export PYTHONPATH=../..

ENSEMBLE_SIZE=4
mpiexec -n $ENSEMBLE_SIZE python experiments_convergence.py

python ../plot_pickles.py ../../RESULTS_CONVERGENCE
