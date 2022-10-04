#!/bin/zsh

export OMP_NUM_THREADS=1
export PYTHONPATH=../..

ENSEMBLE_SIZE=8
mpiexec -n $ENSEMBLE_SIZE python example_enkf.py
python ../plot_pickles.py ../../RESULTS_EXPERIMENTS

ENSEMBLE_SIZE=16
mpiexec -n $ENSEMBLE_SIZE python example_enkf.py
python ../plot_pickles.py ../../RESULTS_EXPERIMENTS

ENSEMBLE_SIZE=32
mpiexec -n $ENSEMBLE_SIZE python example_enkf.py
python ../plot_pickles.py ../../RESULTS_EXPERIMENTS
