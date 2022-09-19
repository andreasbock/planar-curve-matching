#!/bin/zsh

export OMP_NUM_THREADS=1
export PYTHONPATH=../..

ENSEMBLE_SIZE=16
mpiexec -n $ENSEMBLE_SIZE python experiments_observation_noise.py
python ../plot_pickles.py ../../RESULTS_EXPERIMENTS_OBSERVATIONS
