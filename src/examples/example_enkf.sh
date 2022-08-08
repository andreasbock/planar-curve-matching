#!/bin/zsh

ENSEMBLE_SIZE=16
export OMP_NUM_THREADS=1
export PYTHONPATH=../..

mpiexec -n $ENSEMBLE_SIZE python example_enkf.py
python ../plot_pickles.py ../../RESULTS_EXAMPLES_ENKF/*
