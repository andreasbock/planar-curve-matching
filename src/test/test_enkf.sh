#!/bin/zsh

ENSEMBLE_SIZE=10
mpiexec -n $ENSEMBLE_SIZE python test_enkf.py
