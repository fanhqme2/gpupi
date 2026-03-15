#!/bin/bash
set -e

nvcc -O3 -std=c++17 -o batch_shift_test batch_shift_test.cu batch_shift.cu -lgmp -lcurand
