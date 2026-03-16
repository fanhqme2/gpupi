#!/bin/bash
set -e

nvcc -O3 -std=c++17 -o batch_shift_add_test batch_shift_add_test.cu batch_shift_add.cu -lgmp -lcurand
