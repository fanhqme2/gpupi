#!/bin/bash
set -e

nvcc -O3 -std=c++17 -o batch_shift_sub_test batch_shift_sub_test.cu batch_shift_sub.cu -lgmp -lcurand
