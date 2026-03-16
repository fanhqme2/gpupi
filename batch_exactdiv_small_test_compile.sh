#!/bin/bash
set -e

nvcc -O3 -std=c++17 -o batch_exactdiv_small_test batch_exactdiv_small_test.cu batch_exactdiv_small.cu -lgmp
