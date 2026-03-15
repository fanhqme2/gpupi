#!/bin/bash
set -e

nvcc -O3 -std=c++17 -o batch_mul_small_test batch_mul_small_test.cu batch_mul_small.cu -lgmp -lcurand
