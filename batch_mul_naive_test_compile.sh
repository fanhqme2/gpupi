#!/bin/bash
set -e

# Compile batch_mul_naive_test.cu and batch_mul_naive.cu into batch_mul_naive_test
nvcc -O3 -o batch_mul_naive_test batch_mul_naive_test.cu batch_mul_naive.cu -lgmp
