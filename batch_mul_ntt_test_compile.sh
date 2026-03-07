#!/bin/bash

# Compile batch_mul_ntt_test.cu and batch_mul_ntt.cu into batch_mul_ntt_test

nvcc -O3 -o batch_mul_ntt_test batch_mul_ntt_test.cu batch_mul_ntt.cu -lgmp
