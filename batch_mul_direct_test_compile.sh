#!/bin/bash

# Compile batch_mul_direct_test.cu and batch_mul_direct.cu into batch_mul_direct_test

nvcc -O3 -o batch_mul_direct_test batch_mul_direct_test.cu batch_mul_direct.cu -lgmp

