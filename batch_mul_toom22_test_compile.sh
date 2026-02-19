#!/bin/bash

# Compile batch_mul_toom22_test.cu, batch_mul_toom22.cu and batch_mul_direct.cu into batch_mul_toom22_test

nvcc -O3 -o batch_mul_toom22_test batch_mul_toom22_test.cu batch_mul_toom22.cu batch_mul_direct.cu -lgmp
