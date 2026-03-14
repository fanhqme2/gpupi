#!/bin/bash

nvcc -O3 -std=c++17 -o batch_arith_test batch_arith_test.cu batch_arith.cu batch_mul_ntt.cu batch_mul_naive.cu -lgmp
