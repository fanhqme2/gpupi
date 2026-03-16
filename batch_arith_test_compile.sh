#!/bin/bash

nvcc -O3 -std=c++17 -o batch_arith_test \
    batch_arith_test.cu \
    batch_arith.cu \
    batch_add.cu \
    batch_add_small.cu \
    batch_shift_add.cu \
    batch_mul_small.cu \
    batch_sub.cu \
    batch_sub_small.cu \
    batch_shift.cu \
    batch_bitlength.cu \
    batch_mul_ntt.cu \
    batch_mul_naive.cu \
    -lgmp
