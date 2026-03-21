#!/usr/bin/env bash
set -euo pipefail

nvcc -O3 -std=c++17 -o convert_decimal_test \
    convert_decimal_test.cu \
    convert_decimal.cu \
    batch_decimal_naive.cu \
    batch_arith.cu \
    batch_mul_ntt.cu \
    batch_add.cu \
    batch_sub.cu \
    batch_shift_add.cu \
    batch_shift_sub.cu \
    batch_mul_small.cu \
    batch_exactdiv_small.cu \
    batch_add_small.cu \
    batch_sub_small.cu \
    batch_mul_naive.cu \
    batch_bitlength.cu \
    batch_shift.cu \
    -lgmp
