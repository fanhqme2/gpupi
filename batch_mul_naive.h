#pragma once

#include <stdint.h>

const int BATCH_MUL_NAIVE_L_MAX = 1024;

// multiply N pairs of large integers using the School Book algorithm.
// A has L_a words with stride_A words per batch item.
// B has L_b words with stride_B words per batch item.
// output has L_a + L_b words with stride_ret words per batch item.
// A : N * stride_A, device memory pointer
// B : N * stride_B, device memory pointer
// ret : N * stride_ret, device memory pointer
// we must have L_a <= stride_A, L_b <= stride_B, L_a + L_b <= stride_ret,
// we must have L_a + L_b <= BATCH_MUL_NAIVE_L_MAX = 1024
void batch_mul_naive(
    uint32_t * A,
    uint32_t * B,
    uint32_t * ret,
    int N,
    int L_a,
    int L_b,
    int stride_A,
    int stride_B,
    int stride_ret
);