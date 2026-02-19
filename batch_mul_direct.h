#pragma once

#include <stdint.h>

const int BATCH_MUL_DIRECT_L_MAX = 64; // at most 64 words (2048 bits) per integer
// multiply N large integers of L words each, using schoolbook multiplication, and store the result in ret
// A : N * L, device memory pointer
// B : N * L, device memory pointer
// ret : N * (L * 2), device memory pointer
// launch one thread per instance. Each block contains 16 threads.
void batch_mul_direct(uint32_t * A, uint32_t * B, uint32_t * ret, int N, int L);