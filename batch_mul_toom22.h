#pragma once

#include <stdint.h>

const int BATCH_MUL_TOOM22_L_MAX = 498; // at most 498 words (15936 bits) per integer
const int BATCH_MUL_TOOM22_THREADS_PER_BLOCK = 16; // tunable constant

// multiply N large integers of L words each, using Toom-Cook 2-way multiplication, and store the result in ret
// A : N * L, device memory pointer
// B : N * L, device memory pointer
// ret : N * (L * 2), device memory pointer
// workspace : device memory pointer for temporary storage, must be at least batch_mul_toom22_workspace_size(N, L) bytes
void batch_mul_toom22(uint32_t * A, uint32_t * B, uint32_t * ret, uint32_t * workspace, int N, int L);

// compute the workspace size needed for batch_mul_toom22
// returns the size in bytes
size_t batch_mul_toom22_workspace_size(int N, int L);
