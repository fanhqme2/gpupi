#pragma once

#include <stdint.h>

struct NTTPrecomputedTables{
    uint3 * roots_table_lv1; // size: 65536
    uint3 * roots_table_lv2; // size: 65536
    uint3 * inv2n_table; // size: 33
};

void init_ntt_precomputed_tables(NTTPrecomputedTables * tables);

// multiply N large integers of L words each, using Number Theoretic Transform, and store the result in ret
// A : N * L, device memory pointer
// B : N * L, device memory pointer
// ret : N * (L * 2), device memory pointer
// workspace : device memory pointer for temporary storage, must be at least batch_mul_ntt_workspace_size(N, L) bytes
// we must have L * 2 <= 2^32, and N much smaller than 2^32
void batch_mul_ntt(uint32_t * A, uint32_t * B, uint32_t * ret, uint32_t * workspace, NTTPrecomputedTables tables, uint32_t N, uint32_t L);

// compute the workspace size needed for batch_mul_ntt
// returns the size in bytes
size_t batch_mul_ntt_workspace_size(uint32_t N, uint32_t L);
