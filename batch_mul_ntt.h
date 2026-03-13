#pragma once

#include <stdint.h>

struct NTTPrecomputedTables{
    uint3 * roots_table_lv1; // size: 65536
    uint3 * roots_table_lv2; // size: 65536
    uint3 * inv2n_table; // size: 33
};

void init_ntt_precomputed_tables(NTTPrecomputedTables * tables);

// multiply N pairs of large integers using Number Theoretic Transform:
// A has L_a words with stride_A words per batch item.
// B has L_b words with stride_B words per batch item.
// output has L_a + L_b words with stride_ret words per batch item.
// A : N * stride_A, device memory pointer
// B : N * stride_B, device memory pointer
// ret : N * stride_ret, device memory pointer
// workspace : device memory pointer for temporary storage,
//             must be at least batch_mul_ntt_workspace_size(N, L_a, L_b) bytes
// we must have L_a <= stride_A, L_b <= stride_B, L_a + L_b <= stride_ret,
// ((size_t)L_a) + L_b <= 2^32, and N much smaller than 2^32
void batch_mul_ntt(
    uint32_t * A,
    uint32_t * B,
    uint32_t * ret,
    uint32_t * workspace,
    NTTPrecomputedTables tables,
    uint32_t N,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t stride_A,
    uint32_t stride_B,
    uint32_t stride_ret
);

// compute the workspace size needed for batch_mul_ntt
// returns the size in bytes
size_t batch_mul_ntt_workspace_size(uint32_t N, uint32_t L_a, uint32_t L_b);
