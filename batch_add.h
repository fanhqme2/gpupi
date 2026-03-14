#pragma once

#include <stddef.h>
#include <stdint.h>

// Add N pairs of large integers.
// A has L_a words with stride_A words per batch item.
// B has L_b words with stride_B words per batch item.
// C has L_c words with stride_C words per batch item.
// The mathematical sum is computed over max(L_a, L_b) + 1 limbs.
// If L_c is larger, the remaining high words are set to zero.
// If L_c is smaller, the high part is truncated.
// A, B, C, and workspace are device pointers.
// We must have L_a <= stride_A, L_b <= stride_B, L_c <= stride_C.
// workspace must be at least batch_add_simple_workspace_size(...) bytes
// when the returned size is non-zero.
void batch_add_simple(
    uint32_t * A,
    uint32_t * B,
    uint32_t * C,
    uint32_t * workspace,
    uint32_t N,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t L_c,
    uint32_t stride_A,
    uint32_t stride_B,
    uint32_t stride_C
);

// Compute the workspace size needed for batch_add_simple.
// Returns the size in bytes.
size_t batch_add_simple_workspace_size(
    uint32_t N,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t L_c
);
