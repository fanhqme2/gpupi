#pragma once

#include <stddef.h>
#include <stdint.h>

// Multiply N large integers by one host-side single-limb integer.
// A has L_a words with stride_A words per batch item.
// B is the shared uint32_t factor for the whole batch.
// C has L_c words with stride_C words per batch item.
// The mathematical product is computed over L_a + 1 limbs.
// If L_c is larger, the remaining high words are set to zero.
// If L_c is smaller, the high part is truncated.
// A, C, and workspace are device pointers.
// We must have L_a <= stride_A and L_c <= stride_C.
// workspace must be at least batch_mul_small_workspace_size(...) bytes
// when the returned size is non-zero.
void batch_mul_small(
    const uint32_t * A,
    uint32_t B,
    uint32_t * C,
    uint32_t * workspace,
    uint32_t N,
    uint32_t L_a,
    uint32_t L_c,
    uint32_t stride_A,
    uint32_t stride_C
);

// Compute the workspace size needed for batch_mul_small.
// Returns the size in bytes.
size_t batch_mul_small_workspace_size(
    uint32_t N,
    uint32_t L_a,
    uint32_t L_c
);
