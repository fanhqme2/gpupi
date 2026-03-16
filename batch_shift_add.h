#pragma once

#include <stddef.h>
#include <stdint.h>

// Compute C = (A << shift) + B for N pairs of large integers.
// A has L_a words with stride_A words per batch item.
// B has L_b words with stride_B words per batch item.
// C has L_c words with stride_C words per batch item.
// shift is a host-side bit count shared by the whole batch.
// The mathematical sum is computed over max(shifted_len(A), L_b) + 1 limbs,
// where shifted_len(A) is the limb length of A << shift.
// If L_c is larger, the remaining high words are set to zero.
// If L_c is smaller, the high part is truncated.
// A, B, C, and workspace are device pointers.
// We must have L_a <= stride_A, L_b <= stride_B, and L_c <= stride_C.
// In-place operation is not supported: C must differ from A and B.
// workspace must be at least batch_shift_add_simple_workspace_size(...) bytes
// when the returned size is non-zero.
void batch_shift_add_simple(
    const uint32_t * A,
    const uint32_t * B,
    uint32_t * C,
    uint32_t * workspace,
    uint32_t shift,
    uint32_t N,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t L_c,
    uint32_t stride_A,
    uint32_t stride_B,
    uint32_t stride_C
);

// Compute the workspace size needed for batch_shift_add_simple.
// Returns the size in bytes.
size_t batch_shift_add_simple_workspace_size(
    uint32_t N,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t L_c,
    uint32_t shift
);
