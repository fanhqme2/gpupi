#pragma once

#include <stddef.h>
#include <stdint.h>

// Compute the maximum bitlength among N large integers.
// A has L words with stride_A words per batch item.
// A and workspace are device pointers.
// We must have L <= stride_A.
// workspace must be at least batch_bitlength_workspace_size(...) bytes
// when the returned size is non-zero.
uint32_t batch_bitlength_max(
    const uint32_t * A,
    uint32_t * workspace,
    uint32_t N,
    uint32_t L,
    uint32_t stride_A
);

// Compute the maximum non-zero limb count among N large integers.
// Returns the smallest limb length that can contain every value in A.
uint32_t batch_limblength_max(
    const uint32_t * A,
    uint32_t * workspace,
    uint32_t N,
    uint32_t L,
    uint32_t stride_A
);

// Compute the workspace size needed for batch_bitlength_max.
// Returns the size in bytes.
size_t batch_bitlength_workspace_size(
    uint32_t N,
    uint32_t L
);
