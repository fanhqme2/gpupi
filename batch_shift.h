#pragma once

#include <stdint.h>

// Shift N unsigned multi-limb integers by a signed bit count.
// Positive shift_bits performs a left shift, negative shift_bits performs a right shift.
// A has L_in words with stride_in words per batch item.
// B has L_out words with stride_out words per batch item.
// If the shifted result does not fit in L_out words, the high part is truncated.
// If L_out is larger than the shifted result, the remaining high words are set to zero.
// A and B are device pointers. We must have L_in <= stride_in and L_out <= stride_out.
void batch_shift_bits(
    const uint32_t * A,
    uint32_t * B,
    uint32_t N,
    uint32_t L_in,
    uint32_t L_out,
    uint32_t stride_in,
    uint32_t stride_out,
    int32_t shift_bits
);

inline void batch_shift_left(
    const uint32_t * A,
    uint32_t * B,
    uint32_t N,
    uint32_t L_in,
    uint32_t L_out,
    uint32_t stride_in,
    uint32_t stride_out,
    uint32_t shift_bits
) {
    batch_shift_bits(A, B, N, L_in, L_out, stride_in, stride_out, (int32_t)shift_bits);
}

inline void batch_shift_right(
    const uint32_t * A,
    uint32_t * B,
    uint32_t N,
    uint32_t L_in,
    uint32_t L_out,
    uint32_t stride_in,
    uint32_t stride_out,
    uint32_t shift_bits
) {
    batch_shift_bits(A, B, N, L_in, L_out, stride_in, stride_out, -(int32_t)shift_bits);
}
