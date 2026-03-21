#pragma once

#include <stdint.h>

constexpr int BATCH_DECIMAL_NAIVE_L_LIMB_MAX = 2047;

// Convert N fixed-point binary fractions to decimal digits.
// Each row of A stores an unsigned value X in little-endian 32-bit limbs.
// The interpreted fraction is X / 2^(32 * L_limb), so inputs should satisfy
// 0 <= X < 2^(32 * L_limb).
//
// A has L_limb words with stride_A words per batch item.
// B receives L_dec ASCII decimal digits per item with stride_B bytes per item.
// Digits are written least-significant-first, without a decimal point or
// terminator. For example, X = 2^(32 * L_limb - 1) produces "...0005" in the
// mathematical decimal expansion but writes "000...005" with the 5 at index
// L_dec - 1 only when enough digits are requested; the first output byte is the
// 10^-L_dec place. More directly, 1/2 with L_dec = 6 writes "000005".
//
// A and B must be device pointers. We must have:
// L_limb <= stride_A, L_dec <= stride_B,
// and L_limb <= BATCH_DECIMAL_NAIVE_L_LIMB_MAX.
void batch_decimal_naive(
    const uint32_t * A,
    char * B,
    uint32_t N,
    int L_limb,
    uint32_t stride_A,
    int L_dec,
    uint32_t stride_B
);
