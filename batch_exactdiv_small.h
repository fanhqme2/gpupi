#pragma once

#include <stdint.h>

// Divide N large integers in place by one shared odd uint32_t divisor.
// A has L words with stride_A words per batch item.
// Division is exact for every batch item and proceeds modulo 2^(32 * L).
// A is a device pointer. We must have L <= 128 and L <= stride_A.
// B must be odd and non-zero.
void batch_exactdiv_small(
    uint32_t * A,
    uint32_t B,
    uint32_t N,
    uint32_t L,
    uint32_t stride_A
);
