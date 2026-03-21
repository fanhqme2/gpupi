#pragma once

#include <stdint.h>

#include "batch_arith.h"

struct ConvertDecimalPowerTable {
    BatchMPArray * powers;
    uint32_t levels;
    uint32_t leaf_digits;
};

// Build device-side powers 5^(leaf_digits << level) for level in [0, levels).
// Each power is stored as a single right-aligned row and can be broadcast by
// setting stride = 0 in a BatchMPArray view passed to batch_mp_mul.
cudaError_t convert_decimal_precompute_powers(
    BatchMPContext * ctx,
    ConvertDecimalPowerTable * table,
    uint32_t leaf_digits,
    uint32_t levels
);

void convert_decimal_release_powers(ConvertDecimalPowerTable * table);

// Convert one right-aligned fixed-point fraction X / 2^input_bits to
// total_digits decimal digits using balanced equal splits until each leaf has
// leaf_digits digits. total_digits must equal leaf_digits << levels for some
// integer levels, and the supplied power table must match that shape.
//
// input.batch_size must be 1. output_digits must be a device pointer with at
// least total_digits bytes. Digits are written in natural order, so
// output_digits[0] is the 10^-1 place.
cudaError_t convert_decimal_equal_split(
    BatchMPContext * ctx,
    BatchMPArray input,
    uint32_t input_bits,
    uint32_t total_digits,
    uint32_t leaf_digits,
    const ConvertDecimalPowerTable * table,
    char * output_digits
);
