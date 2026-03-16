#pragma once

#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime.h>

struct BatchMPContext;

struct BatchMPArray {
    uint32_t *data;
    uint32_t length;
    uint32_t batch_size;
    uint32_t stride;

    cudaError_t compact(BatchMPContext * ctx);
};

// Allocate device storage for batch_size values with the given stride.
// If stride is zero, length is used as the stride.
// On failure, the returned array has data == nullptr.
BatchMPArray batch_mp_array_create(
    uint32_t batch_size,
    uint32_t length,
    uint32_t stride = 0
);

// Release the device storage owned by the array and reset its fields to zero.
void batch_mp_array_release(BatchMPArray array);

// Allocate and initialize precomputed tables used by batch arithmetic kernels.
// Returns nullptr on allocation or initialization failure.
BatchMPContext * batch_mp_init();

// Release all buffers owned by the context. Accepts nullptr.
void batch_mp_destroy(BatchMPContext * ctx);

// Return the currently allocated shared workspace size in bytes. Accepts nullptr.
size_t batch_mp_workspace_size(const BatchMPContext * ctx);

cudaError_t batch_mp_ensure_workspace(BatchMPContext *ctx, size_t required_bytes);

// Multiply N pairs of large integers.
// Uses the naive kernel when L_a + L_b <= 1024, otherwise uses the NTT kernel.
// Returns cudaErrorInvalidValue if the batch dimensions mismatch or if C.length != A.length + B.length.
cudaError_t batch_mp_mul(
    BatchMPContext * ctx,
    BatchMPArray A,
    BatchMPArray B,
    BatchMPArray C
);

// Multiply N large integers by one host-side single-limb integer and write the truncated result into C.
// Returns cudaErrorInvalidValue if the batch dimensions mismatch.
cudaError_t batch_mp_mul_small(
    BatchMPContext * ctx,
    BatchMPArray A,
    uint32_t B,
    BatchMPArray C
);

// Add N pairs of large integers and write the truncated result into C.
// Returns cudaErrorInvalidValue if the batch dimensions mismatch.
cudaError_t batch_mp_add(
    BatchMPContext * ctx,
    BatchMPArray A,
    BatchMPArray B,
    BatchMPArray C
);

// Add one host-side single-limb integer to N large integers and write the truncated result into C.
// Returns cudaErrorInvalidValue if the batch dimensions mismatch.
cudaError_t batch_mp_add_small(
    BatchMPContext * ctx,
    BatchMPArray A,
    uint32_t B,
    BatchMPArray C
);

// Compute C = (A << shift) + B and write the truncated result into C.
// C must not alias A or B.
// Returns cudaErrorInvalidValue if the batch dimensions mismatch or if C aliases A or B.
cudaError_t batch_mp_shift_add(
    BatchMPContext * ctx,
    BatchMPArray A,
    BatchMPArray B,
    uint32_t shift,
    BatchMPArray C
);

// Subtract N pairs of large integers modulo 2^(32 * C.length).
// Returns cudaErrorInvalidValue if the batch dimensions mismatch.
cudaError_t batch_mp_sub(
    BatchMPContext * ctx,
    BatchMPArray A,
    BatchMPArray B,
    BatchMPArray C
);

// Subtract one host-side single-limb integer from N large integers modulo 2^(32 * C.length).
// Returns cudaErrorInvalidValue if the batch dimensions mismatch.
cudaError_t batch_mp_sub_small(
    BatchMPContext * ctx,
    BatchMPArray A,
    uint32_t B,
    BatchMPArray C
);

// Shift N unsigned multi-limb integers by a signed bit count.
// Returns cudaErrorInvalidValue if the batch dimensions mismatch.
cudaError_t batch_mp_shift_bits(
    BatchMPContext * ctx,
    BatchMPArray A,
    BatchMPArray B,
    int32_t shift_bits
);

// Compute the maximum bitlength among N large integers.
cudaError_t batch_mp_bitlength_max(
    BatchMPContext * ctx,
    BatchMPArray A,
    uint32_t * result
);

// Compute the maximum non-zero limb count among N large integers.
cudaError_t batch_mp_limblength_max(
    BatchMPContext * ctx,
    BatchMPArray A,
    uint32_t * result
);
