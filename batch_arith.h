#pragma once

#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime.h>

struct BatchMPContext;

// Allocate and initialize precomputed tables used by batch arithmetic kernels.
// Returns nullptr on allocation or initialization failure.
BatchMPContext * batch_mp_init();

// Release all buffers owned by the context. Accepts nullptr.
void batch_mp_destroy(BatchMPContext * ctx);

// Return the currently allocated shared workspace size in bytes. Accepts nullptr.
size_t batch_mp_workspace_size(const BatchMPContext * ctx);

// Multiply N pairs of large integers.
// Uses the naive kernel when L_a + L_b <= 1024, otherwise uses the NTT kernel.
// Returns a CUDA error code for context growth or launch validation failures.
cudaError_t batch_mp_mul(
    BatchMPContext * ctx,
    uint32_t * A,
    uint32_t * B,
    uint32_t * ret,
    uint32_t N,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t stride_A,
    uint32_t stride_B,
    uint32_t stride_ret
);

// Add N pairs of large integers and write the truncated result into C.
cudaError_t batch_mp_add(
    BatchMPContext * ctx,
    uint32_t * A,
    uint32_t * B,
    uint32_t * C,
    uint32_t N,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t L_c,
    uint32_t stride_A,
    uint32_t stride_B,
    uint32_t stride_C
);

// Subtract N pairs of large integers modulo 2^(32 * L_c).
cudaError_t batch_mp_sub(
    BatchMPContext * ctx,
    uint32_t * A,
    uint32_t * B,
    uint32_t * C,
    uint32_t N,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t L_c,
    uint32_t stride_A,
    uint32_t stride_B,
    uint32_t stride_C
);

// Shift N unsigned multi-limb integers by a signed bit count.
cudaError_t batch_mp_shift_bits(
    BatchMPContext * ctx,
    const uint32_t * A,
    uint32_t * B,
    uint32_t N,
    uint32_t L_in,
    uint32_t L_out,
    uint32_t stride_in,
    uint32_t stride_out,
    int32_t shift_bits
);

// Compute the maximum bitlength among N large integers.
cudaError_t batch_mp_bitlength_max(
    BatchMPContext * ctx,
    const uint32_t * A,
    uint32_t N,
    uint32_t L,
    uint32_t stride_A,
    uint32_t * result
);
