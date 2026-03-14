#pragma once

#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime.h>

#include "batch_mul_ntt.h"

struct BatchMPContext {
    NTTPrecomputedTables ntt_tables;
    uint32_t *workspace;
    size_t workspace_size_bytes;
};

// Allocate and initialize precomputed tables used by batch arithmetic kernels.
// Returns nullptr on allocation or initialization failure.
BatchMPContext * batch_mp_init();

// Release all buffers owned by the context. Accepts nullptr.
void batch_mp_destroy(BatchMPContext * ctx);

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
