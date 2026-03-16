#include <assert.h>

#include <cuda_runtime.h>

#include <algorithm>

#include "batch_exactdiv_small.h"

namespace {

constexpr uint32_t kTileDim = 32u;
constexpr uint32_t kMaxLimbs = 128u;
constexpr uint32_t kDirectThreshold = 8u;

uint32_t invert_odd_u32(uint32_t value) {
    assert((value & 1u) != 0u);
    uint32_t inv = value;
    inv *= 2u - value * inv;
    inv *= 2u - value * inv;
    inv *= 2u - value * inv;
    inv *= 2u - value * inv;
    inv *= 2u - value * inv;
    return inv;
}

__global__ void batch_exactdiv_small_kernel(
    uint32_t * A,
    uint32_t B,
    uint32_t B_inv,
    uint32_t N,
    uint32_t L,
    uint32_t stride_A
) {
    __shared__ uint32_t tile[kTileDim * kTileDim];

    const uint32_t lane = threadIdx.x;
    const uint32_t block_stride = gridDim.x * kTileDim;

    for (uint32_t base_idx = blockIdx.x * kTileDim; base_idx < N; base_idx += block_stride) {
        const uint32_t active = min(kTileDim, N - base_idx);
        uint32_t carry = 0u;

        for (uint32_t limb_base = 0; limb_base < L; limb_base += kTileDim) {
            const uint32_t chunk_len = min(kTileDim, L - limb_base);

            for (uint32_t row = 0; row < active; ++row) {
                uint32_t word = 0u;
                if (lane < chunk_len) {
                    word = A[(size_t)(base_idx + row) * stride_A + limb_base + lane];
                }
                tile[lane * kTileDim + row] = word;
            }
            __syncthreads();

            if (lane < active) {
                #pragma unroll
                for (uint32_t j = 0; j < kTileDim; ++j) {
                    if (j >= chunk_len) {
                        break;
                    }
                    const uint32_t a_limb = tile[j * kTileDim + lane];
                    const uint32_t q_limb = (a_limb - carry) * B_inv;
                    const uint64_t prod = (uint64_t)q_limb * (uint64_t)B + (uint64_t)carry;
                    tile[j * kTileDim + lane] = q_limb;
                    carry = (uint32_t)(prod >> 32);
                }
            }
            __syncthreads();

            for (uint32_t row = 0; row < active; ++row) {
                if (lane < chunk_len) {
                    A[(size_t)(base_idx + row) * stride_A + limb_base + lane] = tile[lane * kTileDim + row];
                }
            }
            __syncthreads();
        }
    }
}

template<int THREADS>
__global__ void batch_exactdiv_small_direct_kernel(
    uint32_t * A,
    uint32_t B,
    uint32_t B_inv,
    uint32_t N,
    uint32_t L,
    uint32_t stride_A
) {
    for (uint32_t idx0 = blockIdx.x * blockDim.x; idx0 < N; idx0 += gridDim.x * blockDim.x) {
        const uint32_t idx = idx0 + threadIdx.x;
        if (idx >= N) {
            continue;
        }

        uint32_t * row = A + (size_t)idx * stride_A;
        uint32_t carry = 0u;

        #pragma unroll
        for (uint32_t limb = 0; limb < kDirectThreshold; ++limb) {
            if (limb >= L) {
                break;
            }
            const uint32_t a_limb = row[limb];
            const uint32_t q_limb = (a_limb - carry) * B_inv;
            const uint64_t prod = (uint64_t)q_limb * (uint64_t)B + (uint64_t)carry;
            row[limb] = q_limb;
            carry = (uint32_t)(prod >> 32);
        }
    }
}

}  // namespace

void batch_exactdiv_small(
    uint32_t * A,
    uint32_t B,
    uint32_t N,
    uint32_t L,
    uint32_t stride_A
) {
    if (N == 0 || L == 0) {
        return;
    }

    assert(A != nullptr);
    assert(B != 0u);
    assert((B & 1u) != 0u);
    assert(L <= kMaxLimbs);
    assert(L <= stride_A);

    const uint32_t B_inv = invert_odd_u32(B);
    if (L <= kDirectThreshold) {
        const uint32_t threads_per_block = 256u;
        const uint32_t num_blocks = std::min<uint32_t>((N + threads_per_block - 1u) / threads_per_block, 65535u);
        batch_exactdiv_small_direct_kernel<256><<<num_blocks, threads_per_block>>>(A, B, B_inv, N, L, stride_A);
        return;
    }

    const uint32_t num_blocks = std::min<uint32_t>((N + kTileDim - 1u) / kTileDim, 65535u);
    batch_exactdiv_small_kernel<<<num_blocks, kTileDim>>>(A, B, B_inv, N, L, stride_A);
}
