#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "batch_shift.h"

namespace {

constexpr uint32_t kBlockThreads = 256u;
constexpr uint32_t kWordsPerThread = 2u;

__device__ __forceinline__ uint32_t load_limb_or_zero(
    const uint32_t * row,
    uint32_t idx,
    uint32_t L_in
) {
    return (idx < L_in) ? row[idx] : 0u;
}

__global__ __launch_bounds__(kBlockThreads, 2) void batch_shift_kernel(
    const uint32_t * __restrict__ A,
    uint32_t * __restrict__ B,
    uint32_t N,
    uint32_t L_in,
    uint32_t L_out,
    uint32_t stride_in,
    uint32_t stride_out,
    int32_t shift_bits,
    uint64_t total_output_limbs
) {
    const uint64_t thread_linear = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t thread_stride = (uint64_t)gridDim.x * blockDim.x;

    for (uint64_t base = thread_linear * kWordsPerThread;
         base < total_output_limbs;
         base += thread_stride * kWordsPerThread) {
#pragma unroll
        for (uint32_t item = 0; item < kWordsPerThread; ++item) {
            const uint64_t linear_idx = base + item;
            if (linear_idx >= total_output_limbs) {
                continue;
            }

            const uint32_t row_idx = (uint32_t)(linear_idx / L_out);
            const uint32_t out_idx = (uint32_t)(linear_idx - (uint64_t)row_idx * L_out);
            const uint32_t * a_row = A + (size_t)row_idx * stride_in;
            uint32_t * b_row = B + (size_t)row_idx * stride_out;

            if (shift_bits >= 0) {
                const uint32_t shift = (uint32_t)shift_bits;
                const uint32_t word_shift = shift >> 5;
                const uint32_t bit_shift = shift & 31u;
                uint32_t result = 0u;
                if (out_idx >= word_shift) {
                    const uint32_t src_lo = out_idx - word_shift;
                    if (bit_shift == 0u) {
                        result = load_limb_or_zero(a_row, src_lo, L_in);
                    } else {
                        const uint32_t lo = load_limb_or_zero(a_row, src_lo, L_in);
                        const uint32_t hi = (src_lo > 0u) ? load_limb_or_zero(a_row, src_lo - 1u, L_in) : 0u;
                        result = (lo << bit_shift) | (hi >> (32u - bit_shift));
                    }
                }
                b_row[out_idx] = result;
            } else {
                const uint32_t shift = (uint32_t)(-shift_bits);
                const uint32_t word_shift = shift >> 5;
                const uint32_t bit_shift = shift & 31u;
                const uint32_t src_lo = out_idx + word_shift;
                uint32_t result = 0u;
                if (src_lo < L_in) {
                    const uint32_t lo = load_limb_or_zero(a_row, src_lo, L_in);
                    if (bit_shift == 0u) {
                        result = lo;
                    } else {
                        const uint32_t hi = load_limb_or_zero(a_row, src_lo + 1u, L_in);
                        result = (lo >> bit_shift) | (hi << (32u - bit_shift));
                    }
                }
                b_row[out_idx] = result;
            }
        }
    }
}

}  // namespace

void batch_shift_bits(
    const uint32_t * A,
    uint32_t * B,
    uint32_t N,
    uint32_t L_in,
    uint32_t L_out,
    uint32_t stride_in,
    uint32_t stride_out,
    int32_t shift_bits
) {
    if (N == 0 || L_out == 0) {
        return;
    }

    const uint64_t total_output_limbs = (uint64_t)N * L_out;
    const uint64_t logical_threads = (total_output_limbs + kWordsPerThread - 1u) / kWordsPerThread;
    const uint32_t grid_x = (uint32_t)std::min<uint64_t>(
        (logical_threads + kBlockThreads - 1u) / kBlockThreads,
        65535u);
    batch_shift_kernel<<<grid_x, kBlockThreads>>>(
        A, B, N, L_in, L_out, stride_in, stride_out, shift_bits, total_output_limbs
    );
}
