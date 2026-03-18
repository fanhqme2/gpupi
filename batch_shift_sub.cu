#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/warp>

#include <algorithm>

#include "batch_mul_addsub_asm.h"
#include "batch_mul_addsub_warp.h"
#include "batch_shift_sub.h"

namespace {

constexpr uint32_t kChunkSize = 2048;

__device__ __forceinline__ ushort2 combine_borrow_summary(ushort2 a, ushort2 b) {
    ushort compound = a.y + b.x;
    a.x += compound >> 2;
    if ((compound & 3) == 3) {
        a.y = b.y;
    } else {
        a.y = 0;
    }
    return a;
}

__device__ __forceinline__ void batch_shift_sub_64_all_warp(
    uint32_t & r0_value,
    uint32_t & r1_value,
    uint32_t & c0_value,
    uint32_t & c1_value,
    uint32_t borrow_prop[]
) {
    const unsigned int warp_mask = 0xffffffffu;
    r0_value = sub_cc(r0_value, c0_value);
    r1_value = subc_cc(r1_value, c1_value);

    uint32_t borrow_state = -subc(0, 0);
    sub_cc(r0_value, 1);
    subc_cc(r1_value, 0);
    borrow_state = (borrow_state << 1) - subc(0, 0);

    for (int delta = 1; delta < 32; delta *= 2) {
        uint32_t prev_borrow = __shfl_up_sync(warp_mask, borrow_state, delta, 32);
        if (borrow_state == 1u) {
            borrow_state = prev_borrow;
        }
    }
    if (threadIdx.x == 31) {
        borrow_prop[threadIdx.y] = borrow_state;
    }
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x == 0) {
        if (borrow_prop[0] == 1u) {
            borrow_prop[0] = 0u;
        }
        for (int i = 1; i < blockDim.y - 1; ++i) {
            if (borrow_prop[i] == 1u) {
                borrow_prop[i] = borrow_prop[i - 1];
            }
        }
    }

    __syncthreads();
    if (borrow_state == 1u) {
        if (threadIdx.y > 0) {
            borrow_state = borrow_prop[threadIdx.y - 1];
        } else {
            borrow_state = 0u;
        }
    }
    borrow_state = __shfl_up_sync(warp_mask, borrow_state, 1, 32);
    if (threadIdx.x == 0) {
        if (threadIdx.y == 0) {
            borrow_state = 0u;
        } else {
            borrow_state = borrow_prop[threadIdx.y - 1];
        }
    }
    r0_value = sub_cc(r0_value, borrow_state >> 1);
    r1_value = subc_cc(r1_value, 0);
}

__device__ __forceinline__ uint32_t load_shifted_word(
    const uint32_t * a_row,
    uint32_t out_idx,
    uint32_t L_a,
    uint32_t shift_words,
    uint32_t shift_bits
) {
    if (out_idx < shift_words) {
        return 0u;
    }

    const uint32_t src_lo = out_idx - shift_words;
    if (shift_bits == 0u) {
        return (src_lo < L_a) ? a_row[src_lo] : 0u;
    }

    const uint32_t lo = (src_lo < L_a) ? a_row[src_lo] : 0u;
    const uint32_t hi = (src_lo > 0u && src_lo - 1u < L_a) ? a_row[src_lo - 1u] : 0u;
    return (lo << shift_bits) | (hi >> (32u - shift_bits));
}

template<int THREADS>
__global__ void batch_shift_sub_naive_kernel(
    const uint32_t * A,
    const uint32_t * B,
    uint32_t * C,
    uint32_t shift_words,
    uint32_t shift_bits,
    uint32_t N,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t L_c,
    uint32_t stride_A,
    uint32_t stride_B,
    uint32_t stride_C
) {
    for (uint32_t idx0 = blockIdx.x * blockDim.x; idx0 < N; idx0 += gridDim.x * blockDim.x) {
        const uint32_t idx = idx0 + threadIdx.x;
        if (idx >= N) continue;

        const uint32_t * a_row = A + (size_t)idx * stride_A;
        const uint32_t * b_row = B + (size_t)idx * stride_B;
        uint32_t * c_row = C + (size_t)idx * stride_C;

        uint64_t borrow = 0;
        for (uint32_t i = 0; i < L_c; ++i) {
            const uint64_t a = load_shifted_word(a_row, i, L_a, shift_words, shift_bits);
            const uint64_t b = (i < L_b) ? b_row[i] : 0u;
            const uint64_t diff = a - b - borrow;
            c_row[i] = (uint32_t)diff;
            borrow = (b + borrow > a) ? 1u : 0u;
        }
    }
}

__global__ void batch_shift_sub_warp_kernel(
    const uint32_t * A,
    const uint32_t * B,
    uint32_t * C,
    uint32_t shift_words,
    uint32_t shift_bits,
    uint32_t N,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t L_c,
    uint32_t stride_A,
    uint32_t stride_B,
    uint32_t stride_C
) {
    const uint32_t lane = threadIdx.x;
    const uint32_t warps_per_block = blockDim.y;
    const uint32_t warp_global = blockIdx.x * warps_per_block + threadIdx.y;
    const uint32_t warp_stride = gridDim.x * warps_per_block;

    for (uint32_t idx = warp_global; idx < N; idx += warp_stride) {
        const uint32_t * a_row = A + (size_t)idx * stride_A;
        const uint32_t * b_row = B + (size_t)idx * stride_B;
        uint32_t * c_row = C + (size_t)idx * stride_C;

        uint32_t word = 0u;
        uint32_t borrow_state = 0u;
        if (lane < L_c) {
            const uint32_t a = load_shifted_word(a_row, lane, L_a, shift_words, shift_bits);
            const uint32_t b = (lane < L_b) ? b_row[lane] : 0u;
            word = a - b;
            borrow_state = (a < b) ? 2u : (word == 0u ? 1u : 0u);
        }

        for (int delta = 1; delta < 32; delta *= 2) {
            const uint32_t prev_state = __shfl_up_sync(0xffffffffu, borrow_state, delta, 32);
            if (lane >= (uint32_t)delta && borrow_state == 1u) {
                borrow_state = prev_state;
            }
        }

        uint32_t borrow_in = __shfl_up_sync(0xffffffffu, borrow_state, 1, 32);
        if (lane == 0) {
            borrow_in = 0u;
        }

        if (lane < L_c) {
            c_row[lane] = word - (borrow_in >> 1);
        }
        for (uint32_t i = 32u + lane; i < L_c; i += 32u) {
            c_row[i] = 0u;
        }
    }
}

__global__ void batch_shift_sub_single_block_kernel(
    const uint32_t * A,
    const uint32_t * B,
    uint32_t * C,
    uint32_t shift_words,
    uint32_t shift_bits,
    uint32_t N,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t L_c,
    uint32_t stride_A,
    uint32_t stride_B,
    uint32_t stride_C
) {
    __shared__ ushort2 borrow_info[32];
    __shared__ ushort block_borrow_shared;

    const uint32_t linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    const uint32_t threads_per_block = blockDim.x * blockDim.y;

    for (uint32_t idx = blockIdx.x; idx < N; idx += gridDim.x) {
        const uint32_t * a_row = A + (size_t)idx * stride_A;
        const uint32_t * b_row = B + (size_t)idx * stride_B;
        uint32_t * c_row = C + (size_t)idx * stride_C;
        if (linear_tid == 0) {
            block_borrow_shared = 0;
        }
        __syncthreads();

        for (uint32_t i0 = 0; i0 < L_c; i0 += threads_per_block * 2u) {
            const uint32_t i = i0 + linear_tid * 2u;
            uint32_t r0_value = load_shifted_word(a_row, i, L_a, shift_words, shift_bits);
            uint32_t r1_value = load_shifted_word(a_row, i + 1u, L_a, shift_words, shift_bits);
            const uint32_t c0_value = (i < L_b) ? b_row[i] : 0u;
            const uint32_t c1_value = (i + 1u < L_b) ? b_row[i + 1u] : 0u;
            ushort2 borrow;

            r0_value = sub_cc(r0_value, c0_value);
            r1_value = subc_cc(r1_value, c1_value);
            borrow.x = -subc(0, 0);
            sub_cc(r0_value, 2);
            subc_cc(r1_value, 0);
            if (subc(0, 0)) {
                borrow.y = 3u - (r0_value & 3u);
            } else {
                borrow.y = 0;
            }

            for (int delta = 1; delta < 32; delta *= 2) {
                const ushort2 borrow_prev = cuda::device::warp_shuffle_up<32, ushort2>(borrow, delta);
                if (threadIdx.x >= delta) {
                    borrow = combine_borrow_summary(borrow, borrow_prev);
                }
            }
            if (threadIdx.x == 31) {
                borrow_info[threadIdx.y] = borrow;
            }
            __syncthreads();

            if (threadIdx.y == 0) {
                ushort2 warp_borrow = (threadIdx.x < blockDim.y) ? borrow_info[threadIdx.x] : make_ushort2(0, 0);
                for (int delta = 1; delta < 32; delta *= 2) {
                    const ushort2 borrow_prev = cuda::device::warp_shuffle_up<32, ushort2>(warp_borrow, delta);
                    if (threadIdx.x < blockDim.y && threadIdx.x >= delta) {
                        warp_borrow = combine_borrow_summary(warp_borrow, borrow_prev);
                    }
                }
                if (threadIdx.x < blockDim.y) {
                    ushort compound = warp_borrow.y + block_borrow_shared;
                    warp_borrow.x += compound >> 2;
                    warp_borrow.y = 0;
                    borrow_info[threadIdx.x] = warp_borrow;
                }
            }
            __syncthreads();

            if (threadIdx.y < blockDim.y) {
                ushort compound = borrow.y + ((threadIdx.y > 0) ? borrow_info[threadIdx.y - 1].x : block_borrow_shared);
                borrow.x += compound >> 2;
                borrow = cuda::device::warp_shuffle_up<32, ushort2>(borrow, 1);
                if (threadIdx.x == 0) {
                    borrow.x = (threadIdx.y == 0) ? block_borrow_shared : borrow_info[threadIdx.y - 1].x;
                }
                r0_value = sub_cc(r0_value, (uint32_t)borrow.x);
                r1_value = subc_cc(r1_value, 0);
            }

            if (i < L_c) {
                c_row[i] = r0_value;
            }
            if (i + 1u < L_c) {
                c_row[i + 1u] = r1_value;
            }

            if (linear_tid == 0) {
                block_borrow_shared = borrow_info[blockDim.y - 1].x;
            }
            __syncthreads();
        }
    }
}

__global__ void batch_shift_sub_reduce_blocks_kernel(
    const uint32_t * A,
    const uint32_t * B,
    uint32_t * C,
    ushort2 * block_borrow_summary,
    uint32_t shift_words,
    uint32_t shift_bits,
    uint32_t N,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t L_c,
    uint32_t stride_A,
    uint32_t stride_B,
    uint32_t stride_C
) {
    __shared__ ushort2 borrow_info[32];
    const uint32_t linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    const uint32_t threads_per_block = blockDim.x * blockDim.y;
    const uint32_t blocks_per_num = (L_c + kChunkSize - 1u) / kChunkSize;

    for (uint32_t number_idx = blockIdx.y; number_idx < N; number_idx += gridDim.y) {
        const uint32_t * a_row = A + (size_t)number_idx * stride_A;
        const uint32_t * b_row = B + (size_t)number_idx * stride_B;
        uint32_t * c_base = C + (size_t)number_idx * stride_C;
        ushort2 * borrow_base = block_borrow_summary + (size_t)number_idx * blocks_per_num;

        for (uint32_t chunk_start = blockIdx.x * kChunkSize, chunk_idx = blockIdx.x;
             chunk_start < L_c;
             chunk_start += gridDim.x * kChunkSize, chunk_idx += gridDim.x) {
            const uint32_t chunk_len = min(kChunkSize, L_c - chunk_start);
            uint32_t * c_row = c_base + chunk_start;
            ushort2 block_summary = make_ushort2(0, 0);

            for (uint32_t i0 = 0; i0 < chunk_len; i0 += threads_per_block * 2u) {
                const uint32_t i = i0 + linear_tid * 2u;
                const uint32_t g0 = chunk_start + i;
                const uint32_t g1 = g0 + 1u;
                uint32_t r0_value = (i < chunk_len) ? load_shifted_word(a_row, g0, L_a, shift_words, shift_bits) : 0u;
                uint32_t r1_value = (i + 1u < chunk_len) ? load_shifted_word(a_row, g1, L_a, shift_words, shift_bits) : 0u;
                const uint32_t c0_value = (g0 < L_b && i < chunk_len) ? b_row[g0] : 0u;
                const uint32_t c1_value = (g1 < L_b && i + 1u < chunk_len) ? b_row[g1] : 0u;
                ushort2 borrow;

                r0_value = sub_cc(r0_value, c0_value);
                r1_value = subc_cc(r1_value, c1_value);
                borrow.x = -subc(0, 0);
                sub_cc(r0_value, 2);
                subc_cc(r1_value, 0);
                if (subc(0, 0)) {
                    borrow.y = 3u - (r0_value & 3u);
                } else {
                    borrow.y = 0;
                }

                for (int delta = 1; delta < 32; delta *= 2) {
                    const ushort2 borrow_prev = cuda::device::warp_shuffle_up<32, ushort2>(borrow, delta);
                    if (threadIdx.x >= delta) {
                        borrow = combine_borrow_summary(borrow, borrow_prev);
                    }
                }
                if (threadIdx.x == 31) {
                    borrow_info[threadIdx.y] = borrow;
                }
                __syncthreads();

                if (threadIdx.y == 0) {
                    ushort2 warp_borrow = (threadIdx.x < blockDim.y) ? borrow_info[threadIdx.x] : make_ushort2(0, 0);
                    for (int delta = 1; delta < 32; delta *= 2) {
                        const ushort2 borrow_prev = cuda::device::warp_shuffle_up<32, ushort2>(warp_borrow, delta);
                        if (threadIdx.x < blockDim.y && threadIdx.x >= delta) {
                            warp_borrow = combine_borrow_summary(warp_borrow, borrow_prev);
                        }
                    }
                    if (threadIdx.x < blockDim.y) {
                        borrow_info[threadIdx.x] = warp_borrow;
                    }
                }
                __syncthreads();

                if (threadIdx.y < blockDim.y) {
                    ushort compound = borrow.y + ((threadIdx.y > 0) ? borrow_info[threadIdx.y - 1].x : 0);
                    borrow.x += compound >> 2;
                }
                borrow = cuda::device::warp_shuffle_up<32, ushort2>(borrow, 1);
                if (threadIdx.x == 0) {
                    borrow.x = (threadIdx.y == 0) ? 0 : borrow_info[threadIdx.y - 1].x;
                }
                r0_value = sub_cc(r0_value, (uint32_t)borrow.x);
                r1_value = subc_cc(r1_value, 0);

                if (i < chunk_len) {
                    c_row[i] = r0_value;
                }
                if (i + 1u < chunk_len) {
                    c_row[i + 1u] = r1_value;
                }

                __syncthreads();
                if (linear_tid == 0) {
                    block_summary = borrow_info[blockDim.y - 1];
                }
                __syncthreads();
            }

            if (linear_tid == 0) {
                borrow_base[chunk_idx] = block_summary;
            }
            __syncthreads();
        }
    }
}

__global__ void batch_shift_sub_combine_blocks_kernel(ushort2 * block_borrow_summary, uint32_t N, uint32_t L_c) {
    __shared__ ushort2 borrow_info[32];

    const uint32_t blocks_per_num = (L_c + kChunkSize - 1u) / kChunkSize;
    ushort2 * row = block_borrow_summary + (size_t)blockIdx.x * blocks_per_num;

    for (uint32_t idx = blockIdx.x; idx < N; idx += gridDim.x) {
        ushort block_borrow = 0;

        for (uint32_t i0 = 0; i0 < blocks_per_num; i0 += 1024u) {
            const uint32_t i = i0 + threadIdx.y * 32u + threadIdx.x;
            ushort2 borrow = (i < blocks_per_num) ? row[i] : make_ushort2(0, 0);

            for (int delta = 1; delta < 32; delta *= 2) {
                const ushort2 borrow_prev = cuda::device::warp_shuffle_up<32, ushort2>(borrow, delta);
                if (threadIdx.x >= delta) {
                    borrow = combine_borrow_summary(borrow, borrow_prev);
                }
            }
            if (threadIdx.x == 31) {
                borrow_info[threadIdx.y] = borrow;
            }
            __syncthreads();

            if (threadIdx.y == 0) {
                ushort2 warp_borrow = borrow_info[threadIdx.x];
                for (int delta = 1; delta < 32; delta *= 2) {
                    const ushort2 borrow_prev = cuda::device::warp_shuffle_up<32, ushort2>(warp_borrow, delta);
                    if (threadIdx.x >= delta) {
                        warp_borrow = combine_borrow_summary(warp_borrow, borrow_prev);
                    }
                }
                ushort compound = warp_borrow.y + block_borrow;
                warp_borrow.x += compound >> 2;
                warp_borrow.y = 0;
                borrow_info[threadIdx.x] = warp_borrow;
            }
            __syncthreads();

            ushort compound = borrow.y + ((threadIdx.y > 0) ? borrow_info[threadIdx.y - 1].x : block_borrow);
            borrow.x += compound >> 2;
            borrow = cuda::device::warp_shuffle_up<32, ushort2>(borrow, 1);
            if (threadIdx.x == 0) {
                borrow.x = (threadIdx.y == 0) ? block_borrow : borrow_info[threadIdx.y - 1].x;
            }
            if (i < blocks_per_num) {
                row[i] = borrow;
            }

            block_borrow = borrow_info[31].x;
            __syncthreads();
        }
        row += (size_t)blocks_per_num * gridDim.x;
    }
}

__global__ void batch_shift_sub_apply_blocks_kernel(
    uint32_t * C,
    const ushort2 * block_borrow_summary,
    uint32_t N,
    uint32_t L_c,
    uint32_t stride_C
) {
    __shared__ uint32_t borrow_info[32];
    const uint32_t blocks_per_num = (L_c + kChunkSize - 1u) / kChunkSize;

    for (uint32_t number_idx = blockIdx.y; number_idx < N; number_idx += gridDim.y) {
        const ushort2 * borrow_base = block_borrow_summary + (size_t)number_idx * blocks_per_num;
        uint32_t * c_base = C + (size_t)number_idx * stride_C;

        for (uint32_t chunk_start = blockIdx.x * kChunkSize, chunk_idx = blockIdx.x;
             chunk_start < L_c;
             chunk_start += gridDim.x * kChunkSize, chunk_idx += gridDim.x) {
            const uint32_t borrow_in = borrow_base[chunk_idx].x;
            if (borrow_in == 0) {
                continue;
            }

            const uint32_t chunk_len = min(kChunkSize, L_c - chunk_start);
            uint32_t * c_row = c_base + chunk_start;

            for (uint32_t i0 = 0; i0 < chunk_len; i0 += blockDim.x * blockDim.y * 2u) {
                const uint32_t i = i0 + (threadIdx.y * 32u + threadIdx.x) * 2u;
                uint32_t r0_value = (i < chunk_len) ? c_row[i] : 0u;
                uint32_t r1_value = (i + 1u < chunk_len) ? c_row[i + 1u] : 0u;
                uint32_t c0_value = (i == 0 && i0 == 0) ? borrow_in : 0u;
                uint32_t c1_value = 0u;
                const uint32_t original0 = r0_value;
                const uint32_t original1 = r1_value;

                batch_shift_sub_64_all_warp(r0_value, r1_value, c0_value, c1_value, borrow_info);

                if (r0_value != original0 || r1_value != original1 || c0_value != 0u || c1_value != 0u) {
                    if (i < chunk_len) {
                        c_row[i] = r0_value;
                    }
                    if (i + 1u < chunk_len) {
                        c_row[i + 1u] = r1_value;
                    }
                }
                __syncthreads();
            }
        }
    }
}

}  // namespace

size_t batch_shift_sub_simple_workspace_size(
    uint32_t N,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t L_c,
    uint32_t shift
) {
    (void)L_a;
    (void)L_b;
    (void)shift;
    if (L_c < 32u || L_c <= kChunkSize || N >= 85u) {
        return 0;
    }
    const uint32_t blocks_per_num = (L_c + kChunkSize - 1u) / kChunkSize;
    return (size_t)N * blocks_per_num * sizeof(ushort2);
}

void batch_shift_sub_simple(
    const uint32_t * A,
    const uint32_t * B,
    uint32_t * C,
    uint32_t * workspace,
    uint32_t shift,
    uint32_t N,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t L_c,
    uint32_t stride_A,
    uint32_t stride_B,
    uint32_t stride_C
) {
    if (N == 0 || L_c == 0) {
        return;
    }

    const uint32_t shift_words = shift >> 5;
    const uint32_t shift_bits = shift & 31u;

    if (L_c < 32u) {
        if (L_c <= 4u) {
            const int threads_per_block = 32;
            const int num_blocks = std::min<uint32_t>((N + threads_per_block - 1u) / threads_per_block, 65536u);
            batch_shift_sub_naive_kernel<32><<<num_blocks, threads_per_block>>>(
                A, B, C, shift_words, shift_bits, N, L_a, L_b, L_c, stride_A, stride_B, stride_C
            );
        } else {
            const uint32_t warps_per_block = 8u;
            const uint32_t num_blocks = std::min<uint32_t>((N + warps_per_block - 1u) / warps_per_block, 65536u);
            batch_shift_sub_warp_kernel<<<num_blocks, dim3(32u, warps_per_block, 1u)>>>(
                A, B, C, shift_words, shift_bits, N, L_a, L_b, L_c, stride_A, stride_B, stride_C
            );
        }
        return;
    }

    if (L_c <= kChunkSize || N >= 85u || workspace == nullptr) {
        const uint32_t pair_count = (L_c + 1u) >> 1;
        const uint32_t threads_total = std::min<uint32_t>(pair_count == 0 ? 1u : pair_count, 1024u);
        const uint32_t warps_y = (threads_total + 31u) / 32u;
        const int num_blocks = std::min<uint32_t>(N, 65536u);
        batch_shift_sub_single_block_kernel<<<num_blocks, dim3(32u, warps_y, 1u)>>>(
            A, B, C, shift_words, shift_bits, N, L_a, L_b, L_c, stride_A, stride_B, stride_C
        );
    } else {
        ushort2 * block_borrow_summary = reinterpret_cast<ushort2 *>(workspace);
        const uint32_t blocks_per_num = (L_c + kChunkSize - 1u) / kChunkSize;
        const uint32_t num_blocks_x_limit = std::max<uint32_t>(256u, 65535u / std::max<uint32_t>(N, 1u));
        const uint32_t num_blocks_x = std::min<uint32_t>(num_blocks_x_limit, std::max<uint32_t>(1u, blocks_per_num));
        const uint32_t num_blocks_y = std::min<uint32_t>(N, std::max<uint32_t>(1u, 65535u / num_blocks_x));

        batch_shift_sub_reduce_blocks_kernel<<<dim3(num_blocks_x, num_blocks_y, 1u), dim3(32u, 32u, 1u)>>>(
            A, B, C, block_borrow_summary, shift_words, shift_bits, N, L_a, L_b, L_c, stride_A, stride_B, stride_C
        );
        batch_shift_sub_combine_blocks_kernel<<<std::min<uint32_t>(N, 65536u), dim3(32u, 32u, 1u)>>>(
            block_borrow_summary, N, L_c
        );
        batch_shift_sub_apply_blocks_kernel<<<dim3(num_blocks_x, num_blocks_y, 1u), dim3(32u, 32u, 1u)>>>(
            C, block_borrow_summary, N, L_c, stride_C
        );
    }
}
