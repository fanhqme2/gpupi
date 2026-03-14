#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/warp>

#include <algorithm>

#include "batch_add.h"
#include "batch_mul_addsub_asm.h"
#include "batch_mul_addsub_warp.h"

namespace {

constexpr uint32_t kChunkSize = 2048;

__host__ __device__ __forceinline__ uint32_t batch_add_effective_len(uint32_t L_a, uint32_t L_b, uint32_t L_c) {
    const uint32_t sum_len = (L_a > L_b ? L_a : L_b) + 1u;
    return (L_c < sum_len) ? L_c : sum_len;
}

__device__ __forceinline__ ushort2 combine_carry_summary(ushort2 a, ushort2 b) {
    ushort compound = a.y + b.x;
    a.x += compound >> 2;
    if ((compound & 3) == 3) {
        a.y = b.y;
    } else {
        a.y = 0;
    }
    return a;
}

template<int THREADS>
__global__ void batch_add_naive_kernel(
    const uint32_t * A,
    const uint32_t * B,
    uint32_t * C,
    uint32_t N,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t L_c,
    uint32_t calc_len,
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

        uint64_t carry = 0;
        for (uint32_t i = 0; i < calc_len; ++i) {
            const uint64_t a = (i < L_a) ? a_row[i] : 0u;
            const uint64_t b = (i < L_b) ? b_row[i] : 0u;
            const uint64_t sum = a + b + carry;
            c_row[i] = (uint32_t)sum;
            carry = sum >> 32;
        }
        for (uint32_t i = calc_len; i < L_c; ++i) {
            c_row[i] = 0u;
        }
    }
}

__global__ void batch_add_zero_fill_kernel(
    uint32_t * C,
    uint32_t N,
    uint32_t fill_start,
    uint32_t L_c,
    uint32_t stride_C
) {
    for (uint32_t idx = blockIdx.y; idx < N; idx += gridDim.y) {
        uint32_t * c_row = C + (size_t)idx * stride_C;
        for (uint32_t i = fill_start + blockIdx.x * blockDim.x + threadIdx.x; i < L_c; i += gridDim.x * blockDim.x) {
            c_row[i] = 0u;
        }
    }
}

__global__ void batch_add_single_block_kernel(
    const uint32_t * A,
    const uint32_t * B,
    uint32_t * C,
    uint32_t N,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t calc_len,
    uint32_t stride_A,
    uint32_t stride_B,
    uint32_t stride_C
) {
    __shared__ ushort2 carry_info[32];

    const uint32_t linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    const uint32_t threads_per_block = blockDim.x * blockDim.y;

    for (uint32_t idx = blockIdx.x; idx < N; idx += gridDim.x) {
        const uint32_t * a_row = A + (size_t)idx * stride_A;
        const uint32_t * b_row = B + (size_t)idx * stride_B;
        uint32_t * c_row = C + (size_t)idx * stride_C;
        ushort block_carry = 0;

        for (uint32_t i0 = 0; i0 < calc_len; i0 += threads_per_block * 2u) {
            const uint32_t i = i0 + linear_tid * 2u;
            uint32_t r0_value = (i < L_a) ? a_row[i] : 0u;
            uint32_t r1_value = (i + 1u < L_a) ? a_row[i + 1u] : 0u;
            const uint32_t c0_value = (i < L_b) ? b_row[i] : 0u;
            const uint32_t c1_value = (i + 1u < L_b) ? b_row[i + 1u] : 0u;
            ushort2 carry;

            r0_value = add_cc(r0_value, c0_value);
            r1_value = addc_cc(r1_value, c1_value);
            carry.x = addc(0, 0);
            add_cc(r0_value, 2);
            addc_cc(r1_value, 0);
            if (addc(0, 0)) {
                carry.y = r0_value & 3u;
            } else {
                carry.y = 0;
            }

            for (int delta = 1; delta < 32; delta *= 2) {
                const ushort2 carry_prev = cuda::device::warp_shuffle_up<32, ushort2>(carry, delta);
                if (threadIdx.x >= delta) {
                    carry = combine_carry_summary(carry, carry_prev);
                }
            }
            if (threadIdx.x == 31) {
                carry_info[threadIdx.y] = carry;
            }
            __syncthreads();

            if (threadIdx.y == 0) {
                ushort2 warp_carry = (threadIdx.x < blockDim.y) ? carry_info[threadIdx.x] : make_ushort2(0, 0);
                for (int delta = 1; delta < 32; delta *= 2) {
                    const ushort2 carry_prev = cuda::device::warp_shuffle_up<32, ushort2>(warp_carry, delta);
                    if (threadIdx.x < blockDim.y && threadIdx.x >= delta) {
                        warp_carry = combine_carry_summary(warp_carry, carry_prev);
                    }
                }
                if (threadIdx.x < blockDim.y) {
                    ushort compound = warp_carry.y + block_carry;
                    warp_carry.x += compound >> 2;
                    warp_carry.y = 0;
                    carry_info[threadIdx.x] = warp_carry;
                }
            }
            __syncthreads();

            if (threadIdx.y < blockDim.y) {
                ushort compound = carry.y + ((threadIdx.y > 0) ? carry_info[threadIdx.y - 1].x : block_carry);
                carry.x += compound >> 2;
                carry = cuda::device::warp_shuffle_up<32, ushort2>(carry, 1);
                if (threadIdx.x == 0) {
                    carry.x = (threadIdx.y == 0) ? block_carry : carry_info[threadIdx.y - 1].x;
                }
                r0_value = add_cc(r0_value, (uint32_t)carry.x);
                r1_value = addc_cc(r1_value, 0);
            }

            if (i < calc_len) {
                c_row[i] = r0_value;
            }
            if (i + 1u < calc_len) {
                c_row[i + 1u] = r1_value;
            }

            if (linear_tid == 0) {
                block_carry = carry_info[blockDim.y - 1].x;
            }
            __syncthreads();
        }
    }
}

__global__ void batch_add_reduce_blocks_kernel(
    const uint32_t * A,
    const uint32_t * B,
    uint32_t * C,
    ushort2 * block_carry_summary,
    uint32_t N,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t calc_len,
    uint32_t stride_A,
    uint32_t stride_B,
    uint32_t stride_C
) {
    __shared__ ushort2 carry_info[32];

    const uint32_t chunk_idx = blockIdx.x;
    const uint32_t number_idx = blockIdx.y;
    const uint32_t chunk_start = chunk_idx * kChunkSize;
    if (number_idx >= N || chunk_start >= calc_len) return;

    const uint32_t chunk_len = min(kChunkSize, calc_len - chunk_start);
    const uint32_t linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    const uint32_t threads_per_block = blockDim.x * blockDim.y;

    const uint32_t * a_row = A + (size_t)number_idx * stride_A;
    const uint32_t * b_row = B + (size_t)number_idx * stride_B;
    uint32_t * c_row = C + (size_t)number_idx * stride_C + chunk_start;

    ushort2 block_summary = make_ushort2(0, 0);

    for (uint32_t i0 = 0; i0 < chunk_len; i0 += threads_per_block * 2u) {
        const uint32_t i = i0 + linear_tid * 2u;
        const uint32_t g0 = chunk_start + i;
        const uint32_t g1 = g0 + 1u;
        uint32_t r0_value = (g0 < L_a && i < chunk_len) ? a_row[g0] : 0u;
        uint32_t r1_value = (g1 < L_a && i + 1u < chunk_len) ? a_row[g1] : 0u;
        const uint32_t c0_value = (g0 < L_b && i < chunk_len) ? b_row[g0] : 0u;
        const uint32_t c1_value = (g1 < L_b && i + 1u < chunk_len) ? b_row[g1] : 0u;
        ushort2 carry;

        r0_value = add_cc(r0_value, c0_value);
        r1_value = addc_cc(r1_value, c1_value);
        carry.x = addc(0, 0);
        add_cc(r0_value, 2);
        addc_cc(r1_value, 0);
        if (addc(0, 0)) {
            carry.y = r0_value & 3u;
        } else {
            carry.y = 0;
        }

        for (int delta = 1; delta < 32; delta *= 2) {
            const ushort2 carry_prev = cuda::device::warp_shuffle_up<32, ushort2>(carry, delta);
            if (threadIdx.x >= delta) {
                carry = combine_carry_summary(carry, carry_prev);
            }
        }
        if (threadIdx.x == 31) {
            carry_info[threadIdx.y] = carry;
        }
        __syncthreads();

        if (threadIdx.y == 0) {
            ushort2 warp_carry = (threadIdx.x < blockDim.y) ? carry_info[threadIdx.x] : make_ushort2(0, 0);
            for (int delta = 1; delta < 32; delta *= 2) {
                const ushort2 carry_prev = cuda::device::warp_shuffle_up<32, ushort2>(warp_carry, delta);
                if (threadIdx.x < blockDim.y && threadIdx.x >= delta) {
                    warp_carry = combine_carry_summary(warp_carry, carry_prev);
                }
            }
            if (threadIdx.x < blockDim.y) {
                carry_info[threadIdx.x] = warp_carry;
            }
        }
        __syncthreads();

        carry = cuda::device::warp_shuffle_up<32, ushort2>(carry, 1);
        if (threadIdx.x == 0) {
            carry.x = (threadIdx.y == 0) ? 0 : carry_info[threadIdx.y - 1].x;
        }
        r0_value = add_cc(r0_value, (uint32_t)carry.x);
        r1_value = addc_cc(r1_value, 0);

        if (i < chunk_len) {
            c_row[i] = r0_value;
        }
        if (i + 1u < chunk_len) {
            c_row[i + 1u] = r1_value;
        }

        __syncthreads();
        if (linear_tid == 0) {
            block_summary = carry_info[blockDim.y - 1];
        }
        __syncthreads();
    }

    if (linear_tid == 0) {
        const uint32_t blocks_per_num = (calc_len + kChunkSize - 1u) / kChunkSize;
        block_carry_summary[(size_t)number_idx * blocks_per_num + chunk_idx] = block_summary;
    }
}

__global__ void batch_add_combine_blocks_kernel(ushort2 * block_carry_summary, uint32_t N, uint32_t calc_len) {
    __shared__ ushort2 carry_info[32];

    const uint32_t blocks_per_num = (calc_len + kChunkSize - 1u) / kChunkSize;
    ushort2 * row = block_carry_summary + (size_t)blockIdx.x * blocks_per_num;

    for (uint32_t idx = blockIdx.x; idx < N; idx += gridDim.x) {
        ushort block_carry = 0;
        for (uint32_t i0 = 0; i0 < blocks_per_num; i0 += 1024u) {
            const uint32_t i = i0 + threadIdx.y * 32u + threadIdx.x;
            ushort2 carry = (i < blocks_per_num) ? row[i] : make_ushort2(0, 0);

            for (int delta = 1; delta < 32; delta *= 2) {
                const ushort2 carry_prev = cuda::device::warp_shuffle_up<32, ushort2>(carry, delta);
                if (threadIdx.x >= delta) {
                    carry = combine_carry_summary(carry, carry_prev);
                }
            }
            if (threadIdx.x == 31) {
                carry_info[threadIdx.y] = carry;
            }
            __syncthreads();

            if (threadIdx.y == 0) {
                ushort2 warp_carry = carry_info[threadIdx.x];
                for (int delta = 1; delta < 32; delta *= 2) {
                    const ushort2 carry_prev = cuda::device::warp_shuffle_up<32, ushort2>(warp_carry, delta);
                    if (threadIdx.x >= delta) {
                        warp_carry = combine_carry_summary(warp_carry, carry_prev);
                    }
                }
                ushort compound = warp_carry.y + block_carry;
                warp_carry.x += compound >> 2;
                warp_carry.y = 0;
                carry_info[threadIdx.x] = warp_carry;
            }
            __syncthreads();

            ushort compound = carry.y + ((threadIdx.y > 0) ? carry_info[threadIdx.y - 1].x : block_carry);
            carry.x += compound >> 2;
            carry = cuda::device::warp_shuffle_up<32, ushort2>(carry, 1);
            if (threadIdx.x == 0) {
                carry.x = (threadIdx.y == 0) ? block_carry : carry_info[threadIdx.y - 1].x;
            }
            if (i < blocks_per_num) {
                row[i] = carry;
            }

            block_carry = carry_info[31].x;
            __syncthreads();
        }
        row += (size_t)blocks_per_num * gridDim.x;
    }
}

__global__ void batch_add_apply_blocks_kernel(
    uint32_t * C,
    const ushort2 * block_carry_summary,
    uint32_t N,
    uint32_t calc_len,
    uint32_t stride_C
) {
    __shared__ uint32_t carry_info[32];

    const uint32_t chunk_idx = blockIdx.x;
    const uint32_t number_idx = blockIdx.y;
    const uint32_t chunk_start = chunk_idx * kChunkSize;
    if (number_idx >= N || chunk_start >= calc_len) return;

    const uint32_t blocks_per_num = (calc_len + kChunkSize - 1u) / kChunkSize;
    const uint32_t carry_in = block_carry_summary[(size_t)number_idx * blocks_per_num + chunk_idx].x;
    if (carry_in == 0) return;

    const uint32_t chunk_len = min(kChunkSize, calc_len - chunk_start);
    uint32_t * c_row = C + (size_t)number_idx * stride_C + chunk_start;

    for (uint32_t i0 = 0; i0 < chunk_len; i0 += blockDim.x * blockDim.y * 2u) {
        const uint32_t i = i0 + (threadIdx.y * 32u + threadIdx.x) * 2u;
        uint32_t r0_value = (i < chunk_len) ? c_row[i] : 0u;
        uint32_t r1_value = (i + 1u < chunk_len) ? c_row[i + 1u] : 0u;
        uint32_t c0_value = (i == 0 && i0 == 0) ? carry_in : 0u;
        uint32_t c1_value = 0u;
        const uint32_t original = r0_value;

        batch_mul_add_64_all_warp<32>(r0_value, r1_value, c0_value, c1_value, carry_info);

        if (r0_value != original || c0_value != 0u) {
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

}  // namespace

size_t batch_add_simple_workspace_size(
    uint32_t N,
    uint32_t L_a,
    uint32_t L_b,
    uint32_t L_c
) {
    const uint32_t calc_len = batch_add_effective_len(L_a, L_b, L_c);
    if (L_c < 32u || calc_len <= kChunkSize || N >= 85u) {
        return 0;
    }
    const uint32_t blocks_per_num = (calc_len + kChunkSize - 1u) / kChunkSize;
    return (size_t)N * blocks_per_num * sizeof(ushort2);
}

void batch_add_simple(
    uint32_t * A,
    uint32_t * B,
    uint32_t * C,
    uint32_t * workspace,
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

    const uint32_t calc_len = batch_add_effective_len(L_a, L_b, L_c);
    if (L_c < 32u) {
        const int threads_per_block = 32;
        const int num_blocks = std::min<uint32_t>((N + threads_per_block - 1u) / threads_per_block, 65536u);
        batch_add_naive_kernel<32><<<num_blocks, threads_per_block>>>(
            A, B, C, N, L_a, L_b, L_c, calc_len, stride_A, stride_B, stride_C
        );
        return;
    }

    if (calc_len > 0 && (calc_len <= kChunkSize || N >= 85u || workspace == nullptr)) {
        const uint32_t pair_count = (calc_len + 1u) >> 1;
        const uint32_t threads_total = std::min<uint32_t>(pair_count == 0 ? 1u : pair_count, 1024u);
        const uint32_t warps_y = (threads_total + 31u) / 32u;
        const int num_blocks = std::min<uint32_t>(N, 65536u);
        batch_add_single_block_kernel<<<num_blocks, dim3(32u, warps_y, 1u)>>>(
            A, B, C, N, L_a, L_b, calc_len, stride_A, stride_B, stride_C
        );
    } else if (calc_len > 0) {
        ushort2 * block_carry_summary = reinterpret_cast<ushort2 *>(workspace);
        const uint32_t blocks_per_num = (calc_len + kChunkSize - 1u) / kChunkSize;
        const uint32_t num_blocks_x = std::min<uint32_t>(256u, std::max<uint32_t>(1u, blocks_per_num));
        const uint32_t num_blocks_y = std::min<uint32_t>(N, 65535u / num_blocks_x + 1u);

        batch_add_reduce_blocks_kernel<<<dim3(num_blocks_x, num_blocks_y, 1u), dim3(32u, 32u, 1u)>>>(
            A, B, C, block_carry_summary, N, L_a, L_b, calc_len, stride_A, stride_B, stride_C
        );
        batch_add_combine_blocks_kernel<<<std::min<uint32_t>(N, 65536u), dim3(32u, 32u, 1u)>>>(
            block_carry_summary, N, calc_len
        );
        batch_add_apply_blocks_kernel<<<dim3(num_blocks_x, num_blocks_y, 1u), dim3(32u, 32u, 1u)>>>(
            C, block_carry_summary, N, calc_len, stride_C
        );
    }

    if (L_c > calc_len) {
        const uint32_t num_blocks_x = std::min<uint32_t>(256u, std::max<uint32_t>(1u, (L_c - calc_len + 255u) / 256u));
        const uint32_t num_blocks_y = std::min<uint32_t>(N, 65535u / num_blocks_x + 1u);
        batch_add_zero_fill_kernel<<<dim3(num_blocks_x, num_blocks_y, 1u), 256>>>(
            C, N, calc_len, L_c, stride_C
        );
    }
}
