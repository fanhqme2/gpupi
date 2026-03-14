#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>

#include "batch_bitlength.h"

namespace {

constexpr uint32_t kSmallLThreadThreshold = 8u;
constexpr uint32_t kLargeLThreshold = 2048u;
constexpr uint32_t kSmallNThreshold = 84u;
constexpr uint32_t kOptimisticWindowWords = 256u;
constexpr uint32_t kChunkSize = 4096u;

__device__ __forceinline__ uint32_t bitlength_word(uint32_t word) {
    return word == 0u ? 0u : 32u - (uint32_t)__clz(word);
}

template<int THREADS>
__global__ void batch_bitlength_thread_kernel(
    const uint32_t * A,
    uint32_t * result,
    uint32_t N,
    uint32_t L,
    uint32_t stride_A
) {
    uint32_t local_max = 0u;
    for (uint32_t idx0 = blockIdx.x * blockDim.x; idx0 < N; idx0 += gridDim.x * blockDim.x) {
        const uint32_t idx = idx0 + threadIdx.x;
        if (idx >= N) continue;

        const uint32_t * row = A + (size_t)idx * stride_A;
        uint32_t local = 0u;
        for (uint32_t offset = 0; offset < L; ++offset) {
            const uint32_t i = L - 1u - offset;
            const uint32_t word = row[i];
            if (word != 0u) {
                local = i * 32u + bitlength_word(word);
                break;
            }
        }
        local_max = max(local_max, local);
    }
    atomicMax(result, local_max);
}

__global__ void batch_bitlength_warp_kernel(
    const uint32_t * A,
    uint32_t * result,
    uint32_t N,
    uint32_t L,
    uint32_t stride_A
) {
    const uint32_t lane = threadIdx.x;
    const uint32_t warps_per_block = blockDim.y;
    const uint32_t warp_global = blockIdx.x * warps_per_block + threadIdx.y;
    const uint32_t warp_stride = gridDim.x * warps_per_block;
    uint32_t local_max = 0u;
    for (uint32_t idx = warp_global; idx < N; idx += warp_stride) {
        const uint32_t * row = A + (size_t)idx * stride_A;
        uint32_t local = 0u;
        const int64_t top_base = ((int64_t)L - 1) & ~31ll;
        for (int64_t base = top_base; base >= 0; base -= 32) {
            const uint32_t i = (uint32_t)base + lane;
            const uint32_t word = (i < L) ? row[i] : 0u;
            const uint32_t mask = __ballot_sync(0xffffffffu, word != 0u);
            if (mask != 0u) {
                const uint32_t top_lane = 31u - (uint32_t)__clz(mask);
                if (lane == top_lane) {
                    local = i * 32u + bitlength_word(word);
                }
                break;
            }
        }
        local_max = max(local_max, local);
    }
    for (int delta = 1; delta < 32; delta *= 2) {
        local_max = max(local_max, __shfl_down_sync(0xffffffffu, local_max, delta));
    }
    if (lane == 0) {
        atomicMax(result, local_max);
    }
}

__global__ void batch_bitlength_optimistic_kernel(
    const uint32_t * A,
    uint32_t * result,
    uint32_t * unresolved,
    uint32_t * unresolved_count,
    uint32_t N,
    uint32_t L,
    uint32_t stride_A
) {
    const uint32_t lane = threadIdx.x;
    const uint32_t warps_per_block = blockDim.y;
    const uint32_t warp_global = blockIdx.x * warps_per_block + threadIdx.y;
    const uint32_t warp_stride = gridDim.x * warps_per_block;
    const uint32_t optimistic_begin = (L > kOptimisticWindowWords) ? (L - kOptimisticWindowWords) : 0u;

    for (uint32_t idx = warp_global; idx < N; idx += warp_stride) {
        const uint32_t * row = A + (size_t)idx * stride_A;
        uint32_t local = 0u;
        const int64_t top_base = ((int64_t)L - 1) & ~31ll;
        const int64_t optimistic_base = ((int64_t)optimistic_begin) & ~31ll;
        for (int64_t base = top_base; base >= optimistic_base; base -= 32) {
            const uint32_t i = (uint32_t)base + lane;
            const uint32_t word = (i < L) ? row[i] : 0u;
            const uint32_t mask = __ballot_sync(0xffffffffu, word != 0u);
            if (mask != 0u) {
                const uint32_t top_lane = 31u - (uint32_t)__clz(mask);
                if (lane == top_lane) {
                    local = i * 32u + bitlength_word(word);
                }
                break;
            }
        }
        const uint32_t found = __any_sync(0xffffffffu, local != 0u) ? 1u : 0u;
        if (local != 0u) {
            atomicMax(result, local);
        }
        if (lane == 0) {
            if (found != 0u) {
                unresolved[idx] = 0u;
            } else {
                unresolved[idx] = 1u;
                atomicAdd(unresolved_count, 1u);
            }
        }
    }
}

template<int THREADS>
__global__ void batch_bitlength_chunk_kernel(
    const uint32_t * A,
    uint32_t * result,
    const uint32_t * unresolved,
    uint32_t N,
    uint32_t L,
    uint32_t stride_A
) {
    __shared__ uint32_t shared_best[THREADS];

    const uint32_t idx = blockIdx.y;
    if (idx >= N || unresolved[idx] == 0u) {
        return;
    }

    const uint32_t * row = A + (size_t)idx * stride_A;
    uint32_t local_best = 0u;
    const uint32_t chunk_count = (L + kChunkSize - 1u) / kChunkSize;

    for (uint32_t chunk_idx = blockIdx.x; chunk_idx < chunk_count; chunk_idx += gridDim.x) {
        const uint32_t chunk_start = chunk_idx * kChunkSize;
        const uint32_t chunk_end = min(chunk_start + kChunkSize, L);

        for (uint32_t i = chunk_start + threadIdx.x; i < chunk_end; i += THREADS) {
            const uint32_t word = row[i];
            if (word != 0u) {
                local_best = max(local_best, i * 32u + bitlength_word(word));
            }
        }
    }

    shared_best[threadIdx.x] = local_best;
    __syncthreads();

    for (int stride = THREADS / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < (uint32_t)stride) {
            shared_best[threadIdx.x] = max(shared_best[threadIdx.x], shared_best[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && shared_best[0] != 0u) {
        atomicMax(result, shared_best[0]);
    }
}

}  // namespace

size_t batch_bitlength_workspace_size(uint32_t N, uint32_t L) {
    if (N == 0 || L == 0) {
        return 0;
    }
    if (L < kLargeLThreshold || N >= kSmallNThreshold) {
        return 0;
    }
    return (size_t)(N + 1u) * sizeof(uint32_t);
}

uint32_t batch_bitlength_max(
    const uint32_t * A,
    uint32_t * workspace,
    uint32_t N,
    uint32_t L,
    uint32_t stride_A
) {
    if (N == 0 || L == 0) {
        return 0u;
    }

    uint32_t * d_result = nullptr;
    cudaMalloc(&d_result, sizeof(uint32_t));
    cudaMemset(d_result, 0, sizeof(uint32_t));

    if (L <= kSmallLThreadThreshold) {
        const int threads_per_block = 128;
        const int num_blocks = std::min<uint32_t>((N + threads_per_block - 1u) / threads_per_block, 65535u);
        batch_bitlength_thread_kernel<128><<<num_blocks, threads_per_block>>>(
            A, d_result, N, L, stride_A
        );
    } else if (L < kLargeLThreshold || N >= kSmallNThreshold || workspace == nullptr) {
        const uint32_t warps_per_block = 8u;
        const uint32_t num_blocks = std::min<uint32_t>((N + warps_per_block - 1u) / warps_per_block, 65535u);
        batch_bitlength_warp_kernel<<<num_blocks, dim3(32u, warps_per_block, 1u)>>>(
            A, d_result, N, L, stride_A
        );
    } else {
        uint32_t * unresolved = workspace;
        uint32_t * unresolved_count = workspace + N;
        cudaMemset(workspace, 0, (size_t)(N + 1u) * sizeof(uint32_t));

        const uint32_t warps_per_block = 8u;
        const uint32_t optimistic_blocks = std::min<uint32_t>((N + warps_per_block - 1u) / warps_per_block, 65535u);
        batch_bitlength_optimistic_kernel<<<optimistic_blocks, dim3(32u, warps_per_block, 1u)>>>(
            A, d_result, unresolved, unresolved_count, N, L, stride_A
        );

        uint32_t h_unresolved_count = 0u;
        cudaMemcpy(&h_unresolved_count, unresolved_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (h_unresolved_count != 0u) {
            const uint32_t chunk_count = (L + kChunkSize - 1u) / kChunkSize;
            const uint32_t num_blocks_x_limit = std::max<uint32_t>(256u, 65535u / std::max<uint32_t>(N, 1u));
            const uint32_t num_blocks_x = std::min<uint32_t>(num_blocks_x_limit, std::max<uint32_t>(1u, chunk_count));
            const uint32_t num_blocks_y = std::min<uint32_t>(N, std::max<uint32_t>(1u, 65535u / num_blocks_x));
            batch_bitlength_chunk_kernel<256><<<dim3(num_blocks_x, num_blocks_y, 1u), 256>>>(
                A, d_result, unresolved, N, L, stride_A
            );
        }
    }

    uint32_t h_result = 0u;
    cudaMemcpy(&h_result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    return h_result;
}
